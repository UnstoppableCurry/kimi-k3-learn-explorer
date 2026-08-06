/* pipeline.js — Kimi K3 整体流水线：一个 token 穿越整个网络的完整旅程
 *
 * 设计（乔布斯式）：一条水平流水线，从左到右 ——
 *   输入文字 → 分词 → Embedding → [Layer ×4（代表 61 层）] → …×57 → 输出头 → 采样
 * 每个 Layer 都是一座有真实内部结构的「工作站」，不是空盒子：
 *   前舱 MLA 注意力：Q/K/V 向量 + 8×8 注意力权重热力矩阵 + 加权输出向量
 *   后舱 MoE：router top-16 权重条 + 896 专家墙 + 专家 FFN（输入→隐藏→输出）+ 加权聚合
 * 所有数字都是真算的：embedding 查表、注意力 softmax、896 专家真打分取 top-16、
 * logits 真点积、softmax + top-p 真采样。教学维度 d=8（K3 实际 d_model≈7168）。
 *
 * 接口：PIPELINE.init(scene) / update(delta) / play() / pause() / restart()
 *        setFollow(bool) / notifyUserDrag() / debugGo(stageIndex)
 * 相机跟随通过 CORE_CAMERA.flyTo 实现；DOM 字幕写入 #caption / #stageTitle。
 */
window.PIPELINE = (function () {
  'use strict';

  // ============================================================
  // 1. 真实数学（玩具维度 d=8，每个数字都是真前向传播）
  // ============================================================
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { return (r() + r() + r() + r() - 2) * 1.2; }

  var SENT = '月之暗面发布K3';
  var IDS = [48213, 20047, 5531, 39196, 8873, 610, 74, 102931];
  var D = 8, EXPERTS = 896, TOPK = 16, LAYERS_SHOWN = 4, LAYERS_TOTAL = 61;
  var TEMP = 1.0, TOP_P = 0.95;

  var embedCache = {};
  function embed(id) {
    if (!embedCache[id]) {
      var r = rng(id * 7919), v = [];
      for (var i = 0; i < D; i++) v.push(gauss(r));
      embedCache[id] = v;
    }
    return embedCache[id];
  }
  // 896 个专家的权重向量（真打分用）
  var expW = [];
  (function () {
    var r = rng(20260716);
    for (var e = 0; e < EXPERTS; e++) {
      var v = [];
      for (var i = 0; i < D; i++) v.push(gauss(r) * 0.6);
      expW.push(v);
    }
  })();
  // 输出投影：11 个候选字
  var OUT_CH = ['的', '是', '了', '在', '和', '开', '源', '模', '型', '。', '，'];
  var outW = OUT_CH.map(function (ch, i) {
    var r = rng(IDS[i % IDS.length] * 31 + 7), v = [];
    for (var j = 0; j < D; j++) v.push(gauss(r));
    return { ch: ch, w: v };
  });

  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function softmax(z) {
    var m = Math.max.apply(null, z), ex = z.map(function (v) { return Math.exp(v - m); });
    var s = ex.reduce(function (a, b) { return a + b; }, 0);
    return ex.map(function (v) { return v / s; });
  }
  function fmt(v, n) { return (v >= 0 ? ' ' : '') + v.toFixed(n == null ? 2 : n); }
  function vecShort(v, n) {
    return '[' + v.slice(0, n || 4).map(function (x) { return fmt(x, 1); }).join(', ') + ', …]';
  }

  // 一层的前向（真实链式计算：每层的输出是下一层的输入）
  function layerForward(x, ctx) {
    // MLA 注意力：q=x，对上下文真 softmax
    var scores = ctx.map(function (k) { return dot(x, k) / Math.sqrt(D); });
    var attn = softmax(scores);
    var xa = x.slice();
    for (var i = 0; i < ctx.length; i++) {
      for (var d = 0; d < D; d++) xa[d] += 0.5 * attn[i] * ctx[i][d];
    }
    // MoE 门控：896 专家真打分 → top-16 → 归一化
    var gs = [];
    for (var e = 0; e < EXPERTS; e++) gs.push(dot(xa, expW[e]));
    var idx = gs.map(function (s, i2) { return i2; }).sort(function (a, b) { return gs[b] - gs[a]; });
    var top = idx.slice(0, TOPK);
    var w = softmax(top.map(function (i3) { return gs[i3]; }));
    // 专家聚合 + 残差
    var y = [0, 0, 0, 0, 0, 0, 0, 0];
    top.forEach(function (ei, j) {
      var act = Math.tanh(dot(xa, expW[ei]));
      for (var d2 = 0; d2 < D; d2++) y[d2] += w[j] * act * expW[ei][d2];
    });
    var out = xa.map(function (v, d3) { return v + y[d3]; });
    return { attn: attn, xa: xa, top: top, w: w, y: y, out: out };
  }

  // 专家 FFN 教学展开：输入 → 隐藏层(16) → 加权输出（全部来自真实权重）
  function expertFFN(xa, ei, wi) {
    var act = Math.tanh(dot(xa, expW[ei]));
    var hid = [];
    for (var j = 0; j < 16; j++) hid.push(Math.tanh(xa[j % D] * expW[ei][j % D] * 2.5) * Math.abs(act));
    var out = expW[ei].map(function (v) { return v * act * wi; });
    return { act: act, hid: hid, out: out };
  }

  // 完整旅程：主角 token 是 SENT[step]，链式穿过 4 个展示层
  function computeJourney(step) {
    var ctx = IDS.slice(0, step + 1).map(embed);
    var layers = [], cur = embed(IDS[step]);
    var x0 = cur.slice();
    for (var l = 0; l < LAYERS_SHOWN; l++) {
      var r = layerForward(cur, ctx);
      layers.push(r);
      cur = r.out;
    }
    var logits = outW.map(function (o) { return dot(cur, o.w); });
    // 展示归一化：缩放到 max|logit|=3，让 softmax 分布可分辨（不改变相对顺序）
    var maxAbs = Math.max.apply(null, logits.map(Math.abs)) || 1;
    logits = logits.map(function (v) { return v / maxAbs * 3; });
    var probs = softmax(logits.map(function (v) { return v / TEMP; }));
    // top-p 核采样
    var order = probs.map(function (v, i) { return i; }).sort(function (a, b) { return probs[b] - probs[a]; });
    var keep = {}, acc = 0;
    for (var k = 0; k < order.length; k++) {
      acc += probs[order[k]]; keep[order[k]] = true;
      if (acc >= TOP_P) break;
    }
    return { step: step, ctx: ctx, x0: x0, layers: layers, final: cur, logits: logits, probs: probs, keep: keep, order: order };
  }

  // ============================================================
  // 2. 文本 / 面板绘制（CanvasTexture）
  // ============================================================
  function canvasTex(c) {
    var tex = new THREE.CanvasTexture(c);
    if (THREE.SRGBColorSpace) tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  // 单行文字 Sprite，世界高度 hWorld
  function makeLabel(text, opts) {
    opts = opts || {};
    var size = opts.fontSize || 44;
    var font = (opts.mono ? 'Menlo, monospace' : '-apple-system, sans-serif');
    var c = document.createElement('canvas');
    var g = c.getContext('2d');
    g.font = '600 ' + size + 'px ' + font;
    var tw = Math.ceil(g.measureText(text).width);
    c.width = tw + 24; c.height = size + 24;
    g = c.getContext('2d');
    g.font = '600 ' + size + 'px ' + font;
    g.textBaseline = 'middle';
    if (opts.bg) { g.fillStyle = opts.bg; g.fillRect(0, 0, c.width, c.height); }
    g.fillStyle = opts.color || '#f2eee6';
    g.fillText(text, 12, c.height / 2);
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({
      map: canvasTex(c), transparent: true, depthWrite: false, opacity: opts.opacity == null ? 1 : opts.opacity
    }));
    var h = opts.h || 0.8;
    sp.scale.set(h * c.width / c.height, h, 1);
    return sp;
  }

  // 多行数字面板 Sprite（等宽字体，左侧色条）
  function makePanel(lines, wWorld, accent) {
    var c = document.createElement('canvas');
    c.width = 660; c.height = 52 * lines.length + 20;
    var g = c.getContext('2d');
    g.fillStyle = 'rgba(14,17,28,0.96)'; g.fillRect(0, 0, c.width, c.height);
    g.fillStyle = accent || '#4a4036'; g.fillRect(0, 0, 8, c.height);
    g.font = '22px Menlo, monospace'; g.textBaseline = 'middle';
    lines.forEach(function (ln, i) {
      g.fillStyle = ln.c || '#d4dae8';
      g.fillText(ln.t, 22, 36 + i * 52);
    });
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: canvasTex(c), transparent: true, depthWrite: false }));
    sp.scale.set(wWorld, wWorld * c.height / c.width, 1);
    return sp;
  }

  // 字牌：Plane 双面贴图（字符 / token id）
  function makeTile(text, opts) {
    opts = opts || {};
    var size = 128;
    var c = document.createElement('canvas');
    c.width = size; c.height = size;
    var g = c.getContext('2d');
    g.fillStyle = opts.bg || 'rgba(30,26,20,0.95)';
    g.fillRect(0, 0, size, size);
    g.strokeStyle = opts.border || '#a97c33'; g.lineWidth = 6;
    g.strokeRect(3, 3, size - 6, size - 6);
    g.fillStyle = opts.color || '#ffe2b0';
    g.font = (opts.small ? '600 40px' : '600 72px') + ' Menlo, -apple-system, monospace';
    g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText(text, size / 2, size / 2 + 2);
    var mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(opts.w || 0.95, opts.w || 0.95),
      new THREE.MeshBasicMaterial({ map: canvasTex(c), transparent: true, side: THREE.DoubleSide })
    );
    return mesh;
  }

  // ============================================================
  // 3. 场景布局：水平流水线（沿 x 轴，左 → 右）
  // ============================================================
  var X = {
    input: -28, tokenizer: -21, embed: -14,
    layers: [-5, 6, 17, 28],
    deep: 36.5, output: 47
  };
  var PATH_Y = 2.2;         // token 行进高度

  // 鲜艳的配色（文字用 hex 字符串，材质用 number）
  var ACCENT = {
    input: '#ffb84d', tokenizer: '#ffb84d', embed: '#ffb84d',
    attn: '#6eb5ff', q: '#c09aff', k: '#6eb5ff', v: '#5fe3ff',
    moe: '#3df0c8', router: '#ffc46b', out: '#ffd166', dim: '#8b96ad'
  };
  // 向量柱调色板：正值色 / 负值色 / 各自发光色
  var PAL = {
    main:  { pos: 0x5fb2ff, neg: 0xffa94d, posE: 0x1d4d8f, negE: 0x8f5414 },
    q:     { pos: 0xb07aff, neg: 0xff8f6b, posE: 0x4a2a8f, negE: 0x8f3a1a },
    k:     { pos: 0x5fb2ff, neg: 0xff8f6b, posE: 0x1d4d8f, negE: 0x8f3a1a },
    v:     { pos: 0x5fe3ff, neg: 0xff8f6b, posE: 0x14607a, negE: 0x8f3a1a },
    embed: { pos: 0xffc46b, neg: 0x6eb5ff, posE: 0x8f6214, negE: 0x1d4d8f },
    ffn:   { pos: 0x3df0c8, neg: 0xff8f6b, posE: 0x0f6a55, negE: 0x8f3a1a }
  };

  var S = { // 场景对象注册表
    scene: null, root: null,
    charTiles: [], idTiles: [],
    tokenG: null, tokenVg: null, tokenLabel: null, tokenHalo: null,
    layers: [],               // 每层的完整内部结构（见 buildLayers）
    outBars: [], outLabels: [], outPcts: [],
    beams: [], bases: [],
    embedBoard: null, infoPanel: null, sampledSprite: null, deepGlow: null,
    activeLight: null
  };

  function addTitle(text, x, y, color) {
    var sp = makeLabel(text, { h: 0.7, color: color || ACCENT.dim });
    sp.position.set(x, y, 0);
    S.root.add(sp);
    return sp;
  }

  // 向量柱组：8 根柱子，按调色板正/负着色，金属感 + 发光
  function makeVectorGroup(scale, palette) {
    palette = palette || PAL.main;
    var g = new THREE.Group();
    var bars = [];
    for (var i = 0; i < D; i++) {
      var m = new THREE.Mesh(
        new THREE.BoxGeometry(0.16, 1, 0.16),
        new THREE.MeshStandardMaterial({
          color: palette.pos, emissive: palette.posE, emissiveIntensity: 1,
          metalness: 0.45, roughness: 0.28
        })
      );
      m.position.x = (i - 3.5) * 0.22;
      g.add(m);
      bars.push(m);
    }
    g.scale.setScalar(scale || 1);
    return { group: g, bars: bars, palette: palette };
  }
  function setVectorValues(vg, v, lerpFrom, t) {
    for (var i = 0; i < D; i++) {
      var val = lerpFrom ? lerpFrom[i] + (v[i] - lerpFrom[i]) * t : v[i];
      var h = Math.max(0.06, Math.abs(val) * 0.85);
      var bar = vg.bars[i];
      bar.scale.y = h;
      bar.position.y = val >= 0 ? h / 2 : -h / 2;
      var pos = val >= 0;
      bar.material.color.set(pos ? vg.palette.pos : vg.palette.neg);
      bar.material.emissive.set(pos ? vg.palette.posE : vg.palette.negE);
    }
  }

  // 动态文字标签：可在运行时改文字（复用同一块 canvas，文字没变就不重绘）
  function makeDynLabel(text, opts) {
    opts = opts || {};
    var size = opts.fontSize || 40;
    var c = document.createElement('canvas');
    var font = '600 ' + size + 'px Menlo, -apple-system, monospace';
    function draw(t) {
      var g = c.getContext('2d');
      g.font = font;
      var tw = Math.ceil(g.measureText(t).width);
      if (c.width < tw + 24) c.width = tw + 24;
      c.height = size + 20;
      g = c.getContext('2d');
      g.clearRect(0, 0, c.width, c.height);
      g.font = font; g.textBaseline = 'middle';
      g.fillStyle = opts.color || '#ffd166';
      g.fillText(t, 12, c.height / 2);
    }
    draw(text);
    var tex = canvasTex(c);
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    var h = opts.h || 0.55;
    sp.scale.set(h * c.width / c.height, h, 1);
    var cur = text;
    return {
      sprite: sp,
      set: function (t) {
        if (t === cur) return;
        cur = t;
        draw(t);
        tex.needsUpdate = true;
        sp.scale.set(h * c.width / c.height, h, 1);
      }
    };
  }

  // 站台底座：深色金属台 + 发光描边 + 台面泛光 + 站名牌，让每个工位「落地」
  function buildStationBase(x, w, d, colorNum, colorCss, name) {
    var g = new THREE.Group();
    var slab = new THREE.Mesh(
      new THREE.BoxGeometry(w, 0.18, d),
      new THREE.MeshStandardMaterial({ color: 0x121a29, metalness: 0.75, roughness: 0.32 })
    );
    slab.position.y = 0.09;
    g.add(slab);
    var edge = new THREE.LineSegments(
      new THREE.EdgesGeometry(slab.geometry),
      new THREE.LineBasicMaterial({ color: colorNum, transparent: true, opacity: 0.9 })
    );
    edge.position.y = 0.09;
    g.add(edge);
    var glow = new THREE.Mesh(
      new THREE.PlaneGeometry(w - 0.2, d - 0.2),
      new THREE.MeshBasicMaterial({ color: colorNum, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false })
    );
    glow.rotation.x = -Math.PI / 2;
    glow.position.y = 0.19;
    g.add(glow);
    if (name) {
      var plate = makeLabel(name, { h: 0.46, color: colorCss, bg: 'rgba(14,18,30,0.88)' });
      plate.position.set(0, 0.62, d / 2 + 0.45);
      g.add(plate);
    }
    g.position.x = x;
    S.root.add(g);
    return { group: g, edge: edge, glow: glow };
  }

  // 站间流动光轨：沿 token 行进路径连接各站
  function buildFlowBeams() {
    var stops = [X.input, X.tokenizer, X.embed].concat(X.layers).concat([X.deep, X.output]);
    for (var i = 0; i < stops.length - 1; i++) {
      var x0 = stops[i], x1 = stops[i + 1];
      var beam = new THREE.Mesh(
        new THREE.BoxGeometry(x1 - x0 - 1.0, 0.05, 0.05),
        new THREE.MeshBasicMaterial({ color: 0x2f6fd0, transparent: true, opacity: 0.45 })
      );
      beam.position.set((x0 + x1) / 2, PATH_Y - 0.35, 0);
      S.root.add(beam);
      S.beams.push(beam);
    }
  }

  function buildInput() {
    addTitle('① 输入文字', X.input, 5.6, ACCENT.input);
    S.bases.push(buildStationBase(X.input, 10.4, 3.4, 0x8f6214, ACCENT.input, '输入 · 原始字符'));
    // 暖光把字牌照亮
    var pl = new THREE.PointLight(0xffc46b, 0.6, 12);
    pl.position.set(X.input, 5.0, 2.0);
    S.root.add(pl);
    for (var i = 0; i < SENT.length; i++) {
      var t = makeTile(SENT[i]);
      t.position.set(X.input + (i - 3.5) * 1.08, PATH_Y + 1.2, 0);
      t.scale.setScalar(0.001); // 入场时弹出
      S.root.add(t);
      S.charTiles.push(t);
    }
  }

  function buildTokenizer() {
    addTitle('② 分词 tokenizer', X.tokenizer, 5.6, ACCENT.input);
    S.bases.push(buildStationBase(X.tokenizer, 10.4, 3.4, 0x8f6214, ACCENT.input, '分词 · 查词表得 id'));
    var pl = new THREE.PointLight(0xffc46b, 0.6, 12);
    pl.position.set(X.tokenizer, 5.0, 2.0);
    S.root.add(pl);
    for (var i = 0; i < IDS.length; i++) {
      var t = makeTile(String(IDS[i]), { small: true, color: '#ffd9a0' });
      t.position.set(X.tokenizer + (i - 3.5) * 1.08, PATH_Y + 1.2, 0);
      t.scale.setScalar(0.001);
      S.root.add(t);
      S.idTiles.push(t);
    }
  }

  function buildEmbedding() {
    addTitle('③ Embedding 查表', X.embed, 5.6, ACCENT.input);
    S.bases.push(buildStationBase(X.embed, 9.6, 4.6, 0x8f6214, ACCENT.input, 'Embedding · id → 向量'));
    var pl = new THREE.PointLight(0xffd9a0, 0.55, 12);
    pl.position.set(X.embed, 5.0, 2.0);
    S.root.add(pl);
    // 词表架：背景一块大板（Embedding 阶段会刷新成真实向量数值）
    var board = makePanel([
      { t: 'Embedding 矩阵 E', c: ACCENT.dim },
      { t: '词表 160,000 × d_model', c: '#8b96ad' },
      { t: '查表：id → 向量', c: ACCENT.input }
    ], 5.2, ACCENT.input);
    board.position.set(X.embed, 3.6, -2.6);
    S.root.add(board);
    S.embedBoard = board;
    // 上下文 token 的迷你向量（停在 embedding 站的 8 股向量）
    for (var i = 0; i < IDS.length; i++) {
      var vg = makeVectorGroup(0.5, PAL.embed);
      setVectorValues(vg, embed(IDS[i]));
      vg.group.position.set(X.embed + (i - 3.5) * 0.9, PATH_Y, 1.6);
      S.root.add(vg.group);
      var ch = makeLabel(SENT[i], { h: 0.5, color: '#d8b98a' });
      ch.position.set(X.embed + (i - 3.5) * 0.9, PATH_Y + 1.1, 1.6);
      S.root.add(ch);
    }
  }

  // ---------- 层内部结构零件 ----------

  // 8×8 注意力热力矩阵：行=query token，列=被关注的上下文（下三角因果掩码）
  function makeHeatmap() {
    var geo = new THREE.BoxGeometry(0.19, 0.19, 0.05);
    var mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    var im = new THREE.InstancedMesh(geo, mat, 64);
    var dummy = new THREE.Object3D();
    var base = new THREE.Color();
    for (var r = 0; r < 8; r++) {
      for (var c = 0; c < 8; c++) {
        var i = r * 8 + c;
        dummy.position.set((c - 3.5) * 0.225, (3.5 - r) * 0.225, 0);
        dummy.updateMatrix();
        im.setMatrixAt(i, dummy.matrix);
        base.set(c <= r ? 0x1c3050 : 0x101826); // 因果下三角可见，上三角近乎熄灭
        im.setColorAt(i, base);
      }
    }
    im.instanceMatrix.needsUpdate = true;
    im.instanceColor.needsUpdate = true;
    return im;
  }
  function resetHeatmap(im) {
    var base = new THREE.Color();
    for (var r = 0; r < 8; r++) {
      for (var c = 0; c < 8; c++) {
        base.set(c <= r ? 0x1c3050 : 0x101826);
        im.setColorAt(r * 8 + c, base);
      }
    }
    im.instanceColor.needsUpdate = true;
  }
  // 把「当前 token 行」的注意力权重画进热力矩阵（错峰点亮，亮度 ∝ 权重）
  function paintHeatmapRow(im, attn, step, p) {
    var maxW = Math.max.apply(null, attn) || 1;
    var col = new THREE.Color();
    for (var j = 0; j <= step; j++) {
      var t = ease(seg(p, j * 0.09, j * 0.09 + 0.35));
      var wn = attn[j] / maxW;
      col.setHSL(0.58, 0.9, 0.14 + (0.30 + 0.42 * wn) * t);
      im.setColorAt(step * 8 + j, col);
    }
    im.instanceColor.needsUpdate = true;
  }

  // 896 专家墙：32×28 InstancedMesh（每层一面，常驻）
  // 基底压暗（像沉睡的神经元），激活时高亮弹出 —— 亮暗对比就是「稀疏激活」本身
  function wallBaseColor(base, r) { base.setHSL(0.46, 0.5, 0.022 + r() * 0.014); }
  function makeExpertWall() {
    var geo = new THREE.BoxGeometry(0.17, 0.17, 0.05);
    var mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    var grid = new THREE.InstancedMesh(geo, mat, EXPERTS);
    var dummy = new THREE.Object3D();
    var cols = 32;
    var base = new THREE.Color();
    var r = rng(4242);
    for (var e = 0; e < EXPERTS; e++) {
      dummy.position.set((e % cols - cols / 2 + 0.5) * 0.2, (Math.floor(e / cols) - 14 + 0.5) * 0.2, 0);
      dummy.updateMatrix();
      grid.setMatrixAt(e, dummy.matrix);
      wallBaseColor(base, r);
      grid.setColorAt(e, base);
    }
    grid.instanceMatrix.needsUpdate = true;
    grid.instanceColor.needsUpdate = true;
    return grid;
  }
  function dimExperts(grid) {
    var base = new THREE.Color();
    var r = rng(4242);
    for (var e = 0; e < EXPERTS; e++) {
      wallBaseColor(base, r);
      grid.setColorAt(e, base);
    }
    grid.instanceColor.needsUpdate = true;
  }
  function lightExperts(grid, layerData, p) {
    // top-16 按权重错峰点亮：权重越大越亮、越偏白
    var n = Math.floor(seg(p, 0, 1) * TOPK);
    for (var j = 0; j < n; j++) {
      var ei = layerData.top[j];
      var w = layerData.w[j];
      var col = new THREE.Color().setHSL(0.46, 1.0, Math.min(0.82, 0.45 + w * 2.4));
      grid.setColorAt(ei, col);
    }
    grid.instanceColor.needsUpdate = true;
  }

  // router 权重条：top-16 根琥珀色金属条
  function makeRouterBars() {
    var g = new THREE.Group();
    var bars = [];
    for (var i = 0; i < TOPK; i++) {
      var m = new THREE.Mesh(
        new THREE.BoxGeometry(0.13, 1, 0.13),
        new THREE.MeshStandardMaterial({
          color: 0xffc46b, emissive: 0x7a4d0e, emissiveIntensity: 1,
          metalness: 0.55, roughness: 0.25
        })
      );
      m.position.x = i * 0.185;
      m.scale.y = 0.05;
      g.add(m);
      bars.push(m);
    }
    return { group: g, bars: bars };
  }
  function setRouterBars(rb, weights, p) {
    var maxW = Math.max.apply(null, weights) || 1;
    for (var i = 0; i < TOPK; i++) {
      var t = ease(seg(p, i * 0.04, i * 0.04 + 0.3));
      var h = Math.max(0.05, weights[i] / maxW * 1.7 * t);
      rb.bars[i].scale.y = h;
      rb.bars[i].position.y = h / 2;
    }
  }

  // 专家 FFN 教学展开：输入向量(8) → 隐藏层(16) → 加权输出向量(8)
  function makeFFNDetail() {
    var g = new THREE.Group();
    var inVg = makeVectorGroup(0.42, PAL.ffn);
    inVg.group.position.set(0, 0, 0);
    g.add(inVg.group);
    var hidBars = [];
    var hidG = new THREE.Group();
    for (var i = 0; i < 16; i++) {
      var m = new THREE.Mesh(
        new THREE.BoxGeometry(0.11, 1, 0.11),
        new THREE.MeshStandardMaterial({
          color: 0x3df0c8, emissive: 0x0f6a55, emissiveIntensity: 1,
          metalness: 0.4, roughness: 0.3
        })
      );
      m.position.x = (i - 7.5) * 0.15;
      m.scale.y = 0.05;
      hidG.add(m);
      hidBars.push(m);
    }
    hidG.position.set(0, 1.35, 0);
    g.add(hidG);
    var outVg = makeVectorGroup(0.42, PAL.ffn);
    outVg.group.position.set(0, 2.7, 0);
    g.add(outVg.group);
    var lbIn = makeLabel('输入 x', { h: 0.4, color: ACCENT.moe });
    lbIn.position.set(0, 0.85, 0); g.add(lbIn);
    var lbHid = makeLabel('隐藏层 ×16', { h: 0.4, color: ACCENT.moe });
    lbHid.position.set(0, 2.2, 0); g.add(lbHid);
    var lbOut = makeLabel('输出 ×w', { h: 0.4, color: ACCENT.moe });
    lbOut.position.set(0, 3.55, 0); g.add(lbOut);
    return { group: g, inVg: inVg, hidBars: hidBars, outVg: outVg };
  }
  function setFFNDetail(ffn, data, p) {
    // 三段依次亮起：输入 → 隐藏 → 加权输出
    var t1 = ease(seg(p, 0, 0.35));
    var t2 = ease(seg(p, 0.3, 0.7));
    var t3 = ease(seg(p, 0.65, 1));
    setVectorValues(ffn.inVg, data.xa.map(function (v) { return v * t1; }));
    for (var i = 0; i < 16; i++) {
      var val = data.hid[i] * t2;
      var h = Math.max(0.05, Math.abs(val) * 1.5);
      var bar = ffn.hidBars[i];
      bar.scale.y = h;
      bar.position.y = val >= 0 ? h / 2 : -h / 2;
      bar.material.color.set(val >= 0 ? 0x3df0c8 : 0xff8f6b);
      bar.material.emissive.set(val >= 0 ? 0x0f6a55 : 0x8f3a1a);
    }
    setVectorValues(ffn.outVg, data.out.map(function (v) { return v * t3; }));
  }

  function buildLayers() {
    for (var l = 0; l < LAYERS_SHOWN; l++) {
      var lx = X.layers[l];
      var grp = new THREE.Group();
      grp.position.set(lx, 0, 0);
      var ref = { group: grp };

      // 层外壳：玻璃感半透明盒 + 亮色描边
      var frame = new THREE.Mesh(
        new THREE.BoxGeometry(9.6, 6.4, 9),
        new THREE.MeshStandardMaterial({
          color: 0x2a3c5c, transparent: true, opacity: 0.07,
          metalness: 0.1, roughness: 0.9, depthWrite: false
        })
      );
      frame.position.y = 3.2;
      grp.add(frame);
      var edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(frame.geometry),
        new THREE.LineBasicMaterial({ color: 0x3d5a8c, transparent: true, opacity: 0.9 })
      );
      edges.position.copy(frame.position);
      grp.add(edges);
      ref.frame = frame; ref.edges = edges;

      // 站台底座 + 四角发光立柱（让层像一座真正的「工作站」）
      S.bases.push(buildStationBase(lx, 10.0, 9.4, 0x3d5a8c, '#9fc6ff', null));
      ref.postMats = [];
      var postGeo = new THREE.BoxGeometry(0.14, 6.4, 0.14);
      [[-4.8, -4.5], [-4.8, 4.5], [4.8, -4.5], [4.8, 4.5]].forEach(function (c) {
        var pm = new THREE.MeshStandardMaterial({
          color: 0x2c4266, emissive: 0x16283f, emissiveIntensity: 1,
          metalness: 0.65, roughness: 0.3
        });
        var p = new THREE.Mesh(postGeo, pm);
        p.position.set(c[0], 3.2, c[1]);
        grp.add(p);
        ref.postMats.push(pm);
      });

      // 层标题 + 发光下划线
      var title = makeLabel('Layer ' + (l + 1) + ' / ' + LAYERS_TOTAL, { h: 0.72, color: '#cfe0ff' });
      title.position.set(0, 6.95, 0);
      grp.add(title);
      var underline = new THREE.Mesh(
        new THREE.BoxGeometry(3.4, 0.05, 0.05),
        new THREE.MeshBasicMaterial({ color: 0x4d7fc0 })
      );
      underline.position.set(0, 6.55, 0);
      grp.add(underline);
      ref.underline = underline;

      // 层内点光：把整个工作站从内部点亮
      var light = new THREE.PointLight(0x86c9ff, 0.55, 13);
      light.position.set(0, 5.0, 0.5);
      grp.add(light);
      ref.light = light;

      // ===== 前舱：MLA 注意力 =====
      var at = makeLabel('MLA 注意力', { h: 0.55, color: ACCENT.attn });
      at.position.set(-2.5, 5.85, 3.2);
      grp.add(at);

      // Q / K / V 三组向量（真实数值）
      var qVg = makeVectorGroup(0.6, PAL.q);
      qVg.group.position.set(-3.7, 2.2, 3.2);
      grp.add(qVg.group);
      var kVg = makeVectorGroup(0.6, PAL.k);
      kVg.group.position.set(-2.5, 2.2, 3.2);
      grp.add(kVg.group);
      var vVg = makeVectorGroup(0.6, PAL.v);
      vVg.group.position.set(-1.3, 2.2, 3.2);
      grp.add(vVg.group);
      ref.qVg = qVg; ref.kVg = kVg; ref.vVg = vVg;
      [['Q', -3.7, ACCENT.q], ['K', -2.5, ACCENT.k], ['V', -1.3, ACCENT.v]].forEach(function (it) {
        var lb = makeLabel(it[0], { h: 0.5, color: it[2] });
        lb.position.set(it[1], 4.3, 3.2);
        grp.add(lb);
      });

      // 8×8 注意力权重热力矩阵
      var heat = makeHeatmap();
      heat.position.set(0.75, 3.0, 3.3);
      grp.add(heat);
      ref.heat = heat;
      var hl = makeLabel('attn = softmax(q·K/√d)', { h: 0.42, color: ACCENT.attn });
      hl.position.set(0.75, 4.35, 3.3);
      grp.add(hl);

      // 注意力加权输出向量
      var attnOutVg = makeVectorGroup(0.6, PAL.k);
      attnOutVg.group.position.set(2.9, 2.2, 3.2);
      grp.add(attnOutVg.group);
      ref.attnOutVg = attnOutVg;
      var ao = makeLabel('Σ aⱼ·V', { h: 0.5, color: ACCENT.k });
      ao.position.set(2.9, 4.3, 3.2);
      grp.add(ao);

      // ===== 后舱：MoE（从后舱机位看：router/FFN 在专家墙前方，互不遮挡）=====
      var mt = makeLabel('MoE · 896 专家 → 激活 16', { h: 0.55, color: ACCENT.moe });
      mt.position.set(0.6, 5.85, -3.6);
      grp.add(mt);

      // router top-16 权重条（z 比专家墙更靠后 → 后舱机位下浮在墙面前方）
      var rb = makeRouterBars();
      rb.group.position.set(-4.35, 1.35, -4.55);
      grp.add(rb.group);
      ref.router = rb;
      var rl = makeLabel('router → top-16 权重', { h: 0.45, color: ACCENT.router });
      rl.position.set(-3.0, 3.7, -4.55);
      grp.add(rl);

      // 896 专家墙（本身深色，就是 router/FFN 的天然衬底）
      var wall = makeExpertWall();
      wall.position.set(0.9, 3.2, -4.15);
      grp.add(wall);
      ref.wall = wall;

      // 专家 FFN 展开（top-1 专家：输入 → 隐藏 → 加权输出）
      var ffn = makeFFNDetail();
      ffn.group.position.set(4.0, 1.2, -4.6);
      grp.add(ffn.group);
      ref.ffn = ffn;
      var ft = makeLabel('专家 FFN', { h: 0.42, color: ACCENT.moe });
      ft.position.set(4.0, 5.25, -4.6);
      grp.add(ft);
      ref.ffnTitle = ft;

      S.root.add(grp);
      S.layers.push(ref);
    }
  }

  function buildDeepGate() {
    addTitle('④ …再重复 57 层…', X.deep, 5.6, ACCENT.dim);
    S.bases.push(buildStationBase(X.deep, 3.2, 4.2, 0x3d5a8c, '#9fc6ff', '深层 · 结构不变'));
    // 两根立柱 + 半透明门
    var post = new THREE.BoxGeometry(0.3, 6, 0.3);
    var pm = new THREE.MeshStandardMaterial({ color: 0x3a5a8f, metalness: 0.6, roughness: 0.35 });
    [-1.6, 1.6].forEach(function (dz) {
      var p = new THREE.Mesh(post, pm);
      p.position.set(X.deep, 3, dz);
      S.root.add(p);
    });
    var door = new THREE.Mesh(
      new THREE.PlaneGeometry(0.15, 3.4),
      new THREE.MeshBasicMaterial({ color: 0x5fb2ff, transparent: true, opacity: 0.22, side: THREE.DoubleSide })
    );
    door.position.set(X.deep, PATH_Y + 0.5, 0);
    S.root.add(door);
    S.deepGlow = door;
    var lb = makeLabel('× 57', { h: 1.1, color: '#9fc6ff' });
    lb.position.set(X.deep, 4.6, 0);
    S.root.add(lb);
  }

  function buildOutput() {
    addTitle('⑤ 输出头 → 采样', X.output, 6.4, ACCENT.out);
    S.bases.push(buildStationBase(X.output, 11.2, 3.6, 0x8f6a14, ACCENT.out, '输出 · 概率分布采样'));
    var pl = new THREE.PointLight(0xffd166, 0.6, 14);
    pl.position.set(X.output, 6.0, 2.5);
    S.root.add(pl);
    // 11 个候选字的概率柱：金色金属材质
    for (var i = 0; i < OUT_CH.length; i++) {
      var b = new THREE.Mesh(
        new THREE.BoxGeometry(0.5, 1, 0.5),
        new THREE.MeshStandardMaterial({
          color: 0xffd166, emissive: 0x5a4308, emissiveIntensity: 1,
          metalness: 0.55, roughness: 0.28
        })
      );
      b.position.set(X.output + (i - 5) * 0.85, 0.55, 0);
      b.scale.y = 0.05;
      S.root.add(b);
      S.outBars.push(b);
      var lb = makeLabel(OUT_CH[i], { h: 0.72, color: '#f2eee6' });
      lb.position.set(X.output + (i - 5) * 0.85, 0.62, 0.8);
      S.root.add(lb);
      S.outLabels.push(lb);
      // 每根柱子上方的实时概率百分比（输出阶段逐帧刷新）
      var pct = makeDynLabel('', { h: 0.5, color: '#ffd166' });
      pct.sprite.position.set(X.output + (i - 5) * 0.85, 1.0, 0);
      pct.sprite.visible = false;
      S.root.add(pct.sprite);
      S.outPcts.push(pct);
    }
    // 采样结果大字
    var sp = makeLabel('?', { h: 2.2, color: '#ffffff' });
    sp.position.set(X.output, 8.4, 0);
    sp.visible = false;
    S.root.add(sp);
    S.sampledSprite = sp;
  }

  function buildToken() {
    var vg = makeVectorGroup(1.0, PAL.main);
    S.tokenVg = vg; S.tokenG = vg.group;
    S.tokenG.position.set(X.embed, PATH_Y, 0);
    S.root.add(S.tokenG);
    var lb = makeLabel('K', { h: 0.7, color: '#ffffff', bg: 'rgba(70,110,190,0.9)' });
    lb.position.set(0, 1.6, 0);
    S.tokenG.add(lb);
    S.tokenLabel = lb;
    // 脚下光环：标记「主角」，随 token 走，呼吸脉动
    var halo = new THREE.Mesh(
      new THREE.RingGeometry(0.62, 0.95, 48),
      new THREE.MeshBasicMaterial({ color: 0x6eb5ff, transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false })
    );
    halo.rotation.x = -Math.PI / 2;
    halo.position.y = -PATH_Y + 0.28; // 相对 tokenG：落到世界 y≈0.28 的台面上方
    S.tokenG.add(halo);
    S.tokenHalo = halo;
  }

  // 可刷新的信息面板（每层数字详情），放到层的正上方避免遮挡内部结构
  function setInfoPanel(lines, accent, x, y, z, w) {
    if (S.infoPanel) {
      S.root.remove(S.infoPanel);
      S.infoPanel.material.map.dispose();
      S.infoPanel.material.dispose();
    }
    S.infoPanel = makePanel(lines, w || 6.4, accent);
    S.infoPanel.position.set(x, y, z == null ? 0 : z);
    S.root.add(S.infoPanel);
  }
  function clearInfoPanel() {
    if (S.infoPanel) {
      S.root.remove(S.infoPanel);
      S.infoPanel.material.map.dispose();
      S.infoPanel.material.dispose();
      S.infoPanel = null;
    }
  }

  // 刷新 Embedding 词表架大板的内容（换纹理，不动位置）
  function setEmbedBoard(lines) {
    if (!S.embedBoard) return;
    var np = makePanel(lines, 5.2, ACCENT.input);
    S.embedBoard.material.map.dispose();
    S.embedBoard.material.map = np.material.map;
    S.embedBoard.scale.copy(np.scale);
    S.embedBoard.material.needsUpdate = true;
  }

  function buildFloorPath() {
    var len = X.output - X.input + 6;
    var strip = new THREE.Mesh(
      new THREE.BoxGeometry(len, 0.04, 0.5),
      new THREE.MeshBasicMaterial({ color: 0x2c4d8f, transparent: true, opacity: 0.6 })
    );
    strip.position.set((X.input + X.output) / 2, 0.03, 0);
    S.root.add(strip);
  }

  // ============================================================
  // 4. 旅程时间线：阶段脚本 + 相机跟随
  // ============================================================
  var journey = null;        // computeJourney 结果
  var stageIdx = 0, stageT = 0;
  var playing = true, follow = true;
  var lastDragT = -999, clock = 0;
  var morphFrom = null;      // 向量 morph 起点

  function ease(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }
  function clamp01(t) { return Math.max(0, Math.min(1, t)); }
  function seg(p, a, b) { return clamp01((p - a) / (b - a)); } // 阶段内子时段

  function flyTo(x, y, z, r, theta, phi) {
    if (!follow) return;
    if (clock - lastDragT < 5) return; // 用户刚拖拽过，让出控制权
    window.CORE_CAMERA.flyTo({ x: x, y: y, z: z }, r, phi == null ? 1.12 : phi, theta == null ? 0.35 : theta);
  }

  function caption(title, detail) {
    var el1 = document.getElementById('stageTitle');
    var el2 = document.getElementById('caption');
    if (el1) el1.textContent = title;
    if (el2) el2.textContent = detail || '';
  }

  // 把一层的所有内部结构复位到「待机」状态（结构常驻，数据清空）
  function resetLayerVisual(l) {
    var ref = S.layers[l];
    var zeros = [0, 0, 0, 0, 0, 0, 0, 0];
    setVectorValues(ref.qVg, zeros);
    setVectorValues(ref.kVg, zeros);
    setVectorValues(ref.vVg, zeros);
    setVectorValues(ref.attnOutVg, zeros);
    resetHeatmap(ref.heat);
    setRouterBars(ref.router, new Array(TOPK).fill(0.001), 0);
    dimExperts(ref.wall);
    setVectorValues(ref.ffn.inVg, zeros);
    setVectorValues(ref.ffn.outVg, zeros);
    for (var i = 0; i < 16; i++) { ref.ffn.hidBars[i].scale.y = 0.05; ref.ffn.hidBars[i].position.y = 0.025; }
    ref.light.intensity = 0.55;
  }
  function resetAllLayers() {
    for (var l = 0; l < LAYERS_SHOWN; l++) resetLayerVisual(l);
    S.activeLight = null;
    S.layers.forEach(function (ref) {
      ref.edges.material.color.set(0x3d5a8c);
      ref.edges.material.opacity = 0.9;
      ref.underline.material.color.set(0x4d7fc0);
      ref.postMats.forEach(function (pm) { pm.emissive.set(0x16283f); });
    });
  }

  // ---------- 阶段定义 ----------
  var stages = [
    { // 0 全景
      dur: 4.0,
      enter: function () {
        caption('Kimi K3：一个 token 的完整旅程',
          '输入 → 分词 → Embedding → 61 层（此处展示 4 层，每层内含注意力 + MoE 完整结构）→ 输出 · d=8 教学维度，数字全部为真实前向计算');
        flyTo(12, 2.5, 0, 54, 0.65, 1.22);
      },
      tick: function () {}
    },
    { // 1 输入
      dur: 3.5,
      enter: function () {
        caption('① 输入：「' + SENT + '」', '8 个字符进入网络');
        flyTo(X.input, 3, 0, 10, 0.3);
      },
      tick: function (p) {
        for (var i = 0; i < S.charTiles.length; i++) {
          var s = ease(seg(p, i * 0.09, i * 0.09 + 0.3));
          S.charTiles[i].scale.setScalar(Math.max(0.001, s));
        }
      }
    },
    { // 2 分词
      dur: 4.0,
      enter: function () {
        caption('② 分词：字符 → token id',
          '「' + SENT[journey.step] + '」= id ' + IDS[journey.step] + '（词表 16 万中查到的编号）');
        flyTo(X.tokenizer, 3, 0, 10, 0.3);
      },
      tick: function (p) {
        for (var i = 0; i < S.idTiles.length; i++) {
          var s = ease(seg(p, i * 0.08, i * 0.08 + 0.3));
          S.idTiles[i].scale.setScalar(Math.max(0.001, s));
        }
      }
    },
    { // 3 Embedding
      dur: 5.0,
      enter: function () {
        var x = journey.x0;
        caption('③ Embedding 查表：id ' + IDS[journey.step] + ' → 8 维向量',
          'E[' + IDS[journey.step] + '] = ' + vecShort(x, 4) + ' · 主角 token「' + SENT[journey.step] + '」获得向量本体');
        flyTo(X.embed, 3, 0.6, 11, 0.3);
        setEmbedBoard([
          { t: 'E[' + IDS[journey.step] + '] =（查表命中）', c: ACCENT.dim },
          { t: '[' + x.map(function (v) { return fmt(v, 1); }).join(' ') + ']', c: ACCENT.input },
          { t: '每一维对应一根柱子（亮正暖负）', c: '#8b96ad' }
        ]);
        morphFrom = null;
      },
      tick: function (p) {
        // 主角向量柱子从 0 长出
        var t = ease(seg(p, 0.15, 0.7));
        setVectorValues(S.tokenVg, journey.x0.map(function (v) { return v * t; }));
      },
      leave: function () { clearInfoPanel(); }
    },
    // 4-7：四个展示层（enter/tick 在下方 makeLayerStage 生成）
    makeLayerStage(0), makeLayerStage(1), makeLayerStage(2), makeLayerStage(3),
    { // 8 穿越深层门
      dur: 3.0,
      enter: function () {
        caption('④ …同样的层再重复 57 次（共 ' + LAYERS_TOTAL + ' 层）…',
          '每一层都是：注意力 + MoE + 残差，向量被一层层精炼');
        flyTo(X.deep, 3.4, 0, 9, 0.3);
      },
      tick: function (p) {
        var x = X.layers[3] + (X.deep - X.layers[3]) * ease(p);
        S.tokenG.position.x = x;
        S.deepGlow.material.opacity = 0.22 + 0.5 * Math.sin(p * Math.PI);
      }
    },
    { // 9 输出
      dur: 7.0,
      enter: function () {
        caption('⑤ 输出头：logits → softmax → top-p 采样',
          '向量与词表投影相乘，得到每个候选字的分数');
        flyTo(X.output, 3.4, 0, 13, 0.3);
        S.deepGlow.material.opacity = 0.22;
      },
      tick: function (p) {
        // token 入场
        var tin = ease(seg(p, 0, 0.12));
        S.tokenG.position.x = X.deep + (X.output - 1.5 - X.deep) * tin;
        S.tokenG.position.z = -1.8 * tin; // 退到概率柱后方，不遮挡
        // logits 上升
        var pl = ease(seg(p, 0.12, 0.45));
        var maxL = Math.max.apply(null, journey.logits.map(Math.abs));
        // softmax 概率
        var pp = ease(seg(p, 0.5, 0.8));
        var maxP = Math.max.apply(null, journey.probs);
        for (var i = 0; i < S.outBars.length; i++) {
          var hl = journey.logits[i] / maxL * 2.2 * pl;
          var hp = journey.probs[i] / maxP * 3.2 * pp;
          var h = pl < 1 || pp === 0 ? hl : hl + (hp - hl) * pp;
          h = Math.max(0.05, Math.abs(h));
          S.outBars[i].scale.y = h;
          S.outBars[i].position.y = 0.3 + h / 2;
          // top-p 核外的候选变灰
          var inNucleus = journey.keep[i];
          var dim = pp > 0.5 && !inNucleus;
          S.outBars[i].material.color.set(dim ? 0x555a66 : 0xffd166);
          S.outBars[i].material.emissive.set(dim ? 0x14161c : 0x5a4308);
          // 柱顶实时概率百分比：softmax 阶段起显示，随柱高浮动，核外变灰
          var pct = S.outPcts[i];
          if (pp > 0.05) {
            pct.sprite.visible = true;
            pct.set((journey.probs[i] * 100).toFixed(0) + '%');
            pct.sprite.position.y = 0.42 + h;
            pct.sprite.material.color.set(dim ? 0x7a7f8c : 0xffd166);
          } else {
            pct.sprite.visible = false;
          }
        }
        if (p > 0.5 && p < 0.52 && !S._outCaptioned) {
          S._outCaptioned = true;
          var o = journey.order;
          caption('⑤ softmax 概率 → top-p=' + TOP_P + ' 核采样',
            '「' + OUT_CH[o[0]] + '」' + (journey.probs[o[0]] * 100).toFixed(1) + '% · 「' +
            OUT_CH[o[1]] + '」' + (journey.probs[o[1]] * 100).toFixed(1) + '% · 灰色 = 核外被截断');
        }
        // 采样结果
        if (p > 0.82) {
          var top1 = journey.order[0];
          if (!S.sampledSprite.visible) {
            S.root.remove(S.sampledSprite);
            S.sampledSprite = makeLabel('「' + OUT_CH[top1] + '」', { h: 2.4, color: '#ffe9b0' });
            S.sampledSprite.position.set(X.output, 6.4, 0);
            S.root.add(S.sampledSprite);
            S.sampledSprite.visible = true;
            caption('⑥ 采样输出：「' + OUT_CH[top1] + '」',
              '按概率掷骰子选中它 → 拼到输入末尾 → 开始下一轮旅程');
          }
          var rise = seg(p, 0.82, 1);
          S.sampledSprite.position.y = 6.4 + rise * 2.2;
          S.sampledSprite.material.opacity = 1 - rise * 0.4;
        }
      },
      leave: function () { S._outCaptioned = false; }
    }
  ];

  function makeLayerStage(l) {
    var lx = X.layers[l];
    var prevX = l === 0 ? X.embed : X.layers[l - 1];
    var flipped = false; // 相机是否已转到后舱（MoE）机位
    return {
      dur: 7.0,
      enter: function () {
        var ld = journey.layers[l];
        var ref = S.layers[l];
        var best = 0;
        for (var i = 1; i < ld.attn.length; i++) if (ld.attn[i] > ld.attn[best]) best = i;
        caption(
          'Layer ' + (l + 1) + '/' + LAYERS_TOTAL + '：注意力 → MoE → 残差',
          '「' + SENT[journey.step] + '」最关注「' + SENT[best] + '」(' + ld.attn[best].toFixed(2) +
          ') · 896 专家激活 16 个：#' + ld.top[0] + ' w=' + ld.w[0].toFixed(2) + '、#' +
          ld.top[1] + ' w=' + ld.w[1].toFixed(2) + ' …'
        );
        // 第一机位：前舱注意力特写
        flipped = false;
        flyTo(lx - 0.5, 3.0, 2.6, 8.0, 0.22, 1.05);
        // 本层复位后点亮；其余层保持上一状态（结构 + 数据常驻）
        resetLayerVisual(l);
        S.layers.forEach(function (r2, i) {
          var active = i === l;
          r2.edges.material.color.set(active ? 0x6ea8ff : 0x3d5a8c);
          r2.edges.material.opacity = active ? 1 : 0.9;
          r2.underline.material.color.set(active ? 0x9fc6ff : 0x4d7fc0);
          r2.postMats.forEach(function (pm) { pm.emissive.set(active ? 0x2f5fae : 0x16283f); });
        });
        S.activeLight = ref.light;
        morphFrom = null;
        // 顶部信息面板：本层真实数字
        setInfoPanel([
          { t: 'attn = softmax(q·K/√d)', c: ACCENT.attn },
          { t: '  [' + ld.attn.map(function (a) { return a.toFixed(2); }).join(' ') + ']', c: '#d4dae8' },
          { t: 'top-16 专家权重 Σ=' + ld.w.reduce(function (a, b) { return a + b; }, 0).toFixed(2) +
               ' · top-1 = 专家 #' + ld.top[0], c: ACCENT.moe },
          { t: 'y = Σ wᵢ·FFNᵢ(x)，残差相加后继续', c: '#8b96ad' }
        ], ACCENT.attn, lx, 8.1, 0, 6.8);
        // 刷新 FFN 标题为 top-1 专家编号
        ref.group.remove(ref.ffnTitle);
        ref.ffnTitle.material.map.dispose();
        ref.ffnTitle.material.dispose();
        ref.ffnTitle = makeLabel('专家 #' + ld.top[0] + ' · FFN', { h: 0.42, color: ACCENT.moe });
        ref.ffnTitle.position.set(4.0, 5.25, -4.6);
        ref.group.add(ref.ffnTitle);
      },
      tick: function (p) {
        var ld = journey.layers[l];
        var ref = S.layers[l];
        var xIn = l === 0 ? journey.x0 : journey.layers[l - 1].out;
        var best = 0;
        for (var i = 1; i < ld.attn.length; i++) if (ld.attn[i] > ld.attn[best]) best = i;

        // 0-0.10：token 飞入本层
        var tin = ease(seg(p, 0, 0.10));
        S.tokenG.position.x = prevX + (lx - prevX) * tin;

        // 0.08-0.28：Q/K/V 向量长出（q=本层输入，K/V=最受关注的上下文 token）
        var tq = ease(seg(p, 0.08, 0.28));
        setVectorValues(ref.qVg, xIn.map(function (v) { return v * tq; }));
        setVectorValues(ref.kVg, journey.ctx[best].map(function (v) { return v * tq; }));
        setVectorValues(ref.vVg, journey.ctx[best].map(function (v) { return v * tq; }));

        // 0.22-0.45：热力矩阵当前行按注意力权重点亮
        paintHeatmapRow(ref.heat, ld.attn, journey.step, seg(p, 0.22, 0.45));

        // 0.42-0.55：注意力加权输出 = xa（注意力后的向量）
        var ta = ease(seg(p, 0.42, 0.55));
        setVectorValues(ref.attnOutVg, ld.xa, xIn, ta);

        // 0.50：相机绕到后舱 —— 进入 MoE 段落
        if (p >= 0.50 && !flipped) {
          flipped = true;
          flyTo(lx + 1.0, 3.2, -2.2, 13.0, Math.PI - 0.22, 1.08);
        }

        // 0.52-0.70：router top-16 权重条长出
        setRouterBars(ref.router, ld.w, seg(p, 0.52, 0.70));

        // 0.58-0.78：专家墙 top-16 错峰点亮
        lightExperts(ref.wall, ld, seg(p, 0.58, 0.78));

        // 0.70-0.88：top-1 专家 FFN 展开（输入 → 隐藏 → 加权输出）
        var ffnData = expertFFN(ld.xa, ld.top[0], ld.w[0]);
        setFFNDetail(ref.ffn, { xa: ld.xa, hid: ffnData.hid, out: ffnData.out }, seg(p, 0.70, 0.88));

        // 0.86-1.0：主角向量 morph 到本层输出（残差后的新向量）
        var tm = ease(seg(p, 0.86, 0.99));
        if (tm > 0 && !morphFrom) morphFrom = xIn.slice();
        if (morphFrom) setVectorValues(S.tokenVg, ld.out, morphFrom, tm);
      },
      leave: function () {
        clearInfoPanel();
        morphFrom = null;
        S.activeLight = null;
        if (S.layers[l]) S.layers[l].light.intensity = 0.55;
      }
    };
  }

  // ---------- 旅程控制 ----------
  var STAGE_NAMES = ['全景', '① 输入', '② 分词', '③ Embedding',
    'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', '④ 深层 ×57', '⑤ 输出采样'];
  var stageListener = null; // 阶段切换回调（main.js 用来高亮导航条）

  function enterStage(i) {
    stageIdx = i; stageT = 0;
    stages[i].enter();
    if (stageListener) stageListener(i, stages.length);
  }

  function startJourney(step) {
    journey = computeJourney(step);
    // 重置可动对象
    S.tokenG.position.set(X.embed, PATH_Y, 0);
    S.tokenG.position.z = 0;
    setVectorValues(S.tokenVg, journey.x0.map(function () { return 0.06; }));
    // token 顶上的字符牌
    S.tokenG.remove(S.tokenLabel);
    S.tokenLabel = makeLabel(SENT[step], { h: 0.7, color: '#ffffff', bg: 'rgba(70,110,190,0.9)' });
    S.tokenLabel.position.set(0, 1.6, 0);
    S.tokenG.add(S.tokenLabel);
    S.charTiles.forEach(function (t) { t.scale.setScalar(0.001); });
    S.idTiles.forEach(function (t) { t.scale.setScalar(0.001); });
    S.outBars.forEach(function (b) { b.scale.y = 0.05; b.position.y = 0.3; });
    S.outPcts.forEach(function (p) { p.sprite.visible = false; p.set(''); });
    S.sampledSprite.visible = false;
    S.sampledSprite.material.opacity = 1;
    resetAllLayers();
    clearInfoPanel();
    enterStage(0);
  }

  var curStep = 6; // 主角从「K」开始（前面有 7 个上下文 token）

  function update(delta) {
    clock += delta;
    if (playing) {
      stageT += delta;
      var st = stages[stageIdx];
      var p = clamp01(stageT / st.dur);
      if (st.tick) st.tick(p);
      if (stageT >= st.dur) {
        if (st.leave) st.leave();
        if (stageIdx + 1 >= stages.length) {
          // 旅程结束：主角推进到下一个 token，循环
          curStep = (curStep + 1) % IDS.length;
          if (curStep === 0) curStep = 1; // 至少留 1 个上下文
          startJourney(curStep);
        } else {
          enterStage(stageIdx + 1);
        }
      }
    }
    // 主角呼吸浮动（即使暂停也有一点生命力）
    if (S.tokenG) {
      S.tokenG.position.y = PATH_Y + Math.sin(clock * 1.8) * 0.06;
    }
    // 主角脚下光环脉动
    if (S.tokenHalo) {
      S.tokenHalo.material.opacity = 0.38 + 0.18 * Math.sin(clock * 3.2);
      var hs = 1 + 0.08 * Math.sin(clock * 3.2);
      S.tokenHalo.scale.set(hs, hs, 1);
    }
    // 激活层的内部点光呼吸脉冲
    if (S.activeLight) {
      S.activeLight.intensity = 0.85 + Math.sin(clock * 3) * 0.25;
    }
  }

  return {
    init: function (scene) {
      S.scene = scene;
      S.root = new THREE.Group();
      scene.add(S.root);
      buildFloorPath();
      buildFlowBeams();
      buildInput();
      buildTokenizer();
      buildEmbedding();
      buildLayers();
      buildDeepGate();
      buildOutput();
      buildToken();
      startJourney(curStep);
    },
    update: update,
    play: function () { playing = true; },
    pause: function () { playing = false; },
    isPlaying: function () { return playing; },
    restart: function () { startJourney(curStep); },
    setFollow: function (v) { follow = v; },
    notifyUserDrag: function () { lastDragT = clock; },
    // 手动播放控制
    stageNames: function () { return STAGE_NAMES; },
    currentStage: function () { return stageIdx; },
    setStageListener: function (fn) { stageListener = fn; },
    goTo: function (i) {
      i = Math.max(0, Math.min(stages.length - 1, i));
      if (stages[stageIdx].leave) stages[stageIdx].leave();
      enterStage(i);
      // 快进一帧让 tick(0) 生效
      if (stages[stageIdx].tick) stages[stageIdx].tick(0.0001);
    },
    next: function () {
      if (stageIdx + 1 >= stages.length) {
        // 最后一步再「下一步」：推进到下一个 token 的新旅程
        curStep = (curStep + 1) % IDS.length;
        if (curStep === 0) curStep = 1;
        startJourney(curStep);
      } else {
        this.goTo(stageIdx + 1);
      }
    },
    prev: function () { this.goTo(stageIdx - 1); },
    debugGo: function (i) { this.goTo(i); },
    debugTick: function (p) { // 截图验证用：把当前阶段推进到进度 p
      if (stages[stageIdx].tick) stages[stageIdx].tick(clamp01(p));
    },
    stageCount: function () { return stages.length; }
  };
})();
