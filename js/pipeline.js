/* pipeline.js — Kimi K3 整体流水线：一个 token 穿越整个网络的完整旅程
 *
 * 设计（乔布斯式）：一条水平流水线，从左到右 ——
 *   输入文字 → 分词 → Embedding → [Layer ×4（代表 61 层）] → …×57 → 输出头 → 采样
 * 主角是一个 token 的「向量本体」（8 根真实数值的柱子），不是抽象小球。
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
    return { attn: attn, top: top, w: w, y: y, out: out };
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
    g.fillStyle = opts.color || '#e8e4dc';
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
    g.fillStyle = 'rgba(16,18,26,0.97)'; g.fillRect(0, 0, c.width, c.height);
    g.fillStyle = accent || '#4a4036'; g.fillRect(0, 0, 8, c.height);
    g.font = '22px Menlo, monospace'; g.textBaseline = 'middle';
    lines.forEach(function (ln, i) {
      g.fillStyle = ln.c || '#c9cede';
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
    g.fillStyle = opts.bg || 'rgba(24,28,40,0.95)';
    g.fillRect(0, 0, size, size);
    g.strokeStyle = opts.border || '#3a4258'; g.lineWidth = 6;
    g.strokeRect(3, 3, size - 6, size - 6);
    g.fillStyle = opts.color || '#e8e4dc';
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
  var ACCENT = {
    input: '#e8a94d', tokenizer: '#e8a94d', embed: '#e8a94d',
    attn: '#7aa2f7', moe: '#46c8ae', out: '#c8b06a', dim: '#3a4258'
  };

  var S = { // 场景对象注册表
    scene: null, root: null,
    charTiles: [], idTiles: [], ctxTokens: [], ctxLines: [],
    token: null, tokenBars: [], tokenLabel: null,
    expertGrid: null, expertBase: [],
    layerGroups: [], layerFrames: [],
    outBars: [], outLabels: [], outPanel: null,
    infoPanel: null, sampledSprite: null, deepGlow: null
  };

  function addTitle(text, x, y, color) {
    var sp = makeLabel(text, { h: 0.7, color: color || '#8b96ad' });
    sp.position.set(x, y, 0);
    S.root.add(sp);
    return sp;
  }

  function buildInput() {
    addTitle('① 输入文字', X.input, 5.6, ACCENT.input);
    for (var i = 0; i < SENT.length; i++) {
      var t = makeTile(SENT[i], { border: '#5a4a2e' });
      t.position.set(X.input + (i - 3.5) * 1.08, PATH_Y + 1.2, 0);
      t.scale.setScalar(0.001); // 入场时弹出
      S.root.add(t);
      S.charTiles.push(t);
    }
  }

  function buildTokenizer() {
    addTitle('② 分词 tokenizer', X.tokenizer, 5.6, ACCENT.input);
    for (var i = 0; i < IDS.length; i++) {
      var t = makeTile(String(IDS[i]), { small: true, border: '#5a4a2e', color: '#e8a94d' });
      t.position.set(X.tokenizer + (i - 3.5) * 1.08, PATH_Y + 1.2, 0);
      t.scale.setScalar(0.001);
      S.root.add(t);
      S.idTiles.push(t);
    }
  }

  // 向量柱组：8 根柱子，正蓝负橙，每根都是真实数值
  function makeVectorGroup(scale) {
    var g = new THREE.Group();
    var bars = [];
    for (var i = 0; i < D; i++) {
      var m = new THREE.Mesh(
        new THREE.BoxGeometry(0.16, 1, 0.16),
        new THREE.MeshStandardMaterial({ color: 0x7aa2f7, emissive: 0x223a66, roughness: 0.4 })
      );
      m.position.x = (i - 3.5) * 0.22;
      g.add(m);
      bars.push(m);
    }
    g.scale.setScalar(scale || 1);
    return { group: g, bars: bars };
  }
  function setVectorValues(bars, v, lerpFrom, t) {
    for (var i = 0; i < D; i++) {
      var val = lerpFrom ? lerpFrom[i] + (v[i] - lerpFrom[i]) * t : v[i];
      var h = Math.max(0.06, Math.abs(val) * 0.85);
      bars[i].scale.y = h;
      bars[i].position.y = val >= 0 ? h / 2 : -h / 2;
      var pos = val >= 0;
      bars[i].material.color.set(pos ? 0x7aa2f7 : 0xe8a94d);
      bars[i].material.emissive.set(pos ? 0x223a66 : 0x66441a);
    }
  }

  function buildEmbedding() {
    addTitle('③ Embedding 查表', X.embed, 5.6, ACCENT.input);
    // 词表架：背景一块大板（Embedding 阶段会刷新成真实向量数值）
    var board = makePanel([
      { t: 'Embedding 矩阵 E', c: '#8b96ad' },
      { t: '词表 160,000 × d_model', c: '#6a7488' },
      { t: '查表：id → 向量', c: '#e8a94d' }
    ], 5.2, ACCENT.input);
    board.position.set(X.embed, 3.6, -2.6);
    S.root.add(board);
    S.embedBoard = board;
    // 上下文 token 的迷你向量（停在 embedding 站的 8 股向量）
    for (var i = 0; i < IDS.length; i++) {
      var vg = makeVectorGroup(0.5);
      setVectorValues(vg.bars, embed(IDS[i]));
      vg.group.position.set(X.embed + (i - 3.5) * 0.9, PATH_Y, 1.6);
      S.root.add(vg.group);
      var ch = makeLabel(SENT[i], { h: 0.5, color: '#a89a83' });
      ch.position.set(X.embed + (i - 3.5) * 0.9, PATH_Y + 1.1, 1.6);
      S.root.add(ch);
    }
  }

  function buildLayers() {
    for (var l = 0; l < LAYERS_SHOWN; l++) {
      var lx = X.layers[l];
      var grp = new THREE.Group();
      grp.position.set(lx, 0, 0);

      // 层外壳：半透明盒 + 描边
      var frame = new THREE.Mesh(
        new THREE.BoxGeometry(9.6, 6.4, 9),
        new THREE.MeshBasicMaterial({ color: 0x1a2233, transparent: true, opacity: 0.10, depthWrite: false })
      );
      frame.position.y = 3.2;
      grp.add(frame);
      var edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(frame.geometry),
        new THREE.LineBasicMaterial({ color: 0x2e3d5c, transparent: true, opacity: 0.8 })
      );
      edges.position.copy(frame.position);
      grp.add(edges);
      S.layerFrames.push({ frame: frame, edges: edges });

      addTitle('Layer ' + (l + 1) + ' / ' + LAYERS_TOTAL, 0, 0, '#8b96ad')
        .position.set(0, 6.9, 0); // addTitle 加到 root 了，移进 grp
      var title = S.root.children[S.root.children.length - 1];
      S.root.remove(title); grp.add(title);

      // 注意力子站（前侧）：标题
      var at = makeLabel('MLA 注意力', { h: 0.55, color: ACCENT.attn });
      at.position.set(-2.8, 1.1, 3.6);
      grp.add(at);
      // MoE 子站（后侧）：标题
      var mt = makeLabel('MoE · 896 专家 → 激活 16', { h: 0.55, color: '#9df3dd' });
      mt.position.set(1.4, 6.0, -3.6);
      grp.add(mt);

      S.root.add(grp);
      S.layerGroups.push(grp);
    }

    // 专家矩阵墙（可复用，移动到当前激活层）：32×28 = 896 格
    var geo = new THREE.BoxGeometry(0.2, 0.2, 0.06);
    var mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    var grid = new THREE.InstancedMesh(geo, mat, EXPERTS);
    var dummy = new THREE.Object3D();
    var cols = 32;
    for (var e = 0; e < EXPERTS; e++) {
      var cx = (e % cols - cols / 2 + 0.5) * 0.235;
      var cy = (Math.floor(e / cols) - 14 + 0.5) * 0.235;
      dummy.position.set(cx + 1.4, 3.4 + cy, -3.9);
      dummy.updateMatrix();
      grid.setMatrixAt(e, dummy.matrix);
      grid.setColorAt(e, new THREE.Color(0x1c453b));
      S.expertBase.push(0x1c453b);
    }
    grid.instanceMatrix.needsUpdate = true;
    grid.instanceColor.needsUpdate = true;
    grid.visible = false;
    S.root.add(grid);
    S.expertGrid = grid;

    // 注意力上下文组（可复用）：8 个小块 + 连线到主角
    for (var i = 0; i < IDS.length; i++) {
      var b = new THREE.Mesh(
        new THREE.BoxGeometry(0.4, 0.4, 0.4),
        new THREE.MeshStandardMaterial({ color: 0x7aa2f7, emissive: 0x1c2f55, roughness: 0.5 })
      );
      b.visible = false;
      S.root.add(b);
      var lb = makeLabel(SENT[i], { h: 0.45, color: '#c9cede' });
      lb.visible = false;
      S.root.add(lb);
      var line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
        new THREE.LineBasicMaterial({ color: 0x7aa2f7, transparent: true, opacity: 0 })
      );
      S.root.add(line);
      S.ctxTokens.push({ box: b, label: lb });
      S.ctxLines.push(line);
    }
  }

  function buildDeepGate() {
    addTitle('④ …再重复 57 层…', X.deep, 5.6, '#8b96ad');
    // 两根立柱 + 半透明门
    var post = new THREE.BoxGeometry(0.3, 6, 0.3);
    var pm = new THREE.MeshStandardMaterial({ color: 0x2e3d5c, roughness: 0.6 });
    [-1.6, 1.6].forEach(function (dz) {
      var p = new THREE.Mesh(post, pm);
      p.position.set(X.deep, 3, dz);
      S.root.add(p);
    });
    var door = new THREE.Mesh(
      new THREE.PlaneGeometry(0.15, 3.4),
      new THREE.MeshBasicMaterial({ color: 0x7aa2f7, transparent: true, opacity: 0.18, side: THREE.DoubleSide })
    );
    door.position.set(X.deep, PATH_Y + 0.5, 0);
    S.root.add(door);
    S.deepGlow = door;
    var lb = makeLabel('× 57', { h: 1.1, color: '#7aa2f7' });
    lb.position.set(X.deep, 4.6, 0);
    S.root.add(lb);
  }

  function buildOutput() {
    addTitle('⑤ 输出头 → 采样', X.output, 6.4, ACCENT.out);
    // 11 个候选字的概率柱
    for (var i = 0; i < OUT_CH.length; i++) {
      var b = new THREE.Mesh(
        new THREE.BoxGeometry(0.5, 1, 0.5),
        new THREE.MeshStandardMaterial({ color: 0xc8b06a, emissive: 0x3d3319, roughness: 0.45 })
      );
      b.position.set(X.output + (i - 5) * 0.85, 0.55, 0);
      b.scale.y = 0.05;
      S.root.add(b);
      S.outBars.push(b);
      var lb = makeLabel(OUT_CH[i], { h: 0.72, color: '#e8e4dc' });
      lb.position.set(X.output + (i - 5) * 0.85, 0.62, 0.8);
      S.root.add(lb);
      S.outLabels.push(lb);
    }
    // 采结果大字
    var sp = makeLabel('?', { h: 2.2, color: '#ffffff' });
    sp.position.set(X.output, 8.4, 0);
    sp.visible = false;
    S.root.add(sp);
    S.sampledSprite = sp;
  }

  function buildToken() {
    var vg = makeVectorGroup(1.0);
    S.token = vg.group; S.tokenBars = vg.bars;
    S.token.position.set(X.embed, PATH_Y, 0);
    S.root.add(S.token);
    var lb = makeLabel('K', { h: 0.7, color: '#ffffff', bg: 'rgba(60,80,140,0.85)' });
    lb.position.set(0, 1.6, 0);
    S.token.add(lb);
    S.tokenLabel = lb;
  }

  // 可刷新的信息面板（每层数字详情），放到舞台侧翼避免遮挡主角
  function setInfoPanel(lines, accent, x, y, z, w) {
    if (S.infoPanel) {
      S.root.remove(S.infoPanel);
      S.infoPanel.material.map.dispose();
      S.infoPanel.material.dispose();
    }
    S.infoPanel = makePanel(lines, w || 5.2, accent);
    S.infoPanel.position.set(x, y, z == null ? 4.6 : z);
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
      new THREE.MeshBasicMaterial({ color: 0x223a66, transparent: true, opacity: 0.55 })
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

  function dimExperts() {
    if (!S.expertGrid) return;
    var c = new THREE.Color(0x1c453b);
    for (var e = 0; e < EXPERTS; e++) S.expertGrid.setColorAt(e, c);
    S.expertGrid.instanceColor.needsUpdate = true;
  }

  function lightExperts(layerData, p) {
    // top-16 按权重错峰点亮
    var n = Math.floor(seg(p, 0, 1) * TOPK);
    for (var j = 0; j < n; j++) {
      var ei = layerData.top[j];
      var w = layerData.w[j];
      var col = new THREE.Color().setHSL(0.46, 0.9, 0.38 + w * 2.5);
      S.expertGrid.setColorAt(ei, col);
    }
    S.expertGrid.instanceColor.needsUpdate = true;
  }

  function showContext(layerData, p, layerX) {
    // 上下文小块排一排，连线透明度 = 注意力权重（真实）
    var n = journey.step + 1;
    for (var i = 0; i < IDS.length; i++) {
      var on = i < n;
      var ct = S.ctxTokens[i];
      ct.box.visible = on; ct.label.visible = on;
      var line = S.ctxLines[i];
      if (!on) { line.material.opacity = 0; continue; }
      var px = layerX - 3.2 + i * 0.85;
      ct.box.position.set(px, PATH_Y + 0.4, 2.2);
      ct.label.position.set(px, PATH_Y + 1.15, 2.2);
      var w = layerData.attn[i];
      var grow = seg(p, 0, 0.6);
      line.material.opacity = w * 3.2 * grow;
      var pts = [new THREE.Vector3(px, PATH_Y + 0.4, 2.2), S.token.position.clone()];
      line.geometry.setFromPoints(pts);
    }
  }
  function hideContext() {
    S.ctxTokens.forEach(function (ct) { ct.box.visible = false; ct.label.visible = false; });
    S.ctxLines.forEach(function (l) { l.material.opacity = 0; });
  }

  // ---------- 阶段定义 ----------
  var stages = [
    { // 0 全景
      dur: 4.0,
      enter: function () {
        caption('Kimi K3：一个 token 的完整旅程',
          '输入 → 分词 → Embedding → 61 层（此处展示 4 层）→ 输出 · d=8 教学维度，数字全部为真实前向计算');
        flyTo(12, 2.5, 0, 52, 0.65, 1.22);
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
          { t: 'E[' + IDS[journey.step] + '] =（查表命中）', c: '#8b96ad' },
          { t: '[' + x.map(function (v) { return fmt(v, 1); }).join(' ') + ']', c: '#e8a94d' },
          { t: '每一维对应一根柱子（蓝正橙负）', c: '#6a7488' }
        ]);
        morphFrom = null;
      },
      tick: function (p) {
        // 主角向量柱子从 0 长出
        var t = ease(seg(p, 0.15, 0.7));
        setVectorValues(S.tokenBars, journey.x0.map(function (v) { return v * t; }));
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
        S.token.position.x = x;
        S.deepGlow.material.opacity = 0.18 + 0.5 * Math.sin(p * Math.PI);
      }
    },
    { // 9 输出
      dur: 7.0,
      enter: function () {
        caption('⑤ 输出头：logits → softmax → top-p 采样',
          '向量与词表投影相乘，得到每个候选字的分数');
        flyTo(X.output, 3.4, 0, 13, 0.3);
        S.deepGlow.material.opacity = 0.18;
      },
      tick: function (p) {
        // token 入场
        var tin = ease(seg(p, 0, 0.12));
        S.token.position.x = X.deep + (X.output - 1.5 - X.deep) * tin;
        S.token.position.z = -1.8 * tin; // 退到概率柱后方，不遮挡
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
          S.outBars[i].material.color.set(dim ? 0x4a4a4a : 0xc8b06a);
          S.outBars[i].material.emissive.set(dim ? 0x111111 : 0x3d3319);
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
    return {
      dur: 6.0,
      enter: function () {
        var ld = journey.layers[l];
        var best = 0;
        for (var i = 1; i < ld.attn.length; i++) if (ld.attn[i] > ld.attn[best]) best = i;
        caption(
          'Layer ' + (l + 1) + '/' + LAYERS_TOTAL + '：注意力 → MoE → 残差',
          '「' + SENT[journey.step] + '」最关注「' + SENT[best] + '」(' + ld.attn[best].toFixed(2) +
          ') · 896 专家激活 16 个：#' + ld.top[0] + ' w=' + ld.w[0].toFixed(2) + '、#' +
          ld.top[1] + ' w=' + ld.w[1].toFixed(2) + ' …'
        );
        flyTo(lx, 3.2, 0, 13, 0.32);
        // 专家墙移入本层
        S.expertGrid.position.set(lx, 0, 0);
        S.expertGrid.visible = true;
        dimExperts();
        // 高亮当前层外壳
        S.layerFrames.forEach(function (lf, i) {
          lf.edges.material.color.set(i === l ? 0x7aa2f7 : 0x2e3d5c);
          lf.edges.material.opacity = i === l ? 1 : 0.8;
        });
        morphFrom = null;
        setInfoPanel([
          { t: 'attn = softmax(q·K/√d)', c: ACCENT.attn },
          { t: '  [' + ld.attn.map(function (a) { return a.toFixed(2); }).join(' ') + ']', c: '#c9cede' },
          { t: 'top-16 专家权重 Σ=' + ld.w.reduce(function (a, b) { return a + b; }, 0).toFixed(2), c: ACCENT.moe },
          { t: 'y = Σ wᵢ·Eᵢ(x)，残差相加后继续', c: '#6a7488' }
        ], ACCENT.attn, lx - 3.7, 3.7, -3.8, 4.2);
      },
      tick: function (p) {
        var ld = journey.layers[l];
        // 0-0.15：token 飞入本层
        var tin = ease(seg(p, 0, 0.15));
        S.token.position.x = prevX + (lx - prevX) * tin;
        // 0.15-0.5：注意力（上下文小块 + 连线权重）；MoE 阶段收起，转移视线
        if (p < 0.58) showContext(ld, seg(p, 0.15, 0.5), lx);
        else hideContext();
        // 0.5-0.8：MoE 点亮专家
        lightExperts(ld, seg(p, 0.5, 0.8));
        // 0.8-1.0：向量 morph 到本层输出（残差后的新向量）
        var tm = ease(seg(p, 0.78, 0.98));
        if (tm > 0 && !morphFrom) {
          morphFrom = (l === 0 ? journey.x0 : journey.layers[l - 1].out).slice();
        }
        if (morphFrom) setVectorValues(S.tokenBars, ld.out, morphFrom, tm);
      },
      leave: function () {
        hideContext();
        S.expertGrid.visible = false;
        clearInfoPanel();
        morphFrom = null;
      }
    };
  }

  // ---------- 旅程控制 ----------
  function startJourney(step) {
    journey = computeJourney(step);
    // 重置可动对象
    S.token.position.set(X.embed, PATH_Y, 0);
    setVectorValues(S.tokenBars, journey.x0.map(function () { return 0.06; }));
    // token 顶上的字符牌
    S.token.remove(S.tokenLabel);
    S.tokenLabel = makeLabel(SENT[step], { h: 0.7, color: '#ffffff', bg: 'rgba(60,80,140,0.85)' });
    S.tokenLabel.position.set(0, 1.6, 0);
    S.token.add(S.tokenLabel);
    S.charTiles.forEach(function (t) { t.scale.setScalar(0.001); });
    S.idTiles.forEach(function (t) { t.scale.setScalar(0.001); });
    S.outBars.forEach(function (b) { b.scale.y = 0.05; b.position.y = 0.3; });
    S.sampledSprite.visible = false;
    S.sampledSprite.material.opacity = 1;
    hideContext();
    dimExperts();
    S.expertGrid.visible = false;
    clearInfoPanel();
    stageIdx = 0; stageT = 0;
    stages[0].enter();
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
        stageIdx++;
        stageT = 0;
        if (stageIdx >= stages.length) {
          // 旅程结束：主角推进到下一个 token，循环
          curStep = (curStep + 1) % IDS.length;
          if (curStep === 0) curStep = 1; // 至少留 1 个上下文
          startJourney(curStep);
        } else {
          stages[stageIdx].enter();
        }
      }
    }
    // 主角呼吸浮动（即使暂停也有一点生命力）
    if (S.token) {
      S.token.position.y = PATH_Y + Math.sin(clock * 1.8) * 0.06;
    }
  }

  return {
    init: function (scene) {
      S.scene = scene;
      S.root = new THREE.Group();
      scene.add(S.root);
      buildFloorPath();
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
    debugGo: function (i) {
      if (stages[stageIdx].leave) stages[stageIdx].leave();
      stageIdx = Math.max(0, Math.min(stages.length - 1, i));
      stageT = 0;
      stages[stageIdx].enter();
      // 快进一帧让 tick(0) 生效
      if (stages[stageIdx].tick) stages[stageIdx].tick(0.0001);
    },
    stageCount: function () { return stages.length; }
  };
})();
