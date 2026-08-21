/* numbers.js — 真实数值映射：玩具维度（d=8）但每个数字都是真算的
 * SIM：确定性伪随机前向传播模拟器（种子固定，刷新页面数字不变）
 * 数据流：token id → embedding 查表 → KDA 状态矩阵 S 真更新 → MLA 注意力分数真 softmax
 *        → 门控 896 真打分选 16 → 加权求和真算出 y → logits 真点积 → softmax/温度/top-p → 采样
 * 面板：挂到对应组件旁的数值牌（CanvasTexture Sprite），随 SIM.step 刷新
 */
window.ENGINE = window.ENGINE || {};
(function () {
  var E = window.ENGINE;

  // ---------- 确定性伪随机（mulberry32） ----------
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

  // ---------- 词表与输入 ----------
  var SENT = '月之暗面发布了k3开源大模型';
  function tokenId(ch, i) {
    var s = 0;
    for (var j = 0; j < ch.length; j++) s = (s * 131 + ch.charCodeAt(j)) >>> 0;
    return (s * 997 + i * 7919) % 150000 + 1000;
  }
  var IDS = SENT.split('').map(function (ch, i) { return tokenId(ch, i); });
  var D = 8, EXPERTS = 896, TOPK = 16;

  // 预生成：embedding 表（按 id 种子）、896 专家权重向量、输出投影
  var embedCache = {};
  function embed(id) {
    if (!embedCache[id]) {
      var r = rng(id * 7919), v = [];
      for (var i = 0; i < D; i++) v.push(gauss(r));
      embedCache[id] = v;
    }
    return embedCache[id];
  }
  var expW = [];
  (function () {
    var r = rng(20260716);
    for (var e = 0; e < EXPERTS; e++) { var v = []; for (var i = 0; i < D; i++) v.push(gauss(r) * 0.6); expW.push(v); }
  })();
  var outW = [];
  (function () {
    IDS.concat([999, 888, 777]).forEach(function (id) {
      var r = rng(id * 31 + 7), v = [];
      for (var i = 0; i < D; i++) v.push(gauss(r));
      outW.push({ id: id, ch: null, w: v });
    });
  })();
  var OUT_CH = ['的', '是', '了', '在', '和', '开', '源', '模', '型', '。', '，'];
  for (var oi = 0; oi < OUT_CH.length; oi++) outW[oi].ch = OUT_CH[oi];

  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function softmax(z) {
    var m = Math.max.apply(null, z), ex = z.map(function (v) { return Math.exp(v - m); });
    var s = ex.reduce(function (a, b) { return a + b; }, 0);
    return ex.map(function (v) { return v / s; });
  }
  function fmt(v, n) { return (v >= 0 ? ' ' : '') + v.toFixed(n == null ? 2 : n); }

  // ---------- SIM：随 step 推进的真实计算 ----------
  var SIM = window.SIM = {
    step: 0,                       // 0..7：当前处理到第几个 token
    sent: SENT, ids: IDS,
    kdaS: null,                    // 4×4 状态矩阵（真实累积）
    data: [],
    reset: function () { SIM.step = 0; SIM.kdaS = null; SIM.data = []; SIM.compute(); },
    compute: function () {
      // 重算 0..step 全链路（玩具维度，重算成本可忽略）
      var beta = 0.35;
      var S = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; // 4x4 行主序
      SIM.data = [];
      for (var t = 0; t <= SIM.step; t++) {
        var id = IDS[t], x = embed(id);
        // KDA：delta rule 真更新（取 x 前 4 维做 k、后 4 维做 v）
        var k = x.slice(0, 4), v = x.slice(4, 8);
        for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) {
          S[i * 4 + j] = S[i * 4 + j] * (1 - beta * k[i] * k[j]) + beta * v[i] * k[j];
        }
        // MLA 注意力分数：q=x_t，k_i=x_i，真 softmax
        var scores = [], q = x;
        for (var u = 0; u <= t; u++) scores.push(dot(q, embed(IDS[u])) / Math.sqrt(D));
        var attn = softmax(scores);
        // 门控：896 真打分 → top16 → 归一化权重
        var gs = [];
        for (var e = 0; e < EXPERTS; e++) gs.push(dot(x, expW[e]));
        var idx = gs.map(function (s, i) { return i; }).sort(function (a, b) { return gs[b] - gs[a]; });
        var top = idx.slice(0, TOPK);
        var w = softmax(top.map(function (i) { return gs[i]; }));
        // 聚合：y = Σ w_i · E_i(x)，E_i(x) = tanh(x·expW_i) * expW_i（玩具专家）
        var y = [0, 0, 0, 0, 0, 0, 0, 0];
        top.forEach(function (ei, j) {
          var act = Math.tanh(dot(x, expW[ei]));
          for (var d = 0; d < D; d++) y[d] += w[j] * act * expW[ei][d];
        });
        SIM.data.push({ t: t, id: id, ch: SENT[t], x: x, S: S.slice(), attn: attn, top: top, w: w, y: y });
      }
      // 输出：logits = y·W_out（对 11 个候选字）
      var cur = SIM.data[SIM.data.length - 1];
      SIM.logits = outW.slice(0, 11).map(function (o) { return dot(cur.y, o.w) * 4; });
      SIM.outCh = outW.slice(0, 11).map(function (o) { return o.ch; });
    },
    probs: function (temp, topP) {
      var z = SIM.logits.map(function (v) { return v / (temp || 1); });
      var p = softmax(z);
      if (topP != null && topP < 1) {
        var order = p.map(function (v, i) { return i; }).sort(function (a, b) { return p[b] - p[a]; });
        var keep = {}, acc = 0;
        for (var i = 0; i < order.length; i++) { acc += p[order[i]]; keep[order[i]] = 1; if (acc >= topP) break; }
        var s2 = 0; p = p.map(function (v, i) { return keep[i] ? v : 0; });
        p.forEach(function (v) { s2 += v; });
        p = p.map(function (v) { return v / s2; });
      }
      return p;
    }
  };

  // ---------- 数值面板（Sprite） ----------
  // 类型色：与引擎视觉语言一致
  var TYPE_ACCENT = {
    tokenizer: '#e8a94d', embedding: '#e8a94d', kda: '#e8a94d',
    mla: '#7aa2f7', router: '#46c8ae', experts: '#46c8ae', softmax: '#a89a83'
  };

  function panel(lines, w, h, accent) {
    var c = document.createElement('canvas');
    c.width = 512; c.height = 64 * lines.length + 24;
    var g = c.getContext('2d');
    g.fillStyle = 'rgba(23,19,16,0.92)'; g.fillRect(0, 0, c.width, c.height);
    g.strokeStyle = accent || '#4a4036'; g.lineWidth = 4; g.strokeRect(2, 2, c.width - 4, c.height - 4);
    g.font = '26px Menlo, monospace'; g.textBaseline = 'middle';
    lines.forEach(function (ln, i) {
      g.fillStyle = ln.c || '#c9bba6';
      g.fillText(ln.t, 18, 44 + i * 64);
    });
    var tex = new THREE.CanvasTexture(c);
    if (THREE.SRGBColorSpace) tex.colorSpace = THREE.SRGBColorSpace;
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    sp.scale.set(w || 8, h || (c.height / c.width * 8), 1);
    return sp;
  }

  var panels = []; // {type, sprite, update(), line}
  var guideLines = null; // THREE.LineSegments 引线组
  function clearPanels() {
    var s = E._state;
    panels.forEach(function (p) {
      if (p.sprite.parent) p.sprite.parent.remove(p.sprite);
      if (p.sprite.material.map) p.sprite.material.map.dispose();
      p.sprite.material.dispose();
    });
    panels = [];
    if (guideLines) {
      s.scene.remove(guideLines);
      guideLines.geometry.dispose();
      guideLines.material.dispose();
      guideLines = null;
    }
  }

  // 每个组件的面板放在塔左侧，与右侧 HTML 信息面板互不遮挡
  var TYPE_Z = { tokenizer: 8, embedding: 5, kda: 2, mla: -1, router: -4, experts: -7, softmax: -10 };
  var PANEL_X = -13.5;
  function attach(type, build) {
    var s = E._state;
    var comps = s.components[type];
    if (!comps || !comps.length) return;
    var accent = TYPE_ACCENT[type] || '#4a4036';
    var offset = new THREE.Vector3(PANEL_X, 0, TYPE_Z[type] == null ? 6 : TYPE_Z[type]);
    var c = comps[0].center.clone().add(offset);
    var sp = panel(build(), 9, null, accent);
    sp.position.copy(c);
    s.scene.add(sp);
    var rec = { type: type, sprite: sp, desiredY: comps[0].center.y, center: comps[0].center.clone() };
    rec.update = function () {
      var np = panel(build(), 9, null, accent);
      sp.material.map.dispose();
      sp.material.map = np.material.map;
      sp.scale.copy(np.scale);
      sp.material.needsUpdate = true;
    };
    panels.push(rec);
  }

  function rebuildGuides() {
    var s = E._state;
    if (guideLines) { s.scene.remove(guideLines); guideLines.geometry.dispose(); guideLines.material.dispose(); }
    var pts = [];
    panels.forEach(function (p) {
      var from = p.center.clone();
      var to = p.sprite.position.clone();
      // 面板在塔左侧：引线从组件左缘出，终点到面板右缘
      from.x -= 1.5;
      to.x += p.sprite.scale.x / 2;
      pts.push(from, to);
    });
    if (!pts.length) return;
    guideLines = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0x6a5c4d, transparent: true, opacity: 0.45 })
    );
    s.scene.add(guideLines);
  }

  function vec8(v) { return '[' + v.map(function (x) { return fmt(x, 1); }).join(' ') + ']'; }

  E.refreshNumbers = function () {
    if (!E._state.scene) return;
    if (E._state.inTraining) { clearPanels(); return; }
    clearPanels();
    var cur = SIM.data[SIM.data.length - 1];
    if (!cur) return;
    attach('tokenizer', function () {
      return [
        { t: '输入: 「' + SIM.sent + '」', c: '#efe6d8' },
        { t: 'token[' + cur.t + '] = 「' + cur.ch + '」  id = ' + cur.id, c: '#e8a94d' }
      ];
    });
    attach('embedding', function () {
      return [
        { t: 'E[' + cur.id + '] =', c: '#a89a83' },
        { t: vec8(cur.x), c: '#e8a94d' },
        { t: '查表：8 维向量（教学维度）', c: '#6f6350' }
      ];
    });
    attach('kda', function () {
      var S = cur.S;
      return [
        { t: 'S_t = S·(I-βkkᵀ)+βvkᵀ  β=0.35', c: '#6f6350' },
        { t: '│' + fmt(S[0]) + ' ' + fmt(S[1]) + ' ' + fmt(S[2]) + ' ' + fmt(S[3]) + '│', c: '#e8a94d' },
        { t: '│' + fmt(S[4]) + ' ' + fmt(S[5]) + ' ' + fmt(S[6]) + ' ' + fmt(S[7]) + '│', c: '#e8a94d' },
        { t: '写入第 ' + (cur.t + 1) + ' 个 token 后的真实状态', c: '#6f6350' }
      ];
    });
    attach('mla', function () {
      var ls = [{ t: 'attn = softmax(q·k/√d)：', c: '#6f6350' }];
      cur.attn.forEach(function (a, i) {
        ls.push({ t: ' → 「' + SIM.sent[i] + '」 ' + fmt(a, 3), c: i === cur.t ? '#7aa2f7' : '#a89a83' });
      });
      return ls.slice(0, 9);
    });
    attach('router', function () {
      var ls = [{ t: 'gate = x·W_g（896 真打分）', c: '#6f6350' }, { t: 'Top-16 专家与权重：', c: '#46c8ae' }];
      cur.top.slice(0, 5).forEach(function (ei, j) {
        ls.push({ t: '  专家#' + ei + '  w=' + fmt(cur.w[j], 3), c: '#46c8ae' });
      });
      ls.push({ t: '  … 共 16 个，Σw = ' + fmt(cur.w.reduce(function (a, b) { return a + b; }, 0), 2), c: '#6f6350' });
      return ls;
    });
    attach('experts', function () {
      return [
        { t: 'y = Σ wᵢ·Eᵢ(x)（16 项真求和）', c: '#6f6350' },
        { t: vec8(cur.y), c: '#46c8ae' },
        { t: '这股向量继续上行 → 输出头', c: '#6f6350' }
      ];
    });
    attach('softmax', function () {
      var sp = (E._state._sampler) || { temperature: 1, topP: 1 };
      var p = SIM.probs(sp.temperature, sp.topP);
      var order = p.map(function (v, i) { return i; }).sort(function (a, b) { return p[b] - p[a]; });
      var ls = [{ t: 'T=' + (sp.temperature || 1).toFixed(2) + '  top-p=' + (sp.topP == null ? 1 : sp.topP).toFixed(2), c: '#6f6350' }];
      order.slice(0, 4).forEach(function (i, j) {
        ls.push({ t: '「' + SIM.outCh[i] + '」 ' + (p[i] * 100).toFixed(1) + '%', c: j === 0 ? '#e8a94d' : '#a89a83' });
      });
      return ls;
    });
    // 垂直防重叠：同一 x/z 列内按组件高度排序，高度过近的面板向上推开（保序、尽量贴近原高度）
    var col = panels.slice().sort(function (a, b) { return a.desiredY - b.desiredY; });
    for (var li = 1; li < col.length; li++) {
      var gap = (col[li - 1].sprite.scale.y + col[li].sprite.scale.y) / 2 + 0.8;
      var minY = col[li - 1].sprite.position.y + gap;
      if (col[li].sprite.position.y < minY) col[li].sprite.position.y = minY;
    }
    rebuildGuides();
  };

  // 采样器联动：外壳调 setSampler 后面板同步刷新
  var origSetSampler = E.setSampler;
  if (typeof origSetSampler === 'function') {
    E.setSampler = function (cfg) {
      var r = origSetSampler.apply(E, arguments);
      try { E.refreshNumbers(); } catch (e) {}
      return r;
    };
  }

  // 调试断言用：暴露当前面板清单（类型/世界坐标/可见性），不对外壳暴露其它内部态
  E._numPanelInfo = function () {
    return panels.map(function (p) {
      return {
        type: p.type,
        pos: p.sprite.position.toArray().map(function (v) { return +v.toFixed(2); }),
        scale: [+p.sprite.scale.x.toFixed(2), +p.sprite.scale.y.toFixed(2)],
        visible: p.sprite.visible
      };
    });
  };

  // 单步推进 API：外壳接按钮
  E.simStep = function (dir) {
    SIM.step = Math.max(0, Math.min(IDS.length - 1, SIM.step + (dir || 1)));
    SIM.compute();
    E.refreshNumbers();
    return SIM.step;
  };
  E.simReset = function () { SIM.reset(); E.refreshNumbers(); };

  // buildScene 后自动挂面板
  var origBuild = E.buildScene;
  E.buildScene = function (spec) {
    var r = origBuild.apply(E, arguments);
    try { SIM.reset(); E.refreshNumbers(); } catch (e) { console.warn('[numbers]', e); }
    return r;
  };
})();
