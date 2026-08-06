/**
 * FFN 模块 — Feed-Forward Network 3D 可视化
 *
 * 真实结构：输入向量 x → [W1 矩阵] → 隐藏层 h = GELU(W1·x + b1) → [W2 矩阵] → 输出向量 y
 * - 每层神经元 = 小球，颜色编码激活值（蓝=负，红=正），亮度编码幅值
 * - 权重矩阵 = 两层之间的连线（LineSegments 顶点着色，红=正权重，蓝=负权重）
 * - 数据流 = 脉冲小球沿层间传播 + 向量变形动画（隐藏层/输出层按激活前序逐个"长"出来）
 *
 * 接口（MODULE_SPEC v1）：init / show / hide / update / dispose / screenshot
 * 数据：show({ input:[...], hidden:[...], output:[...] }) — 缺省时用内置权重真实计算
 */
window.FFN = (function () {
  'use strict';

  // ---------- 可调常量 ----------
  var LAYER_X = { input: -6, hidden: 0, output: 6 }; // 三层沿 X 轴排布
  var MAX_IN = 16;   // 输入/输出层最多显示的维度
  var MAX_HID = 32;  // 隐藏层最多显示的维度（比输入大，体现升维）
  var SPHERE_R = 0.22;
  var SPHERE_GAP = 0.62;
  var LOOP_T = 9;    // 一个数据流循环的时长（秒）

  var scene = null, camera = null, renderer = null;
  var group = null;            // 模块根节点（show/hide 控制）
  var layers = { input: [], hidden: [], output: [] }; // 每层的小球 mesh 数组
  var w1Lines = null, w2Lines = null;                  // 权重连线
  var pulses = [];             // 数据流脉冲小球
  var labels = [];
  var time = 0;
  var built = false;

  // ---------- 工具 ----------

  // 确定性伪随机（保证每次 show 默认数据一致）
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gelu(x) {
    return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
  }

  // 激活值 → 颜色：负=蓝，零=深灰，正=红橙；返回 THREE.Color
  function valueColor(v) {
    var c = new THREE.Color();
    if (v >= 0) c.setRGB(0.15 + 0.85 * v, 0.12 + 0.25 * v, 0.08);
    else c.setRGB(0.08, 0.15 + 0.25 * (-v), 0.15 + 0.85 * (-v));
    return c;
  }

  // 文字标签（canvas → sprite）
  function makeLabel(text, scale) {
    var cv = document.createElement('canvas');
    var ctx = cv.getContext('2d');
    var font = 'bold 42px -apple-system, "PingFang SC", sans-serif';
    ctx.font = font;
    cv.width = Math.ceil(ctx.measureText(text).width) + 24;
    cv.height = 64;
    ctx = cv.getContext('2d');
    ctx.font = font;
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 12, 34);
    var tex = new THREE.CanvasTexture(cv);
    tex.anisotropy = 4;
    var mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
    var sp = new THREE.Sprite(mat);
    var s = scale || 1;
    sp.scale.set(cv.width / cv.height * 0.55 * s, 0.55 * s, 1);
    return sp;
  }

  // 一列神经元小球：vals 为归一化激活值，返回 mesh 数组
  // 层高超限时间距自动收紧，保证任意维度都能完整落入视野
  var MAX_LAYER_H = 12.5;
  function buildLayer(x, vals, radius) {
    var meshes = [];
    var n = vals.length;
    var gap = Math.min(SPHERE_GAP, n > 1 ? MAX_LAYER_H / (n - 1) : SPHERE_GAP);
    var geo = new THREE.SphereGeometry(radius, 32, 24);
    var y0 = -(n - 1) * gap / 2;
    for (var i = 0; i < n; i++) {
      var col = valueColor(vals[i]);
      var mat = new THREE.MeshStandardMaterial({
        color: col,
        emissive: col.clone().multiplyScalar(0.35 * Math.abs(vals[i])),
        roughness: 0.32,
        metalness: 0.15
      });
      var m = new THREE.Mesh(geo, mat);
      m.position.set(x, y0 + i * gap, 0);
      m.castShadow = true;
      m.userData.baseColor = col;
      m.userData.value = vals[i];
      group.add(m);
      meshes.push(m);
    }
    return meshes;
  }

  // 权重矩阵连线：from 层每个神经元连到 to 层每个神经元，顶点色编码权重符号/强度
  function buildWeights(fromMeshes, toMeshes, W) {
    var fn = fromMeshes.length, tn = toMeshes.length;
    var pos = new Float32Array(fn * tn * 6);
    var col = new Float32Array(fn * tn * 6);
    var k = 0, c = new THREE.Color();
    for (var i = 0; i < fn; i++) {
      for (var j = 0; j < tn; j++) {
        var w = W[i][j]; // [-1,1]
        var a = Math.abs(w);
        if (w >= 0) c.setRGB(0.25 + 0.75 * a, 0.15, 0.12);
        else c.setRGB(0.12, 0.2, 0.25 + 0.75 * a);
        pos[k]     = fromMeshes[i].position.x; pos[k + 1] = fromMeshes[i].position.y; pos[k + 2] = 0;
        pos[k + 3] = toMeshes[j].position.x;   pos[k + 4] = toMeshes[j].position.y;   pos[k + 5] = 0;
        col[k] = c.r * a; col[k + 1] = c.g * a; col[k + 2] = c.b * a;
        col[k + 3] = c.r * a; col[k + 4] = c.g * a; col[k + 5] = c.b * a;
        k += 6;
      }
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    var mat = new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.2,
      blending: THREE.AdditiveBlending, depthWrite: false
    });
    var lines = new THREE.LineSegments(geo, mat);
    group.add(lines);
    return lines;
  }

  // ---------- 数据 ----------

  // 均匀抽样把真实高维向量降到可显示维度
  function sampleDims(arr, maxN) {
    if (arr.length <= maxN) return arr.slice();
    var out = [], step = arr.length / maxN;
    for (var i = 0; i < maxN; i++) out.push(arr[Math.floor(i * step)]);
    return out;
  }

  function normalize(arr) {
    var m = 1e-9;
    for (var i = 0; i < arr.length; i++) m = Math.max(m, Math.abs(arr[i]));
    return arr.map(function (v) { return v / m; });
  }

  // 生成默认数据：随机 W1/b1/W2/b2 + 输入，真实做矩阵乘 + GELU
  function defaultData() {
    var rnd = mulberry32(42);
    var D = 16, H = 32;
    function vec(n, s) { var v = []; for (var i = 0; i < n; i++) v.push((rnd() * 2 - 1) * s); return v; }
    function mat(r, c, s) { var m = []; for (var i = 0; i < r; i++) m.push(vec(c, s)); return m; }
    var input = vec(D, 1);
    var W1 = mat(D, H, 0.6), b1 = vec(H, 0.3);
    var W2 = mat(H, D, 0.4), b2 = vec(D, 0.2);
    var hidden = [], output = [];
    for (var j = 0; j < H; j++) {
      var s = b1[j];
      for (var i = 0; i < D; i++) s += input[i] * W1[i][j];
      hidden.push(gelu(s));
    }
    for (var i2 = 0; i2 < D; i2++) {
      var s2 = b2[i2];
      for (var j2 = 0; j2 < H; j2++) s2 += hidden[j2] * W2[j2][i2];
      output.push(s2);
    }
    return { input: input, hidden: hidden, output: output, W1: W1, W2: W2 };
  }

  // 权重也按 sampleDims 同样的索引抽样，保证连线和显示维度一致
  function sampleWeights(W, fromLen, toLen) {
    var fi = [], tj = [], i;
    for (i = 0; i < fromLen; i++) fi.push(W.length <= fromLen ? i : Math.floor(i * W.length / fromLen));
    var cols = W[0].length;
    for (i = 0; i < toLen; i++) tj.push(cols <= toLen ? i : Math.floor(i * cols / toLen));
    var out = [];
    for (var a = 0; a < fromLen; a++) {
      var row = [];
      for (var b = 0; b < toLen; b++) row.push(Math.max(-1, Math.min(1, W[fi[a]][tj[b]] * 2.5)));
      out.push(row);
    }
    return out;
  }

  // ---------- 场景搭建 ----------

  function addLights() {
    var amb = new THREE.AmbientLight(0x8899bb, 0.55);
    group.add(amb);
    var key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(4, 12, 8);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -14; key.shadow.camera.right = 14;
    key.shadow.camera.top = 14; key.shadow.camera.bottom = -14;
    key.shadow.radius = 4;
    group.add(key);
    var rim = new THREE.DirectionalLight(0x6688ff, 0.5);
    rim.position.set(-6, 4, -10);
    group.add(rim);
  }

  function addGround(bottomY) {
    var geo = new THREE.PlaneGeometry(60, 30);
    var mat = new THREE.MeshStandardMaterial({ color: 0x0c0f16, roughness: 0.95, metalness: 0 });
    var g = new THREE.Mesh(geo, mat);
    g.rotation.x = -Math.PI / 2;
    g.position.y = bottomY - 1.2;
    g.receiveShadow = true;
    group.add(g);
    // 细腻网格
    var grid = new THREE.GridHelper(40, 40, 0x223044, 0x16202e);
    grid.position.y = bottomY - 1.19;
    grid.material.transparent = true;
    grid.material.opacity = 0.5;
    group.add(grid);
  }

  function build(data) {
    // 清理旧内容
    while (group.children.length) {
      var ch = group.children.pop();
      disposeObject(ch);
    }
    layers = { input: [], hidden: [], output: [] };
    pulses = []; labels = [];

    var d = data && data.input ? data : defaultData();
    var vin = normalize(sampleDims(d.input, MAX_IN));
    var vhid = normalize(sampleDims(d.hidden || [], MAX_HID));
    var vout = normalize(sampleDims(d.output || [], MAX_IN));
    if (!vhid.length) vhid = normalize(defaultData().hidden);

    addLights();
    addGround(-MAX_LAYER_H / 2);

    layers.input = buildLayer(LAYER_X.input, vin, SPHERE_R);
    layers.hidden = buildLayer(LAYER_X.hidden, vhid, SPHERE_R * 0.85);
    layers.output = buildLayer(LAYER_X.output, vout, SPHERE_R);

    var dd = defaultData();
    var W1 = (d.W1) ? d.W1 : dd.W1;
    var W2 = (d.W2) ? d.W2 : dd.W2;
    w1Lines = buildWeights(layers.input, layers.hidden, sampleWeights(W1, vin.length, vhid.length));
    w2Lines = buildWeights(layers.hidden, layers.output, sampleWeights(W2, vhid.length, vout.length));

    // 数据流脉冲小球（发光）
    var pgeo = new THREE.SphereGeometry(0.12, 16, 12);
    for (var p = 0; p < 8; p++) {
      var pmat = new THREE.MeshBasicMaterial({ color: 0xffe9a8, transparent: true, opacity: 0 });
      var pm = new THREE.Mesh(pgeo, pmat);
      group.add(pm);
      pulses.push(pm);
    }

    // 标签（以最高层为参照，固定在视野内）
    var yH = MAX_LAYER_H / 2;
    var specs = [
      ['x · 输入 (' + d.input.length + 'd)', LAYER_X.input, yH + 1.1],
      ['W₁ 升维', (LAYER_X.input + LAYER_X.hidden) / 2, yH + 0.4],
      ['h = GELU(W₁x+b₁) · 隐藏层 (' + (d.hidden ? d.hidden.length : vhid.length) + 'd)', LAYER_X.hidden, yH + 1.1],
      ['W₂ 降维', (LAYER_X.hidden + LAYER_X.output) / 2, yH + 0.4],
      ['y = W₂h+b₂ · 输出 (' + (d.output ? d.output.length : vout.length) + 'd)', LAYER_X.output, yH + 1.1]
    ];
    specs.forEach(function (s) {
      var lb = makeLabel(s[0], 0.9);
      lb.position.set(s[1], s[2], 0);
      group.add(lb);
      labels.push(lb);
    });

    // 初始状态：隐藏层/输出层待"生长"
    time = 0;
    built = true;
    applyPhase(0);
  }

  function disposeObject(obj) {
    obj.traverse(function (o) {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        var mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach(function (m) { if (m.map) m.map.dispose(); m.dispose(); });
      }
    });
  }

  // ---------- 动画 ----------

  // 缓动
  function ease(t) { return t * t * (3 - 2 * t); }

  /**
   * 相位机（LOOP_T 秒一循环）：
   * 0–1.2s   输入向量波浪脉动
   * 1.2–3.6s 脉冲流向隐藏层，隐藏层小球按序"变形生长"（缩放 0→1，颜色从灰→激活色）
   * 3.6–4.6s GELU 闪光（隐藏层高亮）
   * 4.6–7.0s 脉冲流向输出层，输出层生长
   * 7.0–9s   整体保持 + 呼吸
   */
  function applyPhase(t) {
    var i, m, f;
    var nH = layers.hidden.length, nO = layers.output.length;

    // 隐藏层生长进度
    var hGrow = Math.max(0, Math.min(1, (t - 1.2) / 2.4));
    // 输出层生长进度
    var oGrow = Math.max(0, Math.min(1, (t - 4.6) / 2.4));

    // 输入层：始终可见，前 1.2s 波浪脉动
    for (i = 0; i < layers.input.length; i++) {
      m = layers.input[i];
      var wave = t < 1.2 ? 1 + 0.35 * Math.sin(t * 10 - i * 0.6) * (1 - t / 1.2) : 1;
      m.scale.setScalar(wave);
    }

    // 隐藏层：逐个生长 + GELU 闪光
    for (i = 0; i < nH; i++) {
      m = layers.hidden[i];
      var local = Math.max(0, Math.min(1, hGrow * nH * 0.35 + hGrow * 0.65 - i / nH * 0.9 + 0.1));
      f = ease(Math.max(0, Math.min(1, local * 1.4)));
      var flash = (t > 3.6 && t < 4.6) ? 0.8 * Math.sin((t - 3.6) * Math.PI) : 0;
      m.scale.setScalar(Math.max(0.001, f * (1 + 0.3 * flash)));
      var col = m.userData.baseColor;
      m.material.color.copy(col).multiplyScalar(0.25 + 0.75 * f);
      m.material.emissive.copy(col).multiplyScalar((0.4 * f + flash) * Math.abs(m.userData.value) + flash * 0.15);
    }

    // 输出层：逐个生长
    for (i = 0; i < nO; i++) {
      m = layers.output[i];
      var l2 = Math.max(0, Math.min(1, oGrow * nO * 0.35 + oGrow * 0.65 - i / nO * 0.9 + 0.1));
      var f2 = ease(Math.max(0, Math.min(1, l2 * 1.4)));
      m.scale.setScalar(Math.max(0.001, f2));
      var c2 = m.userData.baseColor;
      m.material.color.copy(c2).multiplyScalar(0.25 + 0.75 * f2);
      m.material.emissive.copy(c2).multiplyScalar(0.4 * f2 * Math.abs(m.userData.value));
    }

    // 权重连线：流动阶段加亮
    w1Lines.material.opacity = 0.1 + 0.28 * Math.max(0, Math.sin(Math.min(1, hGrow) * Math.PI));
    w2Lines.material.opacity = 0.1 + 0.28 * Math.max(0, Math.sin(Math.min(1, oGrow) * Math.PI));

    // 脉冲小球：阶段1 input→hidden，阶段2 hidden→output
    for (i = 0; i < pulses.length; i++) {
      var pm = pulses[i];
      var off = i / pulses.length;
      if (t > 1.2 && t < 3.6) {
        var pr = ((t - 1.2) / 2.4 + off) % 1;
        pm.position.set(LAYER_X.input + (LAYER_X.hidden - LAYER_X.input) * pr,
          Math.sin(off * 40) * 3.5 * Math.sin(pr * Math.PI), Math.cos(off * 31) * 1.5);
        pm.material.opacity = Math.sin(pr * Math.PI) * 0.95;
      } else if (t > 4.6 && t < 7.0) {
        var pr2 = ((t - 4.6) / 2.4 + off) % 1;
        pm.position.set(LAYER_X.hidden + (LAYER_X.output - LAYER_X.hidden) * pr2,
          Math.sin(off * 40) * 3.5 * Math.sin(pr2 * Math.PI), Math.cos(off * 31) * 1.5);
        pm.material.opacity = Math.sin(pr2 * Math.PI) * 0.95;
      } else {
        pm.material.opacity = 0;
      }
    }
  }

  // ---------- 公开接口 ----------

  return {
    init: function (s, c, r) {
      scene = s; camera = c; renderer = r;
      group = new THREE.Group();
      scene.add(group);
    },

    show: function (data) {
      if (!group) return;
      build(data || null);
      group.visible = true;
    },

    hide: function () {
      if (group) group.visible = false;
    },

    update: function (delta) {
      if (!built || !group.visible) return;
      time = (time + delta) % LOOP_T;
      applyPhase(time);
      // 呼吸感：整体极缓浮动
      group.position.y = Math.sin(time * 0.7) * 0.08;
    },

    dispose: function () {
      if (!group) return;
      disposeObject(group);
      scene.remove(group);
      group = null;
      built = false;
      layers = { input: [], hidden: [], output: [] };
      pulses = []; labels = [];
      w1Lines = w2Lines = null;
    },

    screenshot: function () {
      renderer.render(scene, camera);
      return renderer.domElement.toDataURL('image/png');
    }
  };
})();
