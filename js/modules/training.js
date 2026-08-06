/**
 * training.js — 训练可视化模块（pretrain / SFT / RL 三阶段）
 *
 * 真实结构：
 *   1. loss（RL 为 reward）曲线 = 3D 折线图，发光标记点沿曲线前进表示训练步
 *   2. 梯度流 = 粒子从 loss 标记点沿弧线飞回权重矩阵（反向传播方向）
 *   3. 参数更新 = 8×8 权重矩阵（立方体，高度/颜色=权重值）随梯度到达实时变化
 *      w ← w - η∇w，顶部向量箭头显示更新方向
 *
 * 接口：init / show / hide / update / dispose / screenshot（见 MODULE_SPEC.md）
 */
(function () {
  'use strict';

  // ---------------------------------------------------------------- 常量

  var CHART = {
    x0: -11, x1: 1.5,   // 图表 x 范围（世界坐标）
    y0: 0,   yMax: 7.2, // loss 轴高度
    z: 0                 // 曲线所在平面
  };

  var PANEL = {
    n: 8,               // 8×8 权重矩阵
    cell: 0.62,         // 单元格边长
    gap: 0.16,
    x0: 4.2, z0: -2.8,  // 左上角（世界坐标，面板平放在地面）
    maxH: 2.1, minH: 0.16
  };

  var STAGES = {
    pretrain: {
      title: 'PRETRAIN',
      metric: 'LOSS',
      steps: 240,
      color: 0x53c8ff,
      lr: 0.030,
      curve: function (t, rnd) { // t ∈ [0,1]
        return 9.4 * Math.exp(-2.3 * t) + 0.85 + 0.10 * Math.sin(23 * t) * (1 - t);
      }
    },
    sft: {
      title: 'SFT',
      metric: 'LOSS',
      steps: 120,
      color: 0xffb454,
      lr: 0.014,
      curve: function (t) {
        return 2.3 * Math.exp(-3.2 * t) + 0.34 + 0.05 * Math.sin(31 * t) * (1 - t);
      }
    },
    rl: {
      title: 'RL · POLICY GRADIENT',
      metric: 'REWARD',
      steps: 160,
      color: 0x7dffa8,
      lr: 0.020,
      curve: function (t) { // reward 上升，带噪声
        return 0.42 + 0.50 * (1 - Math.exp(-2.6 * t)) +
               0.035 * Math.sin(17 * t) + 0.02 * Math.sin(47 * t + 1.3);
      }
    }
  };

  var PARTICLE_POOL = 420;

  // ---------------------------------------------------------------- 工具

  // 确定性伪随机（曲线可复现）
  function mulberry(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeTextSprite(text, opts) {
    opts = opts || {};
    var px = opts.fontSize || 44;
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');
    ctx.font = '600 ' + px + 'px "Helvetica Neue", Helvetica, Arial, sans-serif';
    var w = Math.ceil(ctx.measureText(text).width) + 24;
    canvas.width = w; canvas.height = px + 28;
    ctx = canvas.getContext('2d');
    ctx.font = '600 ' + px + 'px "Helvetica Neue", Helvetica, Arial, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = opts.color || '#e8ecf4';
    ctx.fillText(text, 12, canvas.height / 2 + 2);
    var tex = new THREE.CanvasTexture(canvas);
    tex.anisotropy = 4;
    var mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
    var sprite = new THREE.Sprite(mat);
    var h = opts.height || 0.55;
    sprite.scale.set(h * (canvas.width / canvas.height), h, 1);
    return sprite;
  }

  // 权重值 → 颜色（负=蓝，0=暗灰，正=暖橙），loss 语义
  var _cNeg = new THREE.Color(0x3d7bff);
  var _cMid = new THREE.Color(0x35415c);
  var _cPos = new THREE.Color(0xff9540);
  function weightColor(v, out) {
    if (v >= 0) out.copy(_cMid).lerp(_cPos, Math.min(1, v));
    else out.copy(_cMid).lerp(_cNeg, Math.min(1, -v));
    return out;
  }

  // ---------------------------------------------------------------- 模块

  window.MODULE_TRAINING = {
    name: 'training',

    _scene: null, _camera: null, _renderer: null,
    _root: null,            // 模块根 Group
    _stage: 'pretrain',
    _visible: false,

    _curveGroup: null,      // 曲线 + 网格 + 标签
    _marker: null, _markerLight: null,
    _trail: null,           // 已走过的曲线段（高亮）
    _curvePts: [],          // 曲线采样点（Vector3）
    _progress: 0,           // 标记点进度 0..1
    _speed: 0.055,          // 每秒前进的曲线比例

    _weights: null,         // { cubes:[], values:[], targets:[] }
    _particles: null,       // Points + pool 数据
    _arrows: [],            // 参数更新向量
    _emitAcc: 0,
    _clock: 0,

    // ---------------------------------------------------------- init
    init: function (scene, camera, renderer) {
      this._scene = scene;
      this._camera = camera;
      this._renderer = renderer;
      this._root = new THREE.Group();
      this._root.name = 'module-training';
      this._root.visible = false;
      scene.add(this._root);

      this._buildGround();
      this._buildWeights();
      this._buildParticles();
      this._buildArrows();
    },

    // ---------------------------------------------------------- show
    show: function (data) {
      data = data || {};
      var stage = (data.stage && STAGES[data.stage]) ? data.stage : 'pretrain';
      this._stage = stage;

      if (this._curveGroup) {
        this._root.remove(this._curveGroup);
        this._disposeDeep(this._curveGroup);
      }
      this._buildCurve(stage, data.lossCurve || null);
      this._resetWeights();
      this._resetParticles();
      this._progress = 0;
      this._clock = 0;
      this._visible = true;
      this._root.visible = true;
    },

    hide: function () {
      this._visible = false;
      if (this._root) this._root.visible = false;
    },

    // ---------------------------------------------------------- update
    update: function (delta) {
      if (!this._visible || !this._root) return;
      delta = Math.max(0, Math.min(delta, 0.05));
      this._clock += delta;

      this._updateMarker(delta);
      this._updateParticles(delta);
      this._updateWeights(delta);
      this._updateArrows(delta);
    },

    // ---------------------------------------------------------- dispose
    dispose: function () {
      if (!this._root) return;
      this._visible = false;
      this._scene.remove(this._root);
      this._disposeDeep(this._root);
      this._root = null;
      this._curveGroup = null;
      this._weights = null;
      this._particles = null;
      this._arrows = [];
      this._scene = this._camera = this._renderer = null;
    },

    screenshot: function () {
      return this._renderer.domElement.toDataURL('image/png');
    },

    // ==========================================================
    // 内部构建
    // ==========================================================

    _buildGround: function () {
      // 只接收阴影的透明地面（不抢视觉焦点）
      var geo = new THREE.PlaneGeometry(80, 60);
      var mat = new THREE.ShadowMaterial({ opacity: 0.35 });
      var plane = new THREE.Mesh(geo, mat);
      plane.rotation.x = -Math.PI / 2;
      plane.position.y = -0.01;
      plane.receiveShadow = true;
      this._root.add(plane);
    },

    // ---------------- loss 曲线（3D 折线图） ----------------
    _buildCurve: function (stageKey, customCurve) {
      var st = STAGES[stageKey];
      var g = new THREE.Group();
      g.name = 'loss-curve';
      this._curveGroup = g;

      var rnd = mulberry(stageKey === 'pretrain' ? 7 : stageKey === 'sft' ? 13 : 29);
      var N = st.steps;
      var pts = [];
      var isRL = (stageKey === 'rl');
      // 指标量纲 → 世界 y
      var raw = [];
      var lo = Infinity, hi = -Infinity;
      for (var i = 0; i < N; i++) {
        var t = i / (N - 1);
        var v = customCurve ? customCurve[i] : st.curve(t, rnd);
        v += (rnd() - 0.5) * 0.02; // 轻微采样噪声
        raw.push(v);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      var span = Math.max(hi - lo, 1e-6);
      for (i = 0; i < N; i++) {
        t = i / (N - 1);
        var x = CHART.x0 + t * (CHART.x1 - CHART.x0);
        var y = CHART.y0 + ((raw[i] - lo) / span) * CHART.yMax;
        pts.push(new THREE.Vector3(x, y, CHART.z));
      }
      this._curvePts = pts;

      // --- 图表网格：底面 + 背面，细线
      var gridMat = new THREE.LineBasicMaterial({ color: 0x3a4356, transparent: true, opacity: 0.5 });
      var gridPts = [];
      var gx, gy;
      for (gx = 0; gx <= 8; gx++) { // 底面纵向线 + 背面纵向线
        var wx = CHART.x0 + (CHART.x1 - CHART.x0) * gx / 8;
        gridPts.push(new THREE.Vector3(wx, 0, -1.6), new THREE.Vector3(wx, 0, 1.6));
        gridPts.push(new THREE.Vector3(wx, 0, -1.6), new THREE.Vector3(wx, CHART.yMax + 0.6, -1.6));
      }
      for (gy = 0; gy <= 4; gy++) { // 背面水平线 + 底面横向线
        var wy = (CHART.yMax + 0.6) * gy / 4;
        gridPts.push(new THREE.Vector3(CHART.x0, wy, -1.6), new THREE.Vector3(CHART.x1, wy, -1.6));
        gridPts.push(new THREE.Vector3(CHART.x0, 0, -1.6 + 3.2 * gy / 4), new THREE.Vector3(CHART.x1, 0, -1.6 + 3.2 * gy / 4));
      }
      var gridGeo = new THREE.BufferGeometry().setFromPoints(gridPts);
      g.add(new THREE.LineSegments(gridGeo, gridMat));

      // --- 曲线本体：顶点颜色沿时间从红（高 loss）渐变到绿（低 loss / 高 reward）
      var curveGeo = new THREE.BufferGeometry().setFromPoints(pts);
      var colors = new Float32Array(N * 3);
      var cStart = new THREE.Color(isRL ? 0x4a5568 : 0xff5d5d);
      var cEnd = new THREE.Color(isRL ? 0x7dffa8 : 0x6dff9e);
      var tmp = new THREE.Color();
      for (i = 0; i < N; i++) {
        tmp.copy(cStart).lerp(cEnd, i / (N - 1));
        colors[i * 3] = tmp.r; colors[i * 3 + 1] = tmp.g; colors[i * 3 + 2] = tmp.b;
      }
      curveGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      var curveMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 });
      g.add(new THREE.Line(curveGeo, curveMat));

      // --- 已走过的 trail（高亮发光线，随标记点增长）
      var trailGeo = new THREE.BufferGeometry();
      trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
      trailGeo.setDrawRange(0, 0);
      var trailMat = new THREE.LineBasicMaterial({
        color: st.color, transparent: true, opacity: 0.95
      });
      this._trail = new THREE.Line(trailGeo, trailMat);
      this._trail.frustumCulled = false;
      g.add(this._trail);

      // --- 训练步标记：发光小球 + 点光
      var mkMat = new THREE.MeshStandardMaterial({
        color: st.color, emissive: st.color, emissiveIntensity: 1.6,
        roughness: 0.25, metalness: 0.1
      });
      this._marker = new THREE.Mesh(new THREE.SphereGeometry(0.20, 24, 24), mkMat);
      this._marker.castShadow = true;
      this._marker.position.copy(pts[0]);
      g.add(this._marker);
      this._markerLight = new THREE.PointLight(st.color, 12, 6, 2);
      this._marker.add(this._markerLight);

      // --- 轴标签（极简 sprite 文字）
      var title = makeTextSprite(st.title, { color: '#f2f5fa', height: 0.52 });
      title.position.set(CHART.x0 + 1.1, CHART.yMax + 1.15, 0);
      g.add(title);
      var yLab = makeTextSprite(st.metric, { color: '#9aa7bd', height: 0.36 });
      yLab.position.set(CHART.x0 - 1.5, CHART.yMax * 0.55, 0);
      g.add(yLab);
      var xLab = makeTextSprite('STEP', { color: '#9aa7bd', height: 0.36 });
      xLab.position.set(CHART.x1 + 0.9, 0.35, -1.6);
      g.add(xLab);

      this._root.add(g);
    },

    // ---------------- 权重矩阵（参数 + 更新动画） ----------------
    _buildWeights: function () {
      var n = PANEL.n;
      var pitch = PANEL.cell + PANEL.gap;
      var group = new THREE.Group();
      group.name = 'weight-matrix';
      var cubes = [], values = [], targets = [];
      var geo = new THREE.BoxGeometry(PANEL.cell, 1, PANEL.cell);
      // 平移几何体，使立方体以底面为原点，高度变化从地面长起
      geo.translate(0, 0.5, 0);

      for (var r = 0; r < n; r++) {
        for (var c = 0; c < n; c++) {
          var mat = new THREE.MeshStandardMaterial({
            color: 0x232a38, roughness: 0.45, metalness: 0.55
          });
          var cube = new THREE.Mesh(geo, mat);
          cube.castShadow = true;
          cube.receiveShadow = true;
          cube.position.set(PANEL.x0 + c * pitch, 0, PANEL.z0 + r * pitch);
          cubes.push(cube);
          values.push(0);
          targets.push(0);
          group.add(cube);
        }
      }
      // 面板底座
      var baseGeo = new THREE.BoxGeometry(n * pitch + 0.5, 0.12, n * pitch + 0.5);
      var base = new THREE.Mesh(baseGeo, new THREE.MeshStandardMaterial({
        color: 0x141a26, roughness: 0.3, metalness: 0.7
      }));
      base.position.set(PANEL.x0 + (n - 1) * pitch / 2, -0.06, PANEL.z0 + (n - 1) * pitch / 2);
      base.receiveShadow = true;
      group.add(base);

      var lab = makeTextSprite('WEIGHTS  W', { color: '#9aa7bd', height: 0.4 });
      lab.position.set(PANEL.x0 + (n - 1) * pitch / 2, 0.7, PANEL.z0 + (n - 1) * pitch + 1.6);
      group.add(lab);

      this._weights = { group: group, cubes: cubes, values: values, targets: targets };
      this._root.add(group);
    },

    _resetWeights: function () {
      if (!this._weights) return;
      var rnd = mulberry(99);
      var w = this._weights;
      for (var i = 0; i < w.values.length; i++) {
        w.values[i] = (rnd() - 0.5) * 1.0;
        w.targets[i] = w.values[i];
        this._applyWeightVisual(i, true);
      }
    },

    _applyWeightVisual: function (i, snap) {
      var w = this._weights;
      var v = w.values[i];
      var h = PANEL.minH + Math.abs(v) * PANEL.maxH;
      if (snap) w.cubes[i].scale.y = h;
      weightColor(v, w.cubes[i].material.color);
    },

    _updateWeights: function (delta) {
      var w = this._weights;
      if (!w) return;
      var k = 1 - Math.pow(0.002, delta); // 平滑逼近目标高度
      for (var i = 0; i < w.cubes.length; i++) {
        var target = PANEL.minH + Math.abs(w.values[i]) * PANEL.maxH;
        w.cubes[i].scale.y += (target - w.cubes[i].scale.y) * k;
        // 梯度到达的发光脉冲快速衰减
        var m = w.cubes[i].material;
        if (m.emissiveIntensity > 0.001) m.emissiveIntensity *= Math.pow(0.01, delta);
      }
    },

    // 梯度到达 → 更新一个权重：w ← w - η·g
    _applyGradient: function (idx, grad) {
      var st = STAGES[this._stage];
      var w = this._weights;
      var nv = w.values[idx] - st.lr * grad;
      w.values[idx] = Math.max(-1, Math.min(1, nv));
      weightColor(w.values[idx], w.cubes[idx].material.color);
      // 到达瞬间轻微发光脉冲
      w.cubes[idx].material.emissive.setHex(st.color);
      w.cubes[idx].material.emissiveIntensity = 0.55;
    },

    // ---------------- 梯度粒子流 ----------------
    _buildParticles: function () {
      var pool = PARTICLE_POOL;
      var geo = new THREE.BufferGeometry();
      var pos = new Float32Array(pool * 3);
      var col = new Float32Array(pool * 3);
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
      var mat = new THREE.PointsMaterial({
        size: 0.2, vertexColors: true, transparent: true, opacity: 0.95,
        blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
      });
      var points = new THREE.Points(geo, mat);
      points.frustumCulled = false;
      this._root.add(points);

      var items = [];
      for (var i = 0; i < pool; i++) {
        items.push({ active: false, t: 0, dur: 1, from: new THREE.Vector3(), ctrl: new THREE.Vector3(), to: new THREE.Vector3(), grad: 0, target: -1 });
        pos[i * 3 + 1] = -100; // 藏起来
      }
      this._particles = { points: points, items: items, pos: pos, col: col };
    },

    _resetParticles: function () {
      var p = this._particles;
      if (!p) return;
      for (var i = 0; i < p.items.length; i++) {
        p.items[i].active = false;
        p.pos[i * 3 + 1] = -100;
      }
      p.points.geometry.attributes.position.needsUpdate = true;
    },

    _emitParticle: function () {
      var p = this._particles, w = this._weights;
      for (var i = 0; i < p.items.length; i++) {
        var it = p.items[i];
        if (it.active) continue;
        var n = PANEL.n, pitch = PANEL.cell + PANEL.gap;
        var target = Math.floor(Math.random() * w.cubes.length);
        var r = Math.floor(target / n), c = target % n;
        it.active = true;
        it.t = 0;
        it.dur = 0.7 + Math.random() * 0.5;
        it.from.copy(this._marker.position);
        it.to.set(PANEL.x0 + c * pitch, 0.4 + Math.abs(w.values[target]) * PANEL.maxH, PANEL.z0 + r * pitch);
        // 弧线控制点：中点抬高，向后上方绕
        it.ctrl.copy(it.from).add(it.to).multiplyScalar(0.5);
        it.ctrl.y += 2.2 + Math.random() * 1.6;
        it.ctrl.z -= 1.5 + Math.random() * 1.5;
        it.grad = (Math.random() - 0.5) * 2; // 梯度方向有正有负
        it.target = target;

        // 梯度大小 → 颜色（大=红，小=蓝）
        var mag = Math.abs(it.grad);
        var cc = new THREE.Color().setHSL(0.62 - 0.62 * Math.min(1, mag), 0.9, 0.6);
        p.col[i * 3] = cc.r; p.col[i * 3 + 1] = cc.g; p.col[i * 3 + 2] = cc.b;
        p.points.geometry.attributes.color.needsUpdate = true;
        return;
      }
    },

    _updateMarker: function (delta) {
      var pts = this._curvePts;
      if (!pts.length) return;
      this._progress += this._speed * delta;
      if (this._progress >= 1) { // 一轮训练结束 → 重新开一轮
        this._progress = 0;
      }
      var f = this._progress * (pts.length - 1);
      var i0 = Math.floor(f), i1 = Math.min(i0 + 1, pts.length - 1);
      this._marker.position.lerpVectors(pts[i0], pts[i1], f - i0);

      // trail 增长
      var attr = this._trail.geometry.attributes.position;
      var count = i0 + 2;
      for (var i = 0; i < count; i++) {
        attr.array[i * 3] = pts[Math.min(i, pts.length - 1)].x;
        attr.array[i * 3 + 1] = pts[Math.min(i, pts.length - 1)].y;
        attr.array[i * 3 + 2] = pts[Math.min(i, pts.length - 1)].z;
      }
      this._trail.geometry.setDrawRange(0, count);
      attr.needsUpdate = true;

      // 呼吸发光
      var pulse = 1.4 + 0.5 * Math.sin(this._clock * 6);
      this._marker.material.emissiveIntensity = pulse;
      this._markerLight.intensity = 8 + 4 * Math.sin(this._clock * 6);

      // 发射梯度粒子（反向传播）：速率随阶段 lr
      this._emitAcc += delta * (14 + STAGES[this._stage].lr * 500);
      while (this._emitAcc >= 1) {
        this._emitAcc -= 1;
        this._emitParticle();
      }
    },

    _updateParticles: function (delta) {
      var p = this._particles;
      var a = new THREE.Vector3(), b = new THREE.Vector3();
      var dirty = false;
      for (var i = 0; i < p.items.length; i++) {
        var it = p.items[i];
        if (!it.active) continue;
        it.t += delta / it.dur;
        if (it.t >= 1) {
          it.active = false;
          p.pos[i * 3 + 1] = -100;
          this._applyGradient(it.target, it.grad);
          dirty = true;
          continue;
        }
        // 二次贝塞尔
        var t = it.t, u = 1 - t;
        a.copy(it.from).multiplyScalar(u * u);
        b.copy(it.ctrl).multiplyScalar(2 * u * t);
        a.add(b);
        b.copy(it.to).multiplyScalar(t * t);
        a.add(b);
        p.pos[i * 3] = a.x; p.pos[i * 3 + 1] = a.y; p.pos[i * 3 + 2] = a.z;
        dirty = true;
      }
      if (dirty) p.points.geometry.attributes.position.needsUpdate = true;
    },

    // ---------------- 参数更新向量箭头 ----------------
    _buildArrows: function () {
      var n = PANEL.n, pitch = PANEL.cell + PANEL.gap;
      var cx = PANEL.x0 + (n - 1) * pitch / 2;
      var cz = PANEL.z0 + (n - 1) * pitch / 2;
      // 5 个向量箭头悬在权重矩阵上方：-η∇w 方向（向下压入矩阵）
      for (var i = 0; i < 5; i++) {
        var dir = new THREE.Vector3((Math.random() - 0.5) * 0.5, -1, (Math.random() - 0.5) * 0.5).normalize();
        var origin = new THREE.Vector3(
          cx + (Math.random() - 0.5) * n * pitch * 0.7,
          2.6 + Math.random() * 0.7,
          cz + (Math.random() - 0.5) * n * pitch * 0.7
        );
        var arrow = new THREE.ArrowHelper(dir, origin, 1.1, 0xffd166, 0.3, 0.18);
        arrow.line.material.transparent = true;
        arrow.cone.material.transparent = true;
        arrow.userData.phase = Math.random() * Math.PI * 2;
        arrow.userData.baseY = origin.y;
        this._arrows.push(arrow);
        this._root.add(arrow);
      }
      var lab = makeTextSprite('Δw = −η∇w', { color: '#ffd166', height: 0.38 });
      lab.position.set(cx, 4.2, cz);
      this._root.add(lab);
    },

    _updateArrows: function (delta) {
      var st = STAGES[this._stage];
      for (var i = 0; i < this._arrows.length; i++) {
        var ar = this._arrows[i];
        var ph = this._clock * 2.2 + ar.userData.phase;
        var s = 0.5 + 0.5 * Math.sin(ph); // 脉冲
        ar.position.y = ar.userData.baseY - s * 0.5; // 向下压 = 参数被更新
        var op = 0.25 + 0.55 * s;
        ar.line.material.opacity = op;
        ar.cone.material.opacity = op;
        ar.setLength(0.7 + s * 0.6 * (st.lr / 0.03), 0.28, 0.16);
      }
    },

    // ---------------- 资源清理 ----------------
    _disposeDeep: function (obj) {
      obj.traverse(function (o) {
        if (o.geometry) o.geometry.dispose();
        if (o.material) {
          var mats = Array.isArray(o.material) ? o.material : [o.material];
          mats.forEach(function (m) {
            if (m.map) m.map.dispose();
            m.dispose();
          });
        }
      });
    }
  };
})();
