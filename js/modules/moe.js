/**
 * MoE (Mixture-of-Experts) 3D 可视化模块
 *
 * 真实结构布局（x 轴 = 数据流方向）：
 *   输入向量小球 → Router 核心 → 路由分数柱状图 → 专家球阵列（选中高亮）
 *   → 选中专家发出带权重的输出向量光束 → Σ 聚合球 → 聚合输出向量
 *
 * 接口（MODULE_SPEC.md v1）：
 *   init(scene, camera, renderer) / show(data) / hide() / update(delta)
 *   dispose() / screenshot()
 *
 * data 格式：
 *   {
 *     routerScores:    [0.1, 0.2, ...],   // 每个专家的路由分数
 *     selectedExperts: [3, 7, ...],       // top-k 选中的专家索引
 *     expertOutputs:   [[...], ...],      // 每个选中专家的输出向量
 *     aggregated:      [...]              // 加权聚合后的向量
 *   }
 *   不传 data 时自动生成一组演示数据（8 专家、top-2 路由）。
 */
(function () {
  'use strict';

  // ── 可调常量 ─────────────────────────────────────────────
  var NUM_EXPERTS   = 8;      // 演示用专家数（data.routerScores 可覆盖）
  var EXPERT_RADIUS = 0.85;   // 专家球半径
  var BAR_MAX_H     = 3.6;    // 分数柱最大高度
  var BEAM_SEGS     = 48;     // 光束管分段

  var COL = {
    input:      0x35d6ff,   // 输入 token — 青
    router:     0xb46bff,   // 路由器 — 紫
    barOff:     0x2b4a6b,   // 未选中分数柱 — 冷蓝
    barOn:      0xff8c42,   // 选中分数柱 — 暖橙
    expertOff:  0x39536e,   // 未选中专家 — 灰蓝
    expertOn:   0xff9d4d,   // 选中专家 — 亮橙
    beam:       0xffb36b,   // 输出光束 — 金
    agg:        0xffd166,   // 聚合球 — 金
    output:     0x7bff9e,   // 输出向量 — 绿
    ground:     0x0c1220
  };

  // ── 模块内部状态 ─────────────────────────────────────────
  var scene, camera, renderer;
  var root = null;            // 模块根 Group
  var clock = { t: 0 };
  var visible = false;

  var bars = [];              // { mesh, targetH, idx }
  var experts = [];           // { mesh, halo, baseY, idx, selected, weight }
  var beams = [];             // { mesh, from, to, weight }
  var particles = [];         // { mesh, curve, t, speed }
  var disposables = [];       // 需要 dispose 的资源
  var routerCore, aggCore, outArrow, inArrow, aggRings = [];

  function track(obj) { disposables.push(obj); return obj; }

  // ── 工具：文本标签 Sprite ───────────────────────────────
  function makeLabel(text, color, scale) {
    scale = scale || 1;
    var pad = 24, fs = 56;
    var cv = document.createElement('canvas');
    var cx = cv.getContext('2d');
    cx.font = '600 ' + fs + 'px -apple-system, "PingFang SC", sans-serif';
    cv.width  = Math.ceil(cx.measureText(text).width) + pad * 2;
    cv.height = fs + pad * 2;
    cx = cv.getContext('2d');
    cx.font = '600 ' + fs + 'px -apple-system, "PingFang SC", sans-serif';
    cx.textBaseline = 'middle';
    cx.shadowColor = 'rgba(0,0,0,0.8)';
    cx.shadowBlur = 10;
    cx.fillStyle = color || '#e8eef7';
    cx.fillText(text, pad, cv.height / 2);
    var tex = track(new THREE.CanvasTexture(cv));
    tex.anisotropy = 4;
    var mat = track(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    var sp = new THREE.Sprite(mat);
    var w = 2.4 * scale, h = w * cv.height / cv.width;
    sp.scale.set(w, h, 1);
    return sp;
  }

  // ── 工具：向量箭头 ──────────────────────────────────────
  function makeArrow(dir, origin, length, color, headScale) {
    var g = new THREE.Group();
    var d = dir.clone().normalize();
    var shaftGeo = track(new THREE.CylinderGeometry(0.07, 0.07, length, 16));
    var headGeo  = track(new THREE.ConeGeometry(0.2 * (headScale || 1), 0.55 * (headScale || 1), 20));
    var mat = track(new THREE.MeshStandardMaterial({
      color: color, emissive: color, emissiveIntensity: 0.45,
      roughness: 0.3, metalness: 0.2
    }));
    var shaft = new THREE.Mesh(shaftGeo, mat);
    shaft.position.y = length / 2;
    shaft.castShadow = true;
    var head = new THREE.Mesh(headGeo, mat);
    head.position.y = length + 0.27 * (headScale || 1);
    head.castShadow = true;
    g.add(shaft); g.add(head);
    g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d);
    g.position.copy(origin);
    return g;
  }

  // ── 工具：发光小球 ──────────────────────────────────────
  function glowSphere(radius, color, emissiveI, rough, metal) {
    var geo = track(new THREE.SphereGeometry(radius, 48, 32));
    var mat = track(new THREE.MeshStandardMaterial({
      color: color,
      emissive: color,
      emissiveIntensity: emissiveI != null ? emissiveI : 0.25,
      roughness: rough != null ? rough : 0.25,
      metalness: metal != null ? metal : 0.55
    }));
    var m = new THREE.Mesh(geo, mat);
    m.castShadow = true;
    m.receiveShadow = true;
    return m;
  }

  // ── 工具：光束管（curve 沿 from→to 微弯）────────────────
  function beamCurve(from, to) {
    var mid = from.clone().lerp(to, 0.5);
    mid.y += 0.9; // 向上微拱，让数据流有"跨过"的感觉
    return new THREE.QuadraticBezierCurve3(from, mid, to);
  }

  function makeBeam(from, to, weight) {
    var curve = beamCurve(from, to);
    var radius = 0.05 + weight * 0.16; // 权重越大光束越粗
    var geo = track(new THREE.TubeGeometry(curve, BEAM_SEGS, radius, 12, false));
    var mat = track(new THREE.MeshStandardMaterial({
      color: COL.beam, emissive: COL.beam,
      emissiveIntensity: 0.7, roughness: 0.35, metalness: 0.1,
      transparent: true, opacity: 0.55 + weight * 0.4
    }));
    var m = new THREE.Mesh(geo, mat);
    return { mesh: m, curve: curve, weight: weight };
  }

  // ── 演示数据 ────────────────────────────────────────────
  function demoData(n) {
    var scores = [0.05, 0.12, 0.28, 0.04, 0.31, 0.08, 0.02, 0.10];
    while (scores.length < n) scores.push(Math.random() * 0.06);
    scores.length = n;
    // top-2
    var idx = scores.map(function (s, i) { return [s, i]; })
      .sort(function (a, b) { return b[0] - a[0]; })
      .slice(0, 2).map(function (p) { return p[1]; });
    var selW = idx.map(function (i) { return scores[i]; });
    var wSum = selW[0] + selW[1];
    var outs = idx.map(function (_, k) {
      var v = [];
      for (var d = 0; d < 4; d++) v.push(Math.sin(d * 1.7 + k) * 0.8);
      return v;
    });
    var agg = [0, 0, 0, 0];
    outs.forEach(function (v, k) {
      for (var d = 0; d < 4; d++) agg[d] += v[d] * (selW[k] / wSum);
    });
    return {
      routerScores: scores,
      selectedExperts: idx,
      expertOutputs: outs,
      aggregated: agg,
      weights: selW.map(function (w) { return w / wSum; })
    };
  }

  // ── 场景构建 ────────────────────────────────────────────
  function build(data) {
    root = new THREE.Group();
    scene.add(root);

    var n = data.routerScores.length;
    var zStep = 1.9;
    var zSpan = (n - 1) * zStep;

    // 权重归一化（选中专家的 softmax 权重）
    var weights = data.weights;
    if (!weights) {
      var s = 0;
      data.selectedExperts.forEach(function (i) { s += data.routerScores[i]; });
      weights = data.selectedExperts.map(function (i) { return data.routerScores[i] / s; });
    }
    var wOf = {};
    data.selectedExperts.forEach(function (i, k) { wOf[i] = weights[k]; });

    // ── 地面：阴影承接 + 细网格 ──
    var groundGeo = track(new THREE.PlaneGeometry(46, 30));
    var groundMat = track(new THREE.MeshStandardMaterial({
      color: COL.ground, roughness: 0.9, metalness: 0.1
    }));
    var ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -2.2;
    ground.receiveShadow = true;
    root.add(ground);
    var grid = new THREE.GridHelper(44, 44, 0x1c2b44, 0x14203a);
    grid.position.y = -2.19;
    grid.material.transparent = true;
    grid.material.opacity = 0.5;
    track(grid.material);
    root.add(grid);

    // ── 1) 输入 token 向量 ──
    var inPos = new THREE.Vector3(-11, 0, 0);
    var inBall = glowSphere(0.55, COL.input, 0.55, 0.2, 0.6);
    inBall.position.copy(inPos);
    root.add(inBall);
    inArrow = makeArrow(new THREE.Vector3(1, 0, 0), new THREE.Vector3(-10.2, 0, 0), 1.6, COL.input);
    root.add(inArrow);
    var inLabel = makeLabel('输入向量 x', '#7ee7ff', 0.9);
    inLabel.position.set(-11, 1.4, 0);
    root.add(inLabel);

    // ── 2) Router 核心 ──
    var routerPos = new THREE.Vector3(-7, 0, 0);
    var routerGeo = track(new THREE.IcosahedronGeometry(0.75, 1));
    var routerMat = track(new THREE.MeshStandardMaterial({
      color: COL.router, emissive: COL.router, emissiveIntensity: 0.4,
      roughness: 0.2, metalness: 0.7, flatShading: true
    }));
    routerCore = new THREE.Mesh(routerGeo, routerMat);
    routerCore.position.copy(routerPos);
    routerCore.castShadow = true;
    root.add(routerCore);
    var routerLabel = makeLabel('Router (g = W·x)', '#d9b8ff', 0.9);
    routerLabel.position.set(-7, 1.5, 0);
    root.add(routerLabel);
    // router → 分数柱 的细连线
    var toBars = makeArrow(new THREE.Vector3(1, 0, 0), new THREE.Vector3(-6.1, 0, 0), 1.0, COL.router, 0.8);
    root.add(toBars);

    // ── 3) 路由分数柱状图 ──
    var barsX = -4.2;
    var maxScore = Math.max.apply(null, data.routerScores) || 1;
    for (var b = 0; b < n; b++) {
      var sel = data.selectedExperts.indexOf(b) >= 0;
      var h = Math.max(0.06, (data.routerScores[b] / maxScore) * BAR_MAX_H);
      var geo = track(new THREE.BoxGeometry(0.55, 1, 0.55));
      geo.translate(0, 0.5, 0); // 底部锚定
      var c = sel ? COL.barOn : COL.barOff;
      var mat = track(new THREE.MeshStandardMaterial({
        color: c, emissive: c, emissiveIntensity: sel ? 0.55 : 0.15,
        roughness: 0.3, metalness: 0.4
      }));
      var mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(barsX, -2.2, -zSpan / 2 + b * zStep);
      mesh.scale.y = 0.001;
      mesh.castShadow = true;
      root.add(mesh);
      bars.push({ mesh: mesh, targetH: h, idx: b, selected: sel });
    }
    var barsLabel = makeLabel('路由分数', '#9fb6d8', 0.85);
    barsLabel.position.set(barsX, 2.2, 0);
    root.add(barsLabel);

    // ── 4) 专家阵列 ──
    var exX = 2.2;
    for (var e = 0; e < n; e++) {
      var isSel = data.selectedExperts.indexOf(e) >= 0;
      var z = -zSpan / 2 + e * zStep;
      var col = isSel ? COL.expertOn : COL.expertOff;
      var ball = glowSphere(EXPERT_RADIUS, col, isSel ? 0.6 : 0.1, 0.25, 0.5);
      ball.position.set(exX, 0, z);
      root.add(ball);
      var halo = null;
      if (isSel) {
        var haloGeo = track(new THREE.SphereGeometry(EXPERT_RADIUS * 1.35, 32, 24));
        var haloMat = track(new THREE.MeshBasicMaterial({
          color: COL.expertOn, transparent: true, opacity: 0.14,
          side: THREE.BackSide, depthWrite: false
        }));
        halo = new THREE.Mesh(haloGeo, haloMat);
        halo.position.copy(ball.position);
        root.add(halo);
      }
      var lb = makeLabel('E' + e + (isSel ? '  ✓ ' + Math.round((wOf[e] || 0) * 100) + '%' : ''),
        isSel ? '#ffc08a' : '#6f87a8', 0.62);
      lb.position.set(exX, -1.5, z);
      root.add(lb);
      experts.push({ mesh: ball, halo: halo, baseY: 0, idx: e, selected: isSel, weight: wOf[e] || 0 });
    }

    // ── 5) 选中专家的输出向量 + 光束 → 聚合 ──
    var aggPos = new THREE.Vector3(8.4, 0, 0);
    data.selectedExperts.forEach(function (ei, k) {
      var w = wOf[ei];
      var from = new THREE.Vector3(exX, 0, -zSpan / 2 + ei * zStep);
      // 专家输出向量（从专家球指向聚合方向的短箭头）
      var dir = aggPos.clone().sub(from).normalize();
      var arr = makeArrow(dir, from.clone().add(dir.clone().multiplyScalar(EXPERT_RADIUS + 0.1)),
        1.1 + w * 0.9, COL.beam, 0.9);
      root.add(arr);
      // 光束
      var beam = makeBeam(from.clone().add(dir.clone().multiplyScalar(0.4)),
        aggPos.clone().sub(dir.clone().multiplyScalar(0.9)), w);
      root.add(beam.mesh);
      beams.push(beam);
      // 沿线流动的粒子（加权求和动画的数据流）
      for (var p = 0; p < 3; p++) {
        var pg = track(new THREE.SphereGeometry(0.09 + w * 0.06, 12, 10));
        var pm = track(new THREE.MeshBasicMaterial({ color: 0xffe0a8 }));
        var pmesh = new THREE.Mesh(pg, pm);
        root.add(pmesh);
        particles.push({ mesh: pmesh, curve: beam.curve, t: p / 3, speed: 0.35 + w * 0.25 });
      }
    });

    // ── 6) Σ 聚合球 ──
    aggCore = glowSphere(0.9, COL.agg, 0.7, 0.18, 0.65);
    aggCore.position.copy(aggPos);
    root.add(aggCore);
    // 两层旋转环，表达"求和汇聚"
    for (var r = 0; r < 2; r++) {
      var ringGeo = track(new THREE.TorusGeometry(1.25 + r * 0.28, 0.03, 10, 64));
      var ringMat = track(new THREE.MeshStandardMaterial({
        color: COL.agg, emissive: COL.agg, emissiveIntensity: 0.5,
        roughness: 0.3, metalness: 0.6, transparent: true, opacity: 0.7
      }));
      var ring = new THREE.Mesh(ringGeo, ringMat);
      ring.position.copy(aggPos);
      ring.rotation.x = Math.PI / 2 + r * 0.5;
      root.add(ring);
      aggRings.push(ring);
    }
    var aggLabel = makeLabel('y = Σ wᵢ · Eᵢ(x)', '#ffe3a1', 0.95);
    aggLabel.position.set(aggPos.x, 2.1, 0);
    root.add(aggLabel);

    // ── 7) 聚合输出向量 ──
    var outLen = 1.8;
    if (data.aggregated && data.aggregated.length) {
      var mag = Math.sqrt(data.aggregated.reduce(function (a, v) { return a + v * v; }, 0));
      outLen = 1.2 + Math.min(2.0, mag);
    }
    outArrow = makeArrow(new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(aggPos.x + 1.0, 0, 0), outLen, COL.output);
    root.add(outArrow);
    var outLabel = makeLabel('聚合输出 y', '#a9ffc4', 0.9);
    outLabel.position.set(aggPos.x + 2.4, 1.4, 0);
    root.add(outLabel);
  }

  // ── 相机取景 ────────────────────────────────────────────
  function frameCamera() {
    camera.position.set(10.5, 11, 20.5);
    camera.lookAt(-1.2, -0.2, 0);
  }

  // ── 模块接口 ────────────────────────────────────────────
  window.MOE = {
    init: function (sc, cam, ren) {
      scene = sc; camera = cam; renderer = ren;
    },

    show: function (data) {
      this.hide();
      data = data && data.routerScores ? data : demoData(NUM_EXPERTS);
      build(data);
      frameCamera();
      clock.t = 0;
      visible = true;
    },

    hide: function () {
      if (root) {
        scene.remove(root);
        root.traverse(function (o) {
          if (o.geometry) o.geometry.dispose();
          if (o.material) {
            (Array.isArray(o.material) ? o.material : [o.material]).forEach(function (m) {
              if (m.map) m.map.dispose();
              m.dispose();
            });
          }
        });
      }
      root = null;
      bars = []; experts = []; beams = []; particles = []; aggRings = [];
      disposables = [];
      routerCore = aggCore = outArrow = inArrow = null;
      visible = false;
    },

    update: function (delta) {
      if (!visible || !root) return;
      clock.t += delta;
      var t = clock.t;

      // 分数柱弹起动画（show 后 ease-out 到位）
      bars.forEach(function (b) {
        var cur = b.mesh.scale.y;
        var target = b.targetH;
        b.mesh.scale.y = cur + (target - cur) * Math.min(1, delta * 4);
      });

      // Router 缓慢自转
      if (routerCore) {
        routerCore.rotation.y += delta * 0.6;
        routerCore.rotation.x += delta * 0.2;
      }

      // 专家浮动 + 选中者呼吸发光
      experts.forEach(function (e) {
        e.mesh.position.y = e.baseY + Math.sin(t * 1.4 + e.idx * 0.9) * 0.12;
        if (e.selected) {
          var pulse = 0.55 + Math.sin(t * 3 + e.idx) * 0.25;
          e.mesh.material.emissiveIntensity = pulse;
          if (e.halo) {
            e.halo.position.y = e.mesh.position.y;
            e.halo.scale.setScalar(1 + Math.sin(t * 3 + e.idx) * 0.06);
          }
        }
      });

      // 数据流粒子沿光束流动（加权求和动画）
      particles.forEach(function (p) {
        p.t = (p.t + delta * p.speed) % 1;
        p.curve.getPoint(p.t, p.mesh.position);
        p.mesh.material.opacity = 1;
      });

      // 聚合球脉动 + 环旋转（加权汇聚感）
      if (aggCore) {
        var s = 1 + Math.sin(t * 2.2) * 0.05;
        aggCore.scale.setScalar(s);
        aggCore.material.emissiveIntensity = 0.6 + Math.sin(t * 2.2) * 0.2;
      }
      aggRings.forEach(function (ring, i) {
        ring.rotation.z += delta * (0.5 + i * 0.3) * (i % 2 ? -1 : 1);
      });

      // 输出箭头呼吸
      if (outArrow) {
        outArrow.position.y = Math.sin(t * 2.2) * 0.05;
      }
    },

    dispose: function () {
      this.hide();
      scene = camera = renderer = null;
    },

    screenshot: function () {
      return renderer.domElement.toDataURL('image/png');
    }
  };
})();
