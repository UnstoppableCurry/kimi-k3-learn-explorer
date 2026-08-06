/**
 * attention.js — Attention 模块（MODULE_SPEC v1）
 *
 * 真实结构：
 *   Q 向量（当前 token 的查询，橙色球柱）
 *   K 向量 × N token（青色球柱，每柱对应一个 token）
 *   V 向量 × N token（绿色球柱）
 *   attention 权重矩阵 = softmax(QKᵀ/√d)，8×8 热力图（蓝=低，红=高，高权重向前凸起）
 *   输出向量 = Σ wᵢ·Vᵢ（紫色球柱，入场时逐元素生长 = 向量变形动画）
 *
 * 数据流（粒子）：
 *   Q → 每个 K（算分，弧线透明度 ∝ 权重）
 *   每个 V → 输出（加权求和，粒子亮度/大小 ∝ 权重）
 *   权重矩阵当前行 → 输出
 *
 * 接口：init / show / hide / update / dispose / screenshot
 */
window.ATTENTION_MODULE = (function () {
  'use strict';

  // ---- 常量 ----
  var DIM = 12;        // 展示的向量维度
  var NTOK = 8;        // token 数（矩阵 8×8）
  var SPHERE_R = 0.15; // 向量元素球半径
  var SPHERE_GAP = 0.36;
  var COL_X0 = -3.5, COL_DX = 1.0;  // K/V 柱排布
  var QX = -7, OUTX = 7;            // Q / 输出 的 x 位置
  var KZ = -3.2, VZ = 3.2;          // K 排 / V 排 的 z（拉开避免互相遮挡）
  var MAT_POS = new THREE.Vector3(0, 2.8, -7.5); // 权重矩阵中心（靠后上方，避免遮挡 K 排）
  var BASE_Y = -1.8;   // 向量柱底部 y

  // ---- 状态 ----
  var scene, camera, renderer;
  var group = null;          // 模块根节点（含灯光，自给自足）
  var content = null;        // 数据相关内容（show 时重建）
  var particles = [];        // {mesh, curve, phase, speed, size}
  var bobbers = [];          // {obj, baseY, phase, speed} 轻微浮动
  var outSpheres = [];       // 输出向量球（变形动画）
  var state = { shown: false, appear: 0, time: 0 };
  var sharedGeo = {};        // 共享几何体

  // 确定性随机（同一输入永远同一画面）
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---- 数据：token → q/k/v → softmax 权重 → 输出（真实计算）----
  function computeAttention(data) {
    data = data || {};
    var tokens = data.tokens;
    if (!tokens || !tokens.length) {
      var texts = ['你', '好', '，', '世', '界', '！', '今', '天'];
      tokens = texts.map(function (t, i) { return { id: i, text: t, position: i }; });
    }
    tokens = tokens.slice(0, NTOK);
    while (tokens.length < NTOK) {
      tokens.push({ id: tokens.length, text: '·', position: tokens.length });
    }
    var M = tokens.length;
    var rnd = mulberry32(20260806 + M * 97 + (data.seed || 0));

    // embedding（若未给，用确定性随机）
    var emb = tokens.map(function (tk, i) {
      if (tk.embedding && tk.embedding.length) return tk.embedding;
      var e = [];
      for (var d = 0; d < DIM; d++) e.push(rnd() * 2 - 1);
      return e;
    });

    // 三个投影矩阵 Wq/Wk/Wv（DIM×DIM，确定性）
    function projMatrix() {
      var W = [];
      for (var i = 0; i < DIM; i++) {
        var row = [];
        for (var j = 0; j < DIM; j++) row.push((rnd() * 2 - 1) / Math.sqrt(DIM));
        W.push(row);
      }
      return W;
    }
    var Wq = projMatrix(), Wk = projMatrix(), Wv = projMatrix();

    function proj(W, x) {
      var y = [];
      for (var i = 0; i < DIM; i++) {
        var s = 0;
        for (var j = 0; j < DIM; j++) s += W[i][j] * (x[j] || 0);
        y.push(s);
      }
      return y;
    }

    var Q = emb.map(function (e) { return proj(Wq, e); });
    var K = emb.map(function (e) { return proj(Wk, e); });
    var V = emb.map(function (e) { return proj(Wv, e); });

    // 权重 = softmax(Q·Kᵀ/√d)，逐行
    // 加位置亲和项（邻近 token 互相注意），让矩阵呈真实自注意力的对角带状分布
    var scale = 1 / Math.sqrt(DIM);
    var weights = [];
    for (var i = 0; i < M; i++) {
      var row = [], max = -Infinity;
      for (var j = 0; j < M; j++) {
        var s = 0;
        for (var d = 0; d < DIM; d++) s += Q[i][d] * K[j][d];
        s *= scale;
        s += 1.6 * Math.exp(-Math.abs(i - j) / 1.8); // 位置亲和 → 对角亮带
        row.push(s);
        if (s > max) max = s;
      }
      var sum = 0;
      for (j = 0; j < M; j++) { row[j] = Math.exp(row[j] - max); sum += row[j]; }
      for (j = 0; j < M; j++) row[j] /= sum;
      weights.push(row);
    }

    // 输出 = 权重 · V（每个 token 一个输出向量）
    var outputs = [];
    for (i = 0; i < M; i++) {
      var o = [];
      for (d = 0; d < DIM; d++) {
        s = 0;
        for (j = 0; j < M; j++) s += weights[i][j] * V[j][d];
        o.push(s);
      }
      outputs.push(o);
    }

    return {
      tokens: tokens, M: M,
      q: Q, k: K, v: V,
      weights: weights,
      output: outputs[M - 1], // 当前 token（最后一个）的输出
      current: M - 1
    };
  }

  // ---- 视觉小工具 ----
  function makeTextSprite(text, opts) {
    opts = opts || {};
    var fs = opts.fontSize || 56;
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');
    ctx.font = '600 ' + fs + 'px -apple-system, "PingFang SC", "Helvetica Neue", sans-serif';
    var w = Math.ceil(ctx.measureText(text).width) + 24;
    canvas.width = w; canvas.height = fs + 28;
    ctx = canvas.getContext('2d');
    ctx.font = '600 ' + fs + 'px -apple-system, "PingFang SC", "Helvetica Neue", sans-serif';
    ctx.fillStyle = opts.color || '#e8ecf4';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(text, w / 2, canvas.height / 2);
    var tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    var mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
    var sprite = new THREE.Sprite(mat);
    var h = opts.height || 0.42;
    sprite.scale.set(h * canvas.width / canvas.height, h, 1);
    return sprite;
  }

  // 值 → 颜色：|v| 越大越亮（在向量自身色相内）
  function valueColor(v, hue) {
    var l = 0.28 + Math.min(1, Math.abs(v) * 1.4) * 0.42;
    return new THREE.Color().setHSL(hue, 0.75, l);
  }

  // 权重 → 热力颜色：深蓝(低) → 紫 → 亮红(高)，不经过绿色
  function heatColor(w) {
    var c = new THREE.Color();
    c.setHSL((0.62 + 0.38 * Math.pow(w, 0.9)) % 1, 0.85, 0.24 + 0.38 * Math.pow(w, 0.8));
    return c;
  }

  // 一根向量柱：DIM 个小球 + 金属底杆 + 顶部箭头
  function makeVectorColumn(vec, hue, opts) {
    opts = opts || {};
    var g = new THREE.Group();
    var totalH = DIM * SPHERE_GAP;
    // 底杆
    var rod = new THREE.Mesh(sharedGeo.rod, new THREE.MeshStandardMaterial({
      color: new THREE.Color().setHSL(hue, 0.3, 0.25),
      metalness: 0.85, roughness: 0.35
    }));
    rod.scale.y = totalH;
    rod.position.y = totalH / 2;
    rod.castShadow = true;
    g.add(rod);
    // 元素球
    for (var i = 0; i < DIM; i++) {
      var v = vec[i] || 0;
      var r = SPHERE_R * (0.75 + Math.min(1, Math.abs(v) * 1.6) * 0.5);
      var s = new THREE.Mesh(sharedGeo.sphere, new THREE.MeshStandardMaterial({
        color: valueColor(v, hue),
        metalness: 0.55, roughness: 0.25,
        emissive: new THREE.Color().setHSL(hue, 0.8, 0.08)
      }));
      s.scale.setScalar(r / SPHERE_R);
      s.position.y = i * SPHERE_GAP + SPHERE_GAP / 2;
      s.castShadow = true;
      g.add(s);
      if (opts.collect) opts.collect.push({ mesh: s, value: v, index: i });
    }
    // 顶部箭头
    var cone = new THREE.Mesh(sharedGeo.cone, new THREE.MeshStandardMaterial({
      color: new THREE.Color().setHSL(hue, 0.8, 0.55),
      metalness: 0.6, roughness: 0.3
    }));
    cone.position.y = totalH + 0.22;
    cone.castShadow = true;
    g.add(cone);
    return g;
  }

  function makeParticle(color, size) {
    var m = new THREE.Mesh(sharedGeo.particle, new THREE.MeshBasicMaterial({
      color: color, transparent: true, opacity: 0.95,
      blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false
    }));
    m.scale.setScalar(size);
    return m;
  }

  function bezier(a, b, lift) {
    var mid = a.clone().add(b).multiplyScalar(0.5);
    mid.y += lift;
    return new THREE.QuadraticBezierCurve3(a, mid, b);
  }

  // ---- 场景搭建 ----
  function build(data) {
    disposeContent();
    content = new THREE.Group();
    group.add(content);
    particles = []; bobbers = []; outSpheres = [];

    var cur = data.current;
    var curW = data.weights[cur]; // 当前 token 对各 key 的注意力

    // ===== Q 向量（当前 token）=====
    var qCol = makeVectorColumn(data.q[cur], 0.07); // 橙
    qCol.position.set(QX, BASE_Y, 0);
    content.add(qCol);
    bobbers.push({ obj: qCol, baseY: BASE_Y, phase: 0, speed: 0.8 });
    var qLabel = makeTextSprite('Q 查询（当前 token）', { color: '#ffb35c', height: 0.5 });
    qLabel.position.set(QX, BASE_Y + DIM * SPHERE_GAP + 0.85, 0);
    content.add(qLabel);

    // ===== K / V 向量排（每 token 一柱）=====
    for (var j = 0; j < data.M; j++) {
      var x = COL_X0 + j * COL_DX;
      var kCol = makeVectorColumn(data.k[j], 0.52); // 青
      kCol.position.set(x, BASE_Y, KZ);
      content.add(kCol);
      bobbers.push({ obj: kCol, baseY: BASE_Y, phase: j * 0.6, speed: 0.7 });
      var vCol = makeVectorColumn(data.v[j], 0.38); // 绿
      vCol.position.set(x, BASE_Y, VZ);
      content.add(vCol);
      bobbers.push({ obj: vCol, baseY: BASE_Y, phase: j * 0.6 + 2, speed: 0.7 });
      // token 标签（在 K 柱上方）
      var tk = makeTextSprite(data.tokens[j].text, { color: '#cfe3ff', height: 0.4 });
      tk.position.set(x, BASE_Y + DIM * SPHERE_GAP + 0.72, KZ);
      content.add(tk);
    }
    var kLabel = makeTextSprite('K 键', { color: '#6fd8e8', height: 0.44 });
    kLabel.position.set(COL_X0 + data.M * COL_DX + 0.3, BASE_Y + 2.2, KZ);
    content.add(kLabel);
    var vLabel = makeTextSprite('V 值', { color: '#8ce89a', height: 0.44 });
    vLabel.position.set(COL_X0 + data.M * COL_DX + 0.3, BASE_Y + 2.2, VZ);
    content.add(vLabel);

    // ===== Q→K 算分弧线（透明度 ∝ 注意力权重）+ 粒子 =====
    var qTop = new THREE.Vector3(QX, BASE_Y + DIM * SPHERE_GAP + 0.1, 0);
    for (j = 0; j < data.M; j++) {
      var kTop = new THREE.Vector3(COL_X0 + j * COL_DX, BASE_Y + DIM * SPHERE_GAP + 0.1, KZ);
      var curve = bezier(qTop, kTop, 1.6 + j * 0.12);
      var lineGeo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(40));
      var line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({
        color: 0xffa040, transparent: true, opacity: 0.15 + curW[j] * 0.7
      }));
      content.add(line);
      var p = makeParticle(0xffb050, 0.55 + curW[j] * 0.9);
      content.add(p);
      particles.push({ mesh: p, curve: curve, phase: j / data.M, speed: 0.28, size: 0.55 + curW[j] * 0.9 });
    }

    // ===== 权重矩阵热力图（8×8，高权重向前凸起）=====
    var matGroup = new THREE.Group();
    matGroup.position.copy(MAT_POS);
    matGroup.rotation.x = -0.22;
    content.add(matGroup);
    var cell = 0.66, gap = 0.06, step = cell + gap;
    var half = (data.M - 1) * step / 2;
    // 背板（深色玻璃感）
    var plate = new THREE.Mesh(
      new THREE.BoxGeometry(data.M * step + 0.35, data.M * step + 0.35, 0.12),
      new THREE.MeshStandardMaterial({ color: 0x141a26, metalness: 0.4, roughness: 0.5 })
    );
    plate.position.z = -0.1;
    plate.receiveShadow = true;
    matGroup.add(plate);
    // 颜色/凸起按矩阵内最大值归一化：最高权重 = 纯红，最低 = 深蓝
    var maxW = 0;
    for (i = 0; i < data.M; i++) for (var j2 = 0; j2 < data.M; j2++) {
      if (data.weights[i][j2] > maxW) maxW = data.weights[i][j2];
    }
    for (var i = 0; i < data.M; i++) {
      for (var jj = 0; jj < data.M; jj++) {
        var wn = data.weights[i][jj] / maxW; // 0..1
        var isCur = (i === cur);
        var c = new THREE.Mesh(sharedGeo.cell, new THREE.MeshStandardMaterial({
          color: heatColor(wn),
          metalness: 0.3, roughness: 0.35,
          emissive: heatColor(wn).multiplyScalar(isCur ? 0.5 : 0.1)
        }));
        var depth = 0.08 + wn * 2.6;
        c.scale.set(1, 1, depth);
        c.position.set(jj * step - half, half - i * step, depth / 2);
        c.castShadow = true;
        matGroup.add(c);
      }
    }
    var matLabel = makeTextSprite('softmax(QKᵀ/√d) 权重矩阵', { color: '#ff8a7a', height: 0.46 });
    matLabel.position.set(0, half + 0.75, 0.4);
    matGroup.add(matLabel);

    // ===== 输出向量（Σ wᵢVᵢ，紫色，入场逐元素生长）=====
    var outCol = makeVectorColumn(data.output, 0.78, { collect: outSpheres }); // 紫
    outCol.position.set(OUTX, BASE_Y, 0);
    content.add(outCol);
    bobbers.push({ obj: outCol, baseY: BASE_Y, phase: 1, speed: 0.8 });
    var oLabel = makeTextSprite('输出 = Σ wᵢ·Vᵢ', { color: '#d9a8ff', height: 0.5 });
    oLabel.position.set(OUTX, BASE_Y + DIM * SPHERE_GAP + 0.85, 0);
    content.add(oLabel);

    // ===== V→输出 加权求和弧线（∝ 权重）+ 粒子 =====
    var outMid = new THREE.Vector3(OUTX, BASE_Y + DIM * SPHERE_GAP * 0.5, 0);
    for (j = 0; j < data.M; j++) {
      var vTop = new THREE.Vector3(COL_X0 + j * COL_DX, BASE_Y + DIM * SPHERE_GAP + 0.1, VZ);
      var c2 = bezier(vTop, outMid, 1.4 + (data.M - j) * 0.1);
      var l2 = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(c2.getPoints(40)),
        new THREE.LineBasicMaterial({ color: 0x70e080, transparent: true, opacity: 0.12 + curW[j] * 0.6 })
      );
      content.add(l2);
      var p2 = makeParticle(0x90f0a0, 0.5 + curW[j] * 1.0);
      content.add(p2);
      particles.push({ mesh: p2, curve: c2, phase: j / data.M + 0.3, speed: 0.24, size: 0.5 + curW[j] * 1.0 });
    }

    // ===== 权重矩阵当前行 → 输出 粒子 =====
    var rowY = MAT_POS.y + (half - cur * step) * Math.cos(0.22);
    var rowStart = new THREE.Vector3(0, rowY, MAT_POS.z + 1.2);
    for (j = 0; j < 4; j++) {
      var c3 = bezier(rowStart, new THREE.Vector3(OUTX, BASE_Y + 1.5 + j * 0.9, 0), 0.8);
      var p3 = makeParticle(0xff6a55, 0.7);
      content.add(p3);
      particles.push({ mesh: p3, curve: c3, phase: j * 0.25, speed: 0.32, size: 0.7 });
    }

    // ===== 地面（接阴影）=====
    var ground = new THREE.Mesh(sharedGeo.ground, new THREE.MeshStandardMaterial({
      color: 0x11151d, metalness: 0.1, roughness: 0.9
    }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = BASE_Y - 0.35;
    ground.receiveShadow = true;
    content.add(ground);
  }

  // 重建时清掉旧内容
  function disposeContent() {
    if (!content) return;
    content.traverse(function (o) {
      if (o.geometry && !isShared(o.geometry)) o.geometry.dispose();
      if (o.material) {
        var mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach(function (m) {
          if (m.map) m.map.dispose();
          m.dispose();
        });
      }
    });
    group.remove(content);
    content = null;
  }
  function isShared(geo) {
    for (var k in sharedGeo) if (sharedGeo[k] === geo) return true;
    return false;
  }

  // ---- 接口 ----
  return {
    outputData: null, // 传给下一模块：{q,k,v,weights,output}

    init: function (sc, cam, ren) {
      scene = sc; camera = cam; renderer = ren;
      sharedGeo.sphere = new THREE.SphereGeometry(SPHERE_R, 24, 18);
      sharedGeo.rod = new THREE.CylinderGeometry(0.045, 0.045, 1, 12);
      sharedGeo.cone = new THREE.ConeGeometry(0.12, 0.3, 16);
      sharedGeo.cell = new THREE.BoxGeometry(0.66, 0.66, 1);
      sharedGeo.particle = new THREE.SphereGeometry(0.09, 12, 10);
      sharedGeo.ground = new THREE.CircleGeometry(16, 48);

      group = new THREE.Group();
      group.visible = false;
      // 自给自足的灯光（乔布斯式质感：主光带阴影 + 天光 + 轮廓光）
      var hemi = new THREE.HemisphereLight(0xbfd4ff, 0x1a1410, 0.85);
      group.add(hemi);
      var key = new THREE.DirectionalLight(0xffffff, 2.4);
      key.position.set(7, 13, 9);
      key.castShadow = true;
      key.shadow.mapSize.set(2048, 2048);
      key.shadow.camera.left = -13; key.shadow.camera.right = 13;
      key.shadow.camera.top = 13; key.shadow.camera.bottom = -13;
      key.shadow.camera.far = 45;
      key.shadow.bias = -0.0004;
      group.add(key);
      var rim = new THREE.DirectionalLight(0x7ea8ff, 0.9);
      rim.position.set(-9, 5, -10);
      group.add(rim);
      scene.add(group);
    },

    show: function (data) {
      var computed = computeAttention(data);
      this.outputData = {
        q: computed.q, k: computed.k, v: computed.v,
        weights: computed.weights, output: computed.output
      };
      build(computed);
      state.shown = true;
      state.appear = 0;
      group.visible = true;
    },

    hide: function () {
      state.shown = false;
      if (group) group.visible = false;
    },

    update: function (delta) {
      if (!state.shown || !group || !group.visible) return;
      delta = Math.min(delta || 0.016, 0.1);
      state.time += delta;

      // 入场动画（整体缓出放大）
      if (state.appear < 1) {
        state.appear = Math.min(1, state.appear + delta * 0.9);
        var e = 1 - Math.pow(1 - state.appear, 3);
        group.scale.setScalar(0.001 + 0.999 * e);
      }

      // 输出向量变形：逐元素从 0 生长到目标大小（stagger）
      for (var i = 0; i < outSpheres.length; i++) {
        var os = outSpheres[i];
        var t = Math.max(0, Math.min(1, state.appear * 1.8 - os.index * 0.06));
        var grow = 1 - Math.pow(1 - t, 3);
        var target = (0.75 + Math.min(1, Math.abs(os.value) * 1.6) * 0.5);
        os.mesh.scale.setScalar(Math.max(0.001, grow * target));
      }

      // 轻微浮动
      for (i = 0; i < bobbers.length; i++) {
        var b = bobbers[i];
        b.obj.position.y = b.baseY + Math.sin(state.time * b.speed + b.phase) * 0.05;
      }

      // 数据流粒子
      for (i = 0; i < particles.length; i++) {
        var p = particles[i];
        p.phase = (p.phase + delta * p.speed) % 1;
        p.curve.getPoint(p.phase, p.mesh.position);
        var fade = Math.sin(Math.PI * p.phase); // 两端淡入淡出
        p.mesh.scale.setScalar(Math.max(0.001, p.size * fade));
        p.mesh.material.opacity = 0.35 + 0.6 * fade;
      }
    },

    dispose: function () {
      disposeContent();
      if (group) {
        scene.remove(group);
        group.traverse(function (o) {
          if (o.material) {
            var mats = Array.isArray(o.material) ? o.material : [o.material];
            mats.forEach(function (m) { m.dispose(); });
          }
        });
        group = null;
      }
      for (var k in sharedGeo) { sharedGeo[k].dispose(); delete sharedGeo[k]; }
      particles = []; bobbers = []; outSpheres = [];
      state.shown = false;
    },

    screenshot: function () {
      renderer.render(scene, camera);
      return renderer.domElement.toDataURL('image/png');
    }
  };
})();
