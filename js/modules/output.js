/* Output 模块 — logits / softmax / 采样 的真实 3D 可视化
 *
 * 布局（数据从左向右流动）：
 *   hidden state 向量（竖排小球，颜色=数值）
 *     → LM head 权重矩阵 W（体素网格，厚度+颜色=权重值）
 *       → logits 柱状图（前排，青色；可正可负）
 *         → softmax 概率柱状图（后排，绿色）
 *           → 采样动画（金色采样器跳动 → 落点高亮 → 新 token 飞出）
 *
 * 接口见 MODULE_SPEC.md：init / show / hide / update / dispose / screenshot
 */
(function () {
  'use strict';

  // ---- 演示用词表（16 个常见字，足够看清分布形状又不拥挤） ----
  var VOCAB = ['的', '是', '好', '我', '在', '有', '和', '就',
               '不', '人', '都', '一', '个', '上', '也', '很'];
  var D_MODEL = 12;          // hidden state 维度
  var HIDDEN_X = -17;        // hidden 向量列的 x 位置
  var MATRIX_X = -11.5;      // 权重矩阵中心 x
  var BAR_X0 = -6.5;         // 第一根柱子的 x
  var BAR_GAP = 1.0;         // 柱子间距
  var LOGIT_Z = 4.2;         // logits 排（前）
  var PROB_Z = -4.2;         // softmax 排（后）
  var GROUND_Y = -3.0;       // 地面高度
  var PROB_MAX_H = 6.5;      // 概率 100% 对应的柱高

  // ---- 模块状态 ----
  var scene, camera, renderer;
  var root = null;           // 模块自己的 group，整体挂/摘
  var built = false;
  var visible = false;

  var data = null;           // { hidden, logits, probs, sampledIndex }
  var anim = { t: 0, phase: 'idle', hopFrom: 0, hopTo: 0, hopT: 0,
               hops: [], hopIdx: 0 };

  // 需要每帧访问的对象
  var hiddenSpheres = [];
  var matrixMesh = null;
  var logitBars = [], probBars = [];
  var probLabels = [], tokenLabels = [];
  var sampler = null, samplerGlow = null;
  var resultSprite = null, outputToken = null, outputLabel = null;
  var particles = [];

  // ================= 工具 =================

  function softmax(logits) {
    var max = Math.max.apply(null, logits);
    var exps = logits.map(function (v) { return Math.exp(v - max); });
    var sum = exps.reduce(function (a, b) { return a + b; }, 0);
    return exps.map(function (e) { return e / sum; });
  }

  function multinomial(probs) {
    var r = Math.random(), acc = 0;
    for (var i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r <= acc) return i;
    }
    return probs.length - 1;
  }

  function demoData() {
    var logits = [], i;
    for (i = 0; i < VOCAB.length; i++) {
      logits.push((Math.random() - 0.5) * 10);
    }
    // 让一个 token 明显领先，分布更有故事性
    var peak = Math.floor(Math.random() * VOCAB.length);
    logits[peak] = 5.5 + Math.random() * 2;
    var probs = softmax(logits);
    var hidden = [];
    for (i = 0; i < D_MODEL; i++) hidden.push((Math.random() - 0.5) * 2);
    return { hidden: hidden, logits: logits, probs: probs,
             sampledIndex: multinomial(probs) };
  }

  // 文字 → sprite（canvas 纹理，高清 2x）
  function makeTextSprite(text, opt) {
    opt = opt || {};
    var size = opt.size || 44;
    var color = opt.color || '#e5e7eb';
    var pad = 16;
    var c = document.createElement('canvas');
    var ctx = c.getContext('2d');
    var font = '600 ' + size + 'px -apple-system, "PingFang SC", "Helvetica Neue", sans-serif';
    ctx.font = font;
    var w = Math.ceil(ctx.measureText(text).width) + pad * 2;
    var h = size + pad * 2;
    c.width = w * 2; c.height = h * 2;
    ctx = c.getContext('2d');
    ctx.scale(2, 2);
    ctx.font = font;
    if (opt.bg) {
      ctx.fillStyle = opt.bg;
      ctx.beginPath();
      ctx.roundRect(2, 2, w - 4, h - 4, 10);
      ctx.fill();
    }
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, w / 2, h / 2);
    var tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    var mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
    var sp = new THREE.Sprite(mat);
    var k = (opt.world || 1.0) / h;
    sp.scale.set(w * k, h * k, 1);
    return sp;
  }

  // 数值 [-1,1] → 蓝（负）…白…琥珀（正）
  function divergeColor(v, target) {
    var t = Math.max(-1, Math.min(1, v));
    var c = target || new THREE.Color();
    if (t >= 0) c.setRGB(0.94 * t + 0.9 * (1 - t) * 0.2 + 0.08,
                         0.62 * t + 0.35 * (1 - t),
                         0.10 * t + 0.55 * (1 - t));
    else { t = -t; c.setRGB(0.15 * t + 0.2 * (1 - t),
                            0.39 * t + 0.35 * (1 - t),
                            0.93 * t + 0.55 * (1 - t)); }
    return c;
  }

  function barX(i) { return BAR_X0 + i * BAR_GAP; }

  // ================= 搭建 =================

  function build() {
    root = new THREE.Group();

    // --- 地面：圆形阴影承接盘 ---
    var ground = new THREE.Mesh(
      new THREE.CircleGeometry(34, 64),
      new THREE.ShadowMaterial({ opacity: 0.28 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = GROUND_Y;
    ground.receiveShadow = true;
    root.add(ground);

    // --- hidden state：竖排小球 ---
    var sGeo = new THREE.SphereGeometry(0.30, 32, 24);
    for (var i = 0; i < D_MODEL; i++) {
      var m = new THREE.Mesh(sGeo, new THREE.MeshStandardMaterial({
        color: 0xffffff, roughness: 0.35, metalness: 0.1
      }));
      m.position.set(HIDDEN_X, 2.2 - i * 0.72, 0);
      m.castShadow = true;
      root.add(m);
      hiddenSpheres.push(m);
    }
    var hLabel = makeTextSprite('hidden state  h', { world: 1.0, color: '#cbd5e1' });
    hLabel.position.set(HIDDEN_X, 3.4, 0);
    root.add(hLabel);

    // --- LM head 权重矩阵：16 × 12 体素网格，厚度 = |w| ---
    var rows = VOCAB.length, cols = D_MODEL;
    var tile = 0.30, gap = 0.06;
    var mGeo = new THREE.BoxGeometry(tile, tile, 1);
    matrixMesh = new THREE.InstancedMesh(mGeo, new THREE.MeshStandardMaterial({
      roughness: 0.5, metalness: 0.15
    }), rows * cols);
    matrixMesh.castShadow = true;
    matrixMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    root.add(matrixMesh);
    var wLabel = makeTextSprite('W  (LM head)', { world: 1.0, color: '#cbd5e1' });
    wLabel.position.set(MATRIX_X, 3.4, 0);
    root.add(wLabel);

    // --- 段间箭头：h → W，W → logits ---
    root.add(makeArrow(HIDDEN_X + 0.7, 0.6, 0, MATRIX_X - 2.6, 0.6, 0, 0x94a3b8));
    root.add(makeArrow(MATRIX_X + 2.6, 0.6, 0, BAR_X0 - 1.3, 0.6, LOGIT_Z, 0x94a3b8));

    // --- 基线条（两排柱子各自的"坐标轴"） ---
    var baseMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.8 });
    var baseL = new THREE.Mesh(new THREE.BoxGeometry(VOCAB.length * BAR_GAP + 0.6, 0.06, 0.9), baseMat);
    baseL.position.set(barX((VOCAB.length - 1) / 2), GROUND_Y + 4.2, LOGIT_Z);
    baseL.receiveShadow = true;
    root.add(baseL);
    var baseP = baseL.clone();
    baseP.position.z = PROB_Z;
    root.add(baseP);

    // --- logits 柱（前）+ softmax 柱（后） ---
    var bGeo = new THREE.BoxGeometry(0.62, 1, 0.62);
    bGeo.translate(0, 0.5, 0);            // 底部为原点，scale.y 即高度
    for (i = 0; i < VOCAB.length; i++) {
      var bl = new THREE.Mesh(bGeo, new THREE.MeshStandardMaterial({
        color: 0x38bdf8, roughness: 0.3, metalness: 0.05,
        emissive: 0x0c4a6e, emissiveIntensity: 0.25
      }));
      bl.castShadow = true;
      bl.position.set(barX(i), baseL.position.y + 0.03, LOGIT_Z);
      root.add(bl);
      logitBars.push(bl);

      var bp = new THREE.Mesh(bGeo, new THREE.MeshStandardMaterial({
        color: 0x4ade80, roughness: 0.3, metalness: 0.05,
        emissive: 0x14532d, emissiveIntensity: 0.25
      }));
      bp.castShadow = true;
      bp.position.set(barX(i), baseP.position.y + 0.03, PROB_Z);
      root.add(bp);
      probBars.push(bp);

      // token 字符标签（前排下方，避开向下生长的负 logit 柱）
      var tl = makeTextSprite(VOCAB[i], { world: 0.62, color: '#94a3b8' });
      tl.position.set(barX(i), GROUND_Y + 0.6, LOGIT_Z + 1.2);
      root.add(tl);
      tokenLabels.push(tl);

      // 概率百分比标签（后排柱顶，初始隐藏）
      var pl = makeTextSprite('', { world: 0.5, color: '#bbf7d0' });
      pl.position.set(barX(i), baseP.position.y + 0.5, PROB_Z);
      pl.visible = false;
      root.add(pl);
      probLabels.push(pl);
    }

    // 排标签
    var lLabel = makeTextSprite('logits', { world: 0.9, color: '#7dd3fc' });
    lLabel.position.set(barX(VOCAB.length - 1) + 1.6, baseL.position.y + 0.4, LOGIT_Z);
    root.add(lLabel);
    var pLabel = makeTextSprite('softmax 概率', { world: 0.9, color: '#86efac' });
    pLabel.position.set(barX(VOCAB.length - 1) + 1.6, baseP.position.y + 0.4, PROB_Z);
    root.add(pLabel);

    // --- softmax 映射：一根宽箭头（前排 → 后排）+ 标签 ---
    root.add(makeArrow(barX(12), baseL.position.y + 0.3, LOGIT_Z - 0.8,
                       barX(12), baseL.position.y + 0.3, PROB_Z + 0.8,
                       0x64748b, 0.35));
    var smLabel = makeTextSprite('softmax', { world: 0.7, color: '#64748b' });
    smLabel.position.set(barX(12) + 1.6, baseL.position.y + 0.35, 0);
    root.add(smLabel);

    // --- 采样器：金色发光球 ---
    sampler = new THREE.Mesh(
      new THREE.SphereGeometry(0.34, 32, 24),
      new THREE.MeshStandardMaterial({
        color: 0xfbbf24, emissive: 0xf59e0b, emissiveIntensity: 1.2,
        roughness: 0.2
      })
    );
    sampler.castShadow = true;
    sampler.visible = false;
    root.add(sampler);
    samplerGlow = new THREE.PointLight(0xfbbf24, 0, 8);
    root.add(samplerGlow);

    // --- 采样结果标签 + 飞出的新 token ---
    resultSprite = makeTextSprite('', { world: 1.1, color: '#fde68a', bg: 'rgba(120,72,0,0.55)' });
    resultSprite.visible = false;
    root.add(resultSprite);

    outputToken = new THREE.Mesh(
      new THREE.SphereGeometry(0.42, 32, 24),
      new THREE.MeshStandardMaterial({
        color: 0xfbbf24, emissive: 0xd97706, emissiveIntensity: 0.5, roughness: 0.25
      })
    );
    outputToken.castShadow = true;
    outputToken.visible = false;
    root.add(outputToken);
    outputLabel = makeTextSprite('', { world: 0.9, color: '#fef3c7' });
    outputLabel.visible = false;
    root.add(outputLabel);

    // --- 数据流粒子（沿 h → W → logits → softmax 循环流动） ---
    var pGeo = new THREE.SphereGeometry(0.11, 12, 10);
    for (i = 0; i < 6; i++) {
      var p = new THREE.Mesh(pGeo, new THREE.MeshBasicMaterial({ color: 0x7dd3fc }));
      root.add(p);
      particles.push({ mesh: p, u: i / 6 });
    }

    root.visible = false;
    scene.add(root);
    built = true;
  }

  function makeArrow(x1, y1, z1, x2, y2, z2, color, opacity) {
    var from = new THREE.Vector3(x1, y1, z1), to = new THREE.Vector3(x2, y2, z2);
    var dir = to.clone().sub(from), len = dir.length();
    var g = new THREE.Group();
    var mat = new THREE.MeshStandardMaterial({
      color: color, transparent: true, opacity: opacity || 0.55, roughness: 0.6
    });
    var shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, len - 0.35, 10), mat);
    shaft.position.y = (len - 0.35) / 2;
    var head = new THREE.Mesh(new THREE.ConeGeometry(0.14, 0.35, 12), mat);
    head.position.y = len - 0.175;
    g.add(shaft); g.add(head);
    g.position.copy(from);
    g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
    return g;
  }

  // ================= 数据绑定 =================

  function applyData(d) {
    data = d;
    var i;

    // hidden 小球着色
    for (i = 0; i < D_MODEL; i++) {
      hiddenSpheres[i].material.color.copy(divergeColor(d.hidden[i]));
    }

    // 权重矩阵：随机但确定的权重（演示用），厚度+颜色编码
    var dummy = new THREE.Object3D(), col = new THREE.Color();
    var rows = VOCAB.length, cols = D_MODEL, tile = 0.30, gap = 0.06;
    var mw = cols * (tile + gap), mh = rows * (tile + gap);
    var idx = 0;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        // 伪随机权重：与行列相关的确定值，让"好"那一行更亮
        var w = Math.sin(r * 12.9898 + c * 78.233) * 0.9;
        if (r === d.sampledIndex) w = Math.abs(w) * 0.6 + 0.4;
        dummy.position.set(
          MATRIX_X - mw / 2 + c * (tile + gap) + tile / 2,
          0.6 + mh / 2 - r * (tile + gap) - tile / 2,
          0
        );
        dummy.scale.set(1, 1, 0.15 + Math.abs(w) * 0.85);
        dummy.updateMatrix();
        matrixMesh.setMatrixAt(idx, dummy.matrix);
        matrixMesh.setColorAt(idx, divergeColor(w, col));
        idx++;
      }
    }
    matrixMesh.instanceMatrix.needsUpdate = true;
    if (matrixMesh.instanceColor) matrixMesh.instanceColor.needsUpdate = true;

    // logits 柱：高度=值（负值向下），颜色按值在青色系内渐变
    var lmax = Math.max.apply(null, d.logits.map(Math.abs));
    for (i = 0; i < VOCAB.length; i++) {
      var v = d.logits[i];
      var bar = logitBars[i];
      bar.scale.y = Math.max(0.02, Math.abs(v) / lmax * 4.2);
      bar.rotation.x = v < 0 ? Math.PI : 0;
      var t = (v / lmax + 1) / 2;   // 0..1
      bar.material.color.setHSL(0.53, 0.75, 0.30 + t * 0.35);
    }

    // softmax 柱：高度=概率，绿色按概率渐变；百分比标签
    var pmax = Math.max.apply(null, d.probs);
    for (i = 0; i < VOCAB.length; i++) {
      var p = d.probs[i];
      probBars[i].scale.y = Math.max(0.02, p * PROB_MAX_H);
      probBars[i].material.color.setHSL(0.38, 0.65, 0.28 + (p / pmax) * 0.35);
      setSpriteText(probLabels[i], (p * 100).toFixed(1) + '%', '#bbf7d0');
    }

    // 采样跳动序列：按概率抽 6 个中间落点，最后落在 sampledIndex
    anim.hops = [];
    for (i = 0; i < 6; i++) anim.hops.push(multinomial(d.probs));
    if (anim.hops[anim.hops.length - 1] === d.sampledIndex) {
      anim.hops[anim.hops.length - 1] = (d.sampledIndex + 3) % VOCAB.length;
    }
    anim.hops.push(d.sampledIndex);

    // 结果标签 & 输出 token 文字
    setSpriteText(resultSprite, '采样结果  「' + VOCAB[d.sampledIndex] + '」  ' +
      (d.probs[d.sampledIndex] * 100).toFixed(1) + '%', '#fde68a');
    setSpriteText(outputLabel, VOCAB[d.sampledIndex], '#fef3c7');

    // 动画状态复位
    anim.t = 0; anim.phase = 'grow'; anim.hopIdx = 0; anim.hopT = 0;
    anim.hopFrom = anim.hops[0];
    sampler.visible = false; samplerGlow.intensity = 0;
    resultSprite.visible = false;
    outputToken.visible = false; outputLabel.visible = false;
    for (i = 0; i < VOCAB.length; i++) {
      probBars[i].material.emissive.setHex(0x14532d);
      probBars[i].material.emissiveIntensity = 0.25;
      probLabels[i].visible = false;
      logitBars[i].scale.y *= 0.001;   // 生长动画起点
      probBars[i].scale.y *= 0.001;
    }
    anim.fullLogitH = d.logits.map(function (v) {
      return Math.max(0.02, Math.abs(v) / lmax * 4.2);
    });
    anim.fullProbH = d.probs.map(function (p) {
      return Math.max(0.02, p * PROB_MAX_H);
    });
    // 百分比标签放到各自柱顶
    for (i = 0; i < VOCAB.length; i++) {
      probLabels[i].position.y = barY0 + anim.fullProbH[i] + 0.42;
    }
  }

  // sprite 换文字（重建纹理，次数少可接受）
  function setSpriteText(sprite, text, color) {
    var fresh = makeTextSprite(text, {
      world: sprite === resultSprite ? 1.1 : 0.5, color: color,
      bg: sprite === resultSprite ? 'rgba(120,72,0,0.55)' : undefined
    });
    sprite.material.map.dispose();
    sprite.material.map = fresh.material.map;
    sprite.scale.copy(fresh.scale);
    fresh.material.dispose();
  }

  // ================= 动画 =================

  var barY0 = 0;  // 基线 y（build 后缓存）

  function updateAnim(delta) {
    if (!visible || !data) return;
    anim.t += delta;
    var i, k;

    // 阶段 1：柱子生长（0 ~ 0.9s，逐根错峰）
    if (anim.phase === 'grow') {
      var done = true;
      for (i = 0; i < VOCAB.length; i++) {
        k = Math.min(1, Math.max(0, (anim.t - i * 0.03) / 0.55));
        k = 1 - Math.pow(1 - k, 3);   // ease-out
        logitBars[i].scale.y = anim.fullLogitH[i] * Math.max(0.001, k);
        probBars[i].scale.y = anim.fullProbH[i] * Math.max(0.001, k);
        if (k < 1) done = false;
        if (k > 0.6) probLabels[i].visible = data.probs[i] >= 0.005;   // 隐藏 0.0% 噪声
      }
      if (done) {
        anim.phase = 'sample';
        anim.t = 0;
        sampler.visible = true;
        samplerGlow.intensity = 2.5;
      }
    }

    // 阶段 2：采样器跳动（加权随机跳 6 次 → 落定）
    if (anim.phase === 'sample') {
      var HOP = 0.22;
      anim.hopT += delta;
      if (anim.hopT >= HOP && anim.hopIdx < anim.hops.length - 1) {
        anim.hopT = 0;
        anim.hopFrom = anim.hops[anim.hopIdx];
        anim.hopIdx++;
      }
      anim.hopTo = anim.hops[anim.hopIdx];
      k = Math.min(1, anim.hopT / HOP);
      var fx = barX(anim.hopFrom), tx = barX(anim.hopTo);
      var arc = Math.sin(k * Math.PI) * 1.1;   // 抛物线跳
      var bx = fx + (tx - fx) * k;
      var by = barY0 + anim.fullProbH[anim.hopTo] + 0.75 + arc;
      sampler.position.set(bx, by, PROB_Z);
      samplerGlow.position.copy(sampler.position);
      if (anim.hopIdx >= anim.hops.length - 1 && anim.hopT >= HOP) {
        anim.phase = 'land';
        anim.t = 0;
        var sb = probBars[data.sampledIndex];
        sb.material.emissive.setHex(0xf59e0b);
        sb.material.emissiveIntensity = 0.9;
        sb.material.color.setHex(0xfbbf24);
        resultSprite.position.set(barX(data.sampledIndex), barY0 + anim.fullProbH[data.sampledIndex] + 1.7, PROB_Z);
        resultSprite.visible = true;
      }
    }

    // 阶段 3：落定，新 token 飞出（0.6s 后起飞，向右飞出画面）
    if (anim.phase === 'land') {
      var si = data.sampledIndex;
      sampler.position.set(barX(si), barY0 + anim.fullProbH[si] + 0.75 + Math.sin(anim.t * 4) * 0.08, PROB_Z);
      samplerGlow.position.copy(sampler.position);
      if (anim.t > 0.7 && !outputToken.visible) {
        outputToken.visible = true; outputLabel.visible = true;
      }
      if (outputToken.visible) {
        var ft = Math.min(1, (anim.t - 0.7) / 1.4);
        ft = 1 - Math.pow(1 - ft, 3);
        outputToken.position.set(
          barX(si) + ft * (14 - barX(si)),
          barY0 + anim.fullProbH[si] + 0.6 + Math.sin(ft * Math.PI) * 1.4,
          PROB_Z
        );
        outputLabel.position.copy(outputToken.position).add(new THREE.Vector3(0, 0.85, 0));
      }
      if (anim.t > 2.6) anim.phase = 'idle';
    }

    // 数据流粒子：h → W → logits 中点 → softmax 中点 循环
    var path = [
      new THREE.Vector3(HIDDEN_X, 0.6, 0),
      new THREE.Vector3(MATRIX_X, 0.6, 0),
      new THREE.Vector3(BAR_X0 - 1.0, 0.6, LOGIT_Z),
      new THREE.Vector3(barX(7), 0.6, LOGIT_Z),
      new THREE.Vector3(barX(7), 0.6, PROB_Z)
    ];
    var segLens = [], total = 0;
    for (i = 0; i < path.length - 1; i++) {
      var L = path[i].distanceTo(path[i + 1]);
      segLens.push(L); total += L;
    }
    for (i = 0; i < particles.length; i++) {
      var pt = particles[i];
      pt.u = (pt.u + delta * 0.08) % 1;
      var dist = pt.u * total, seg = 0;
      while (dist > segLens[seg]) { dist -= segLens[seg]; seg++; }
      var kk = dist / segLens[seg];
      pt.mesh.position.lerpVectors(path[seg], path[seg + 1], kk);
      pt.mesh.position.y += Math.sin((pt.u * 40 + i) * 2) * 0.05;
    }
  }

  // ================= 模块接口 =================

  window.OUTPUT_MODULE = {

    init: function (sc, cam, ren) {
      scene = sc; camera = cam; renderer = ren;
      if (!built) build();
      barY0 = GROUND_Y + 4.2 + 0.03;
    },

    show: function (d) {
      if (!built) return;
      if (!d || !d.logits) {
        d = demoData();
      } else {
        // 兼容 spec 的 { logits, probs, sampledToken } 形式
        d = {
          hidden: d.hidden || d.input || demoData().hidden,
          logits: d.logits,
          probs: d.probs || softmax(d.logits),
          sampledIndex: (d.sampledToken && VOCAB.indexOf(d.sampledToken.text) >= 0)
            ? VOCAB.indexOf(d.sampledToken.text)
            : multinomial(d.probs || softmax(d.logits))
        };
      }
      applyData(d);
      root.visible = true;
      visible = true;
    },

    hide: function () {
      visible = false;
      if (root) root.visible = false;
    },

    update: function (delta) {
      updateAnim(delta);
    },

    dispose: function () {
      if (!root) return;
      root.traverse(function (o) {
        if (o.geometry) o.geometry.dispose();
        if (o.material) {
          var mats = Array.isArray(o.material) ? o.material : [o.material];
          mats.forEach(function (m) {
            if (m.map) m.map.dispose();
            m.dispose();
          });
        }
      });
      scene.remove(root);
      root = null; built = false; visible = false;
      hiddenSpheres = []; logitBars = []; probBars = [];
      probLabels = []; tokenLabels = []; particles = [];
    },

    screenshot: function () {
      renderer.render(scene, camera);
      return renderer.domElement.toDataURL('image/png');
    }
  };
})();
