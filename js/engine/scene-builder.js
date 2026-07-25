/* scene-builder.js — 核心：吃 ModelSpec 建垂直塔 + 全部特效
 * 暴露: ENGINE.buildScene(spec) / ENGINE.playEffect(name)
 * 布局：塔底 token(≈y2) → Embedding 柱(≈y7) → 3 个可见 Block(y11.5 起) → logits/softmax 塔顶。
 * spec.flow 为模型空间（y 0→N，层距 1.5），buildScene 时等比映射到场景坐标（state.flowScene）。
 * layers 中 tokenizer/embedding/output/softmax 视为全局组件；其余按序在每个可见 Block 内堆叠。
 * 未知 type（attn / lightning-attn / nsa-attn 等占位）→ 暖灰通用块回退，可点击、不报错。
 */
window.ENGINE = window.ENGINE || {};
(function () {
  var E = window.ENGINE;

  // ---------- 布局常量 ----------
  var W = 17, D = 11.5;            // Block 截面尺寸
  var BASE_Y = 11.5, GAP = 1.3, MARGIN = 0.15;
  var TOKEN_N = 8;
  var SLOT = { norm: 0.7, kda: 2.3, mla: 2.3, router: 1.5, experts: 3.2, attnres: 2.3, _unknown: 1.6 };
  var GLOBAL_TYPES = { tokenizer: 1, embedding: 1, output: 1, softmax: 1 };
  var EXP_COLS = 36, EXP_GAP = 0.42, EXP_SIZE = 0.33;

  function COL() { return E.COLORS; }
  function slotH(t) { return SLOT[t] || SLOT._unknown; }
  function S() { return E._state; }

  // ---------- 小动画工具：apply(intensity 0→1→0) ----------
  function pulseAnim(apply, dur, delay, done) {
    S().anims.push({
      t: 0, dur: dur || 0.6, delay: delay || 0,
      update: function (k) { apply(Math.sin(Math.PI * k)); },
      done: done || null
    });
  }

  // ---------- 贴图工具 ----------
  function tokenTexture(id) {
    var c = document.createElement('canvas');
    c.width = c.height = 64;
    var g = c.getContext('2d');
    g.fillStyle = '#241d15'; g.fillRect(0, 0, 64, 64);
    g.strokeStyle = '#4a3d2a'; g.lineWidth = 3; g.strokeRect(2, 2, 60, 60);
    g.fillStyle = '#e8dcc8'; g.font = '20px Menlo, monospace';
    g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText(String(id), 32, 33);
    var t = new THREE.CanvasTexture(c);
    if (THREE.SRGBColorSpace) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }

  function heatTexture() {
    var c = document.createElement('canvas');
    c.width = 96; c.height = 54;
    var g = c.getContext('2d');
    for (var y = 0; y < 54; y += 6) {
      for (var x = 0; x < 96; x += 6) {
        var v = Math.random();
        v = v * v; // 偏暗，少量亮格
        var r = Math.round(14 + (108 - 14) * v);
        var gg = Math.round(26 + (136 - 26) * v);
        var b = Math.round(42 + (247 - 42) * v);
        g.fillStyle = 'rgb(' + r + ',' + gg + ',' + b + ')';
        g.fillRect(x, y, 6, 6);
      }
    }
    var t = new THREE.CanvasTexture(c);
    t.magFilter = THREE.NearestFilter;
    if (THREE.SRGBColorSpace) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }

  function textSprite(text, px, color, scaleX, scaleY) {
    var c = document.createElement('canvas');
    c.width = 256; c.height = 128;
    var g = c.getContext('2d');
    g.font = (px || 56) + 'px "Songti SC", "PingFang SC", serif';
    g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillStyle = color || '#c9bba6';
    g.fillText(text, 128, 66);
    var tex = new THREE.CanvasTexture(c);
    if (THREE.SRGBColorSpace) tex.colorSpace = THREE.SRGBColorSpace;
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    sp.scale.set(scaleX || 7, scaleY || 3.5, 1);
    return sp;
  }

  // ---------- 组件注册 ----------
  function reg(type, group, meshes, center, radius, pulse) {
    var s = S();
    for (var i = 0; i < meshes.length; i++) {
      meshes[i].userData.componentId = type;
      s.pickables.push(meshes[i]);
    }
    if (!s.components[type]) s.components[type] = [];
    s.components[type].push({ group: group, center: center, radius: radius, pulse: pulse || null });
    return s.components[type][s.components[type].length - 1];
  }

  // 通用暖灰件的脉冲：发光一闪
  function emissiveFlash(mats, dur) {
    return function () {
      pulseAnim(function (k) {
        for (var i = 0; i < mats.length; i++) mats[i].emissiveIntensity = 0.3 + 1.3 * k;
      }, dur || 0.7);
    };
  }

  // ============ 各组件建模 ============

  // 底部输入区：8 个 token 立方体（永远建，契约组件 tokenizer）
  function buildTokens(tower) {
    var s = S();
    var group = new THREE.Group();
    var mats = [], cubes = [];
    for (var i = 0; i < TOKEN_N; i++) {
      var id = 10000 + Math.floor(Math.random() * 49999);
      var mat = new THREE.MeshStandardMaterial({
        map: tokenTexture(id), roughness: 0.6, metalness: 0.05,
        emissive: 0x664411, emissiveIntensity: 0.3
      });
      var cube = new THREE.Mesh(new THREE.BoxGeometry(1.9, 1.9, 1.9), mat);
      cube.position.set(-8.75 + i * 2.5, 2.2, 0);
      group.add(cube);
      mats.push(mat); cubes.push(cube);
    }
    tower.add(group);
    reg('tokenizer', group, cubes, new THREE.Vector3(0, 2.2, 0), 9.5, function () {
      for (var j = 0; j < cubes.length; j++) {
        (function (cube, mat, j) {
          pulseAnim(function (k) {
            mat.emissiveIntensity = 0.3 + 1.4 * k;
            cube.position.y = 2.2 + 0.35 * k;
          }, 0.55, j * 0.06);
        })(cubes[j], mats[j], j);
      }
    });
    s._fx.tokens = { cubes: cubes, mats: mats };
  }

  // Embedding 发光柱（永远建）
  function buildEmbedding(tower) {
    var s = S();
    var group = new THREE.Group();
    var core = new THREE.Mesh(
      new THREE.BoxGeometry(2.4, 5.4, 2.4),
      new THREE.MeshStandardMaterial({ color: 0x3a3129, roughness: 0.5, emissive: 0xe8a94d, emissiveIntensity: 0.35 })
    );
    core.position.y = 6.8;
    var glow = new THREE.Mesh(
      new THREE.BoxGeometry(3.1, 6.1, 3.1),
      new THREE.MeshBasicMaterial({ color: 0xe8a94d, transparent: true, opacity: 0.14, blending: THREE.AdditiveBlending, depthWrite: false })
    );
    glow.position.y = 6.8;
    group.add(core); group.add(glow);
    tower.add(group);
    reg('embedding', group, [core], new THREE.Vector3(0, 6.8, 0), 3.2, function () {
      pulseAnim(function (k) {
        glow.material.opacity = 0.14 + 0.4 * k;
        core.material.emissiveIntensity = 0.35 + 1.1 * k;
      }, 0.8);
    });
    s._idleFns.push(function (dt, t) {
      glow.material.opacity = Math.max(glow.material.opacity, 0.13 + 0.05 * Math.sin(t * 1.3));
    });
  }

  // KDA：琥珀 d×d 平板，顶点色，'kda-write' 时写入波从左向右扫
  function buildKda(tower, y, bi) {
    var s = S();
    var group = new THREE.Group();
    var base = new THREE.Mesh(
      new THREE.BoxGeometry(14.2, 0.22, 8.7),
      new THREE.MeshStandardMaterial({ color: 0x5c421e, roughness: 0.7, emissive: 0x201604, emissiveIntensity: 0.5 })
    );
    base.position.y = y - 0.12;
    group.add(base);

    var geo = new THREE.PlaneGeometry(14, 8.5, 16, 10);
    var posAttr = geo.getAttribute('position');
    var n = posAttr.count;
    var baseCol = new Float32Array(n * 3);
    var xs = new Float32Array(n), ys = new Float32Array(n);
    for (var i = 0; i < n; i++) {
      var x = posAttr.getX(i), yy = posAttr.getY(i);
      xs[i] = x; ys[i] = yy;
      var b = 0.55 + 0.3 * Math.sin(x * 1.3) * Math.sin(yy * 1.1);
      if (Math.round((x + 7) / 0.875) % 4 === 0) b *= 0.72; // 记忆通道暗纹
      baseCol[i * 3] = 0.91 * b; baseCol[i * 3 + 1] = 0.66 * b; baseCol[i * 3 + 2] = 0.30 * b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(baseCol.slice(), 3));
    geo.rotateX(-Math.PI / 2);
    var mat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
    var plate = new THREE.Mesh(geo, mat);
    plate.position.y = y + 0.05;
    group.add(plate);
    tower.add(group);

    var plateRec = { geo: geo, attr: geo.getAttribute('color'), base: baseCol, xs: xs, ys: ys, mat: mat };
    s._fx.kda.push(plateRec);

    reg('kda', group, [plate, base], new THREE.Vector3(0, y, 0), 7.5, function () {
      kdaWave(plateRec, 0);
    });
    s._idleFns.push(function (dt, t) {
      mat.color.setScalar(0.9 + 0.1 * Math.sin(t * 1.1 + bi * 1.7));
    });
  }

  function kdaWave(p, delay) {
    S().anims.push({
      t: 0, dur: 1.7, delay: delay || 0,
      update: function (k) {
        var writeX = -7 + 14 * k;
        var arr = p.attr.array, base = p.base, fade = 1 - k * 0.55;
        for (var i = 0; i < p.xs.length; i++) {
          var d = p.xs[i] - writeX;
          var bright = Math.exp(-d * d / 2.2) * fade * (0.8 + 0.2 * Math.cos(p.ys[i]));
          var m = 1 + 2.1 * bright;
          arr[i * 3] = Math.min(base[i * 3] * m, 1);
          arr[i * 3 + 1] = Math.min(base[i * 3 + 1] * m, 1);
          arr[i * 3 + 2] = Math.min(base[i * 3 + 2] * m, 1);
        }
        p.attr.needsUpdate = true;
      },
      done: function () {
        p.attr.array.set(p.base);
        p.attr.needsUpdate = true;
      }
    });
  }

  // MLA：青蓝热力网格，明暗闪动
  function buildMla(tower, y, bi) {
    var s = S();
    var group = new THREE.Group();
    var base = new THREE.Mesh(
      new THREE.BoxGeometry(14.2, 0.22, 8.7),
      new THREE.MeshStandardMaterial({ color: 0x1c2733, roughness: 0.7, emissive: 0x0a1420, emissiveIntensity: 0.5 })
    );
    base.position.y = y - 0.12;
    group.add(base);
    var geo = new THREE.PlaneGeometry(14, 8.5);
    geo.rotateX(-Math.PI / 2);
    var mat = new THREE.MeshBasicMaterial({ map: heatTexture(), side: THREE.DoubleSide });
    var plate = new THREE.Mesh(geo, mat);
    plate.position.y = y + 0.05;
    group.add(plate);
    tower.add(group);

    var rec = { mat: mat, cur: 1, tgt: 1, next: Math.random() * 0.15, boost: 0 };
    s._fx.mla.push(rec);
    reg('mla', group, [plate, base], new THREE.Vector3(0, y, 0), 7.5, function () {
      pulseAnim(function (k) { rec.boost = 1.1 * k; }, 0.7);
    });
    // 明暗闪动：间歇换新目标亮度并缓动
    s._idleFns.push(function (dt, t) {
      rec.next -= dt;
      if (rec.next <= 0) { rec.next = 0.1 + Math.random() * 0.25; rec.tgt = 0.7 + Math.random() * 0.5; }
      rec.cur += (rec.tgt - rec.cur) * Math.min(dt * 9, 1);
      mat.color.setScalar(rec.cur * (1 + rec.boost));
    });
  }

  // Router：松绿小方块
  function buildRouter(tower, y, bi) {
    var s = S();
    var group = new THREE.Group();
    var mat = new THREE.MeshStandardMaterial({ color: COL().moe, roughness: 0.4, emissive: 0x0d3f35, emissiveIntensity: 0.6 });
    var cube = new THREE.Mesh(new THREE.BoxGeometry(1.4, 1.4, 1.4), mat);
    cube.position.y = y;
    group.add(cube);
    tower.add(group);
    reg('router', group, [cube], new THREE.Vector3(0, y, 0), 2.2, function () {
      pulseAnim(function (k) {
        cube.scale.setScalar(1 + 0.35 * k);
        mat.emissiveIntensity = 0.6 + 1.2 * k;
      }, 0.6);
    });
  }

  // Experts：InstancedMesh 立方体方阵 + shared 常蓝；'route' 点亮 active 个并连线 router
  function buildExperts(tower, y, bi, spec) {
    var s = S();
    var moe = spec.moe || { experts: 256, active: 8, shared: 0 };
    var n = Math.max(1, moe.experts | 0);
    var group = new THREE.Group();

    var plate = new THREE.Mesh(
      new THREE.BoxGeometry(15.6, 0.25, 10.9),
      new THREE.MeshStandardMaterial({ color: 0x223530, roughness: 0.8 })
    );
    plate.position.y = y - 1.35;
    group.add(plate);

    var rows = Math.ceil(n / EXP_COLS);
    var mesh = new THREE.InstancedMesh(
      new THREE.BoxGeometry(EXP_SIZE, EXP_SIZE, EXP_SIZE),
      new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.1 }),
      n
    );
    var dummy = new THREE.Object3D();
    var baseColor = new THREE.Color(0x2e6f62);
    for (var i = 0; i < n; i++) {
      var cx = (i % EXP_COLS - (EXP_COLS - 1) / 2) * EXP_GAP;
      var cz = (Math.floor(i / EXP_COLS) - (rows - 1) / 2) * EXP_GAP;
      dummy.position.set(cx, y - 1.0, cz);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, baseColor);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    group.add(mesh);

    // shared 专家：常蓝，排在方阵前缘
    var sharedCubes = [];
    for (var j = 0; j < (moe.shared || 0); j++) {
      var sc = new THREE.Mesh(
        new THREE.BoxGeometry(0.9, 0.9, 0.9),
        new THREE.MeshStandardMaterial({ color: 0x6f9ff0, roughness: 0.4, emissive: 0x24407a, emissiveIntensity: 0.7 })
      );
      sc.position.set((j - ((moe.shared || 1) - 1) / 2) * 1.6, y - 0.9, -(D / 2 + 0.4));
      group.add(sc);
      sharedCubes.push(sc);
    }
    tower.add(group);

    var entry = reg('experts', group, [mesh].concat(sharedCubes), new THREE.Vector3(0, y - 1, 0), 8, function () {
      flashInstances(mesh, baseColor, 8, 0xa8f5e0);
    });
    entry.expertsRec = { mesh: mesh, baseColor: baseColor, n: n, active: moe.active || 8, block: bi, sharedCubes: sharedCubes };
    s._fx.experts.push(entry);
  }

  function flashInstances(mesh, baseColor, count, brightHex) {
    var n = mesh.count;
    var bright = new THREE.Color(brightHex);
    var ids = [];
    for (var i = 0; i < Math.min(count, n); i++) ids.push(Math.floor(Math.random() * n));
    pulseAnim(function (k) {
      var c = new THREE.Color().copy(baseColor).lerp(bright, k);
      for (var j = 0; j < ids.length; j++) mesh.setColorAt(ids[j], c);
      mesh.instanceColor.needsUpdate = true;
    }, 0.9, 0, function () {
      for (var j = 0; j < ids.length; j++) mesh.setColorAt(ids[j], baseColor);
      mesh.instanceColor.needsUpdate = true;
    });
  }

  // 'route'：随机 active 个专家变亮 + 画线连 router
  function routeEffect(entry, delay) {
    var s = S();
    var rec = entry.expertsRec;
    var mesh = rec.mesh, n = rec.n;
    var bright = new THREE.Color(0xa8f5e0);
    var ids = [], used = {};
    while (ids.length < Math.min(rec.active, n)) {
      var id = Math.floor(Math.random() * n);
      if (!used[id]) { used[id] = 1; ids.push(id); }
    }
    var tmpM = new THREE.Matrix4(), tmpP = new THREE.Vector3();
    s.anims.push({
      t: 0, dur: 2.6, delay: delay || 0,
      start: function () {
        // 连线：router（同 Block）→ 被点亮的专家
        var routers = s.components.router || [];
        var from = (routers[rec.block] || routers[0]);
        if (!from) return;
        mesh.updateMatrixWorld(true);
        var pts = [];
        var fromP = from.center.clone(); fromP.y += 0.8;
        for (var j = 0; j < ids.length; j++) {
          mesh.getMatrixAt(ids[j], tmpM);
          tmpP.setFromMatrixPosition(tmpM);
          mesh.localToWorld(tmpP);
          pts.push(fromP.clone(), tmpP.clone());
        }
        var geo = new THREE.BufferGeometry().setFromPoints(pts);
        var mat = new THREE.LineBasicMaterial({ color: 0x9ff5e2, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false });
        this._lines = new THREE.LineSegments(geo, mat);
        s.tower.add(this._lines);
        // shared 专家全程参与：亮度随路由起伏
        for (var q = 0; q < rec.sharedCubes.length; q++) rec.sharedCubes[q].material.emissiveIntensity = 1.4;
      },
      update: function (k) {
        var ramp = k < 0.12 ? k / 0.12 : (k > 0.7 ? Math.max(1 - (k - 0.7) / 0.3, 0) : 1);
        var c = new THREE.Color().copy(rec.baseColor).lerp(bright, Math.min(ramp * 1.6, 1));
        for (var j = 0; j < ids.length; j++) mesh.setColorAt(ids[j], c);
        mesh.instanceColor.needsUpdate = true;
        if (this._lines) this._lines.material.opacity = ramp * 0.85;
      },
      done: function () {
        for (var j = 0; j < ids.length; j++) mesh.setColorAt(ids[j], rec.baseColor);
        mesh.instanceColor.needsUpdate = true;
        if (this._lines) {
          s.tower.remove(this._lines);
          this._lines.geometry.dispose();
          this._lines.material.dispose();
        }
        for (var q = 0; q < rec.sharedCubes.length; q++) rec.sharedCubes[q].material.emissiveIntensity = 0.7;
      }
    });
  }

  // AttnRes：茜红半透明光束柱，射向下方 Block；无此组件的 spec 不建
  function buildAttnres(tower, y, bi) {
    var s = S();
    var group = new THREE.Group();
    var mat = new THREE.MeshBasicMaterial({
      color: COL().attnres, transparent: true, opacity: 0.3,
      blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
    });
    var beam = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.55, 4.4, 16, 1, true), mat);
    beam.position.set(W / 2 - 1.6, y - 1.2, 0);
    group.add(beam);
    tower.add(group);
    var rec = { mat: mat, beam: beam, bi: bi };
    s._fx.beams.push(rec);
    reg('attnres', group, [beam], new THREE.Vector3(W / 2 - 1.6, y - 1.2, 0), 1.6, function () {
      attnresPulse(rec, 0);
    });
    s._idleFns.push(function (dt, t) {
      if (!rec._busy) mat.opacity = 0.28 + 0.07 * Math.sin(t * 1.6 + bi * 2.1);
    });
  }

  function attnresPulse(rec, delay) {
    S().anims.push({
      t: 0, dur: 1.8, delay: delay || 0,
      start: function () { rec._busy = true; },
      update: function (k) {
        rec.mat.opacity = 0.3 + 0.6 * Math.abs(Math.sin(k * Math.PI * 2));
        var r = 1 + 0.4 * Math.sin(k * Math.PI);
        rec.beam.scale.set(r, 1, r);
      },
      done: function () {
        rec.mat.opacity = 0.3;
        rec.beam.scale.set(1, 1, 1);
        rec._busy = false;
      }
    });
  }

  // norm：暖灰薄板
  function buildNorm(tower, y) {
    var group = new THREE.Group();
    var mat = new THREE.MeshStandardMaterial({ color: 0x8a7d68, roughness: 0.8, emissive: 0x332b1e, emissiveIntensity: 0.3 });
    var slab = new THREE.Mesh(new THREE.BoxGeometry(14.6, 0.5, 9.2), mat);
    slab.position.y = y;
    group.add(slab);
    tower.add(group);
    reg('norm', group, [slab], new THREE.Vector3(0, y, 0), 7.5, emissiveFlash([mat], 0.6));
  }

  // 未知 type 回退：暖灰通用块 + 名称标注（attn / lightning-attn / nsa-attn 等占位）
  function buildGeneric(tower, type, y) {
    var group = new THREE.Group();
    var mat = new THREE.MeshStandardMaterial({ color: 0x8a7d68, roughness: 0.75, emissive: 0x332b1e, emissiveIntensity: 0.3 });
    var slab = new THREE.Mesh(new THREE.BoxGeometry(11, 1.0, 7.5), mat);
    slab.position.y = y;
    group.add(slab);
    var edge = new THREE.LineSegments(
      new THREE.EdgesGeometry(slab.geometry),
      new THREE.LineBasicMaterial({ color: 0xa89a83, transparent: true, opacity: 0.5 })
    );
    edge.position.y = y;
    group.add(edge);
    var label = textSprite(type, 40, '#c9bba6', 5.2, 2.6);
    label.position.set(0, y + 1.1, 0);
    group.add(label);
    tower.add(group);
    reg(type, group, [slab], new THREE.Vector3(0, y, 0), 6, emissiveFlash([mat], 0.7));
  }

  function buildLayer(tower, type, y, bi, spec) {
    switch (type) {
      case 'kda': return buildKda(tower, y, bi);
      case 'mla': return buildMla(tower, y, bi);
      case 'router': return buildRouter(tower, y, bi);
      case 'experts': return buildExperts(tower, y, bi, spec);
      case 'attnres': return buildAttnres(tower, y, bi);
      case 'norm': return buildNorm(tower, y);
      default: return buildGeneric(tower, type, y); // 占位/未知 type 回退
    }
  }

  // Block 外框：便于把一组层读作「一个 Block」
  function blockFrame(tower, y0, y1) {
    var h = y1 - y0 + 0.6;
    var geo = new THREE.EdgesGeometry(new THREE.BoxGeometry(W + 0.7, h, D + 0.7));
    var frame = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x4a4036, transparent: true, opacity: 0.55 }));
    frame.position.y = (y0 + y1) / 2;
    tower.add(frame);
  }

  // 塔顶 logits 行（layers 含 output 才建）
  function buildLogits(tower, y) {
    var group = new THREE.Group();
    var mats = [], cubes = [];
    var hot = Math.floor(Math.random() * TOKEN_N);
    for (var i = 0; i < TOKEN_N; i++) {
      var mat = new THREE.MeshStandardMaterial({
        color: 0x8a7d68, roughness: 0.6,
        emissive: 0x554433, emissiveIntensity: i === hot ? 0.9 : 0.3
      });
      var cube = new THREE.Mesh(new THREE.BoxGeometry(1.15, 1.15, 1.15), mat);
      cube.position.set(-6.3 + i * 1.8, y, 0);
      group.add(cube);
      mats.push(mat); cubes.push(cube);
    }
    tower.add(group);
    reg('output', group, cubes, new THREE.Vector3(0, y, 0), 7, emissiveFlash(mats, 0.7));
    return y;
  }

  // 塔顶 softmax 概率条（layers 含 softmax 才建）；'softmax' 特效做高度动画
  function buildSoftmax(tower, baseY) {
    var s = S();
    var group = new THREE.Group();
    var bars = [];
    var heights = [];
    for (var i = 0; i < TOKEN_N; i++) {
      var mat = new THREE.MeshStandardMaterial({ color: 0x8a7d68, roughness: 0.55, emissive: 0x2c251b, emissiveIntensity: 0.4 });
      var bar = new THREE.Mesh(new THREE.BoxGeometry(1.0, 1, 1.0), mat);
      var h = 0.5 + Math.random() * 1.5;
      bar.scale.y = h;
      bar.position.set(-5.95 + i * 1.7, baseY + h / 2, 0);
      group.add(bar);
      bars.push(bar); heights.push(h);
    }
    tower.add(group);
    var rec = { bars: bars, heights: heights, baseY: baseY };
    s._fx.softmax = rec;
    reg('softmax', group, bars, new THREE.Vector3(0, baseY + 2, 0), 7, function () {
      softmaxEffect();
    });
  }

  function softmaxEffect() {
    var rec = S()._fx.softmax;
    if (!rec) return;
    var winner = Math.floor(Math.random() * TOKEN_N);
    var raw = [];
    var sum = 0;
    for (var i = 0; i < TOKEN_N; i++) {
      raw[i] = i === winner ? 1 : Math.pow(Math.random(), 3) * 0.45;
      sum += raw[i];
    }
    var from = rec.heights.slice();
    var to = [];
    for (i = 0; i < TOKEN_N; i++) to[i] = 0.4 + (raw[i] / sum) * 5.4;
    S().anims.push({
      t: 0, dur: 1.0, delay: 0,
      update: function (k) {
        var e = 1 - Math.pow(1 - k, 3); // easeOut
        for (var i = 0; i < TOKEN_N; i++) {
          var h = from[i] + (to[i] - from[i]) * e;
          rec.bars[i].scale.y = h;
          rec.bars[i].position.y = rec.baseY + h / 2;
          rec.heights[i] = h;
        }
      },
      done: function () {
        for (var i = 0; i < TOKEN_N; i++) {
          rec.bars[i].material.color.setHex(i === winner ? COL().kda : 0x8a7d68);
          rec.bars[i].material.emissiveIntensity = i === winner ? 0.9 : 0.4;
        }
      }
    });
  }

  // 自回归弧形轨道（始终建，'autoregress' 时 token 滑回输入端）
  function buildArc(tower, topY) {
    var s = S();
    var pts = [
      new THREE.Vector3(9.2, topY - 0.5, 0),
      new THREE.Vector3(15, topY + 1.5, 4),
      new THREE.Vector3(17.5, topY * 0.5, 6),
      new THREE.Vector3(15, 4.5, 4),
      new THREE.Vector3(9.4, 2.2, 0)
    ];
    var curve = new THREE.CatmullRomCurve3(pts, false, 'catmullrom', 0.4);
    var geo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(48));
    var line = new THREE.Line(geo, new THREE.LineBasicMaterial({
      color: 0xffd9a0, transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, depthWrite: false
    }));
    tower.add(line);
    s._fx.arc = { curve: curve, tower: tower };
  }

  function autoregressEffect() {
    var s = S();
    if (!s._fx.arc) return;
    var curve = s._fx.arc.curve;
    var cube = new THREE.Mesh(
      new THREE.BoxGeometry(1.25, 1.25, 1.25),
      new THREE.MeshBasicMaterial({ color: 0xffd9a0, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false })
    );
    var ease = E._easeInOutCubic || function (k) { return k; };
    s.anims.push({
      t: 0, dur: 2.0, delay: 0,
      start: function () { s.tower.add(cube); },
      update: function (k) {
        curve.getPointAt(ease(k), cube.position);
        cube.rotation.y = k * 4;
      },
      done: function () {
        s.tower.remove(cube);
        cube.geometry.dispose();
        cube.material.dispose();
        E._pulseComponent('tokenizer', 1);
        E._pulseComponent('embedding', 1);
      }
    });
  }

  // spec.flow（模型空间，层距 1.5）→ 场景坐标等比映射
  function mapFlow(spec, topY) {
    var s = S();
    var pts = spec.flow || [];
    if (!pts.length) { s.flowScene = []; return; }
    var maxY = 1;
    for (var i = 0; i < pts.length; i++) if (pts[i][1] > maxY) maxY = pts[i][1];
    s.flowScene = pts.map(function (p) {
      return [p[0] * 2.8, 1.0 + (p[1] / maxY) * (topY - 2.0), p[2] * 2.8];
    });
  }

  function disposeTower() {
    var s = S();
    if (!s.tower) return;
    s.tower.traverse(function (o) {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        var mats = Array.isArray(o.material) ? o.material : [o.material];
        for (var i = 0; i < mats.length; i++) {
          if (mats[i].map) mats[i].map.dispose();
          mats[i].dispose();
        }
      }
    });
    s.scene.remove(s.tower);
    s.tower = null;
  }

  // ============ 主入口 ============
  E.buildScene = function (spec) {
    var s = S();
    if (!s.scene) { console.warn('[engine] 请先调用 ENGINE.init(canvas)'); return; }
    if (!spec) { console.warn('[engine] buildScene: spec 为空'); return; }
    if (typeof window.validateSpec === 'function') {
      var v = window.validateSpec(spec);
      if (v) spec = v;
    }
    s.spec = spec;

    disposeTower();
    s.anims.length = 0;
    s.pickables = [];
    s.components = {};
    s._idleFns = [];
    s._fx = { tokens: null, kda: [], mla: [], experts: [], beams: [], softmax: null, arc: null };

    var tower = new THREE.Group();
    s.tower = tower;

    // 分离全局组件与 Block 内组件（保持 spec.layers 顺序）
    var blockLayers = [], globals = {};
    (spec.layers || []).forEach(function (l) {
      if (!l || !l.type) return;
      if (GLOBAL_TYPES[l.type]) globals[l.type] = true;
      else blockLayers.push(l.type);
    });

    buildTokens(tower);
    buildEmbedding(tower);

    // 3 个可见 Block + ×N 标注
    var blockH = 0;
    blockLayers.forEach(function (t) { blockH += slotH(t) + MARGIN; });
    var y = BASE_Y, b3mid = BASE_Y;
    for (var b = 0; b < 3; b++) {
      if (!blockLayers.length) break;
      var ly = y;
      blockLayers.forEach(function (t) {
        buildLayer(tower, t, ly + slotH(t) / 2, b, spec);
        ly += slotH(t) + MARGIN;
      });
      blockFrame(tower, y, ly - MARGIN);
      if (b === 2) b3mid = (y + ly - MARGIN) / 2;
      y = ly + GAP;
    }
    var blocksTop = blockLayers.length ? (y - GAP) : BASE_Y;

    if (blockLayers.length) {
      var sprite = textSprite('× ' + (spec.blockCount || 1), 60, '#c9bba6', 7, 3.5);
      sprite.position.set(W / 2 + 5.5, b3mid, 0);
      tower.add(sprite);
    }

    // 塔顶
    var topY = blocksTop + 1.2;
    if (globals.output) { buildLogits(tower, topY); topY += 1.6; }
    if (globals.softmax) { buildSoftmax(tower, topY); topY += 6.4; }
    buildArc(tower, topY);

    s.towerH = topY;
    mapFlow(spec, topY);

    // 组件名牌：名字 + Block 编号（解决「不知道这东西属于哪里」）
    var LABELS = {
      tokenizer: '分词', embedding: '嵌入', kda: 'KDA', mla: 'MLA',
      router: '路由', experts: '专家阵', attnres: 'AttnRes',
      norm: '归一化', output: '输出', softmax: 'Softmax'
    };
    Object.keys(s.components).forEach(function (type) {
      var entries = s.components[type] || [];
      entries.forEach(function (entry, idx) {
        var name = LABELS[type] || type;
        if (entries.length > 1) name += ' · B' + (idx + 1);
        var lb = textSprite(name, 40, 'rgba(201,187,166,0.85)', 4.2, 2.0);
        // 名牌放塔左缘成一列：数值面板在组件 x+10.5（右侧），同侧会重叠
        lb.position.set(-(W / 2 + 3.0), entry.center.y + 1.7, entry.center.z);
        tower.add(lb);
      });
    });

    s.scene.add(tower);

    // 默认机位：首次 build 直接落位；切换模型保留用户当前视角
    // r ≈ topY*1.37：fov 46° 下整塔（含塔顶弧线）纵向填满 ~87% 视口，旧系数 0.84 会裁掉塔顶
    s.home = { pos: [topY * 0.82, topY * 0.40, topY * 1.10], target: [0, topY * 0.47, 0] };
    if (!s._builtOnce && E.camera && E.camera.setPose) E.camera.setPose(s.home.pos, s.home.target);
    s._builtOnce = true;

    if (E.flow && E.flow._rebuild) E.flow._rebuild();
  };

  // ============ 特效 ============
  var EFFECTS = {
    'route': function () {
      var list = S()._fx.experts;
      for (var i = 0; i < list.length; i++) routeEffect(list[i], i * 0.3);
    },
    'softmax': function () { softmaxEffect(); },
    'autoregress': function () { autoregressEffect(); },
    'attnres': function () {
      var beams = S()._fx.beams;
      for (var i = 0; i < beams.length; i++) attnresPulse(beams[i], i * 0.2);
    },
    'kda-write': function () {
      var plates = S()._fx.kda;
      for (var i = 0; i < plates.length; i++) kdaWave(plates[i], i * 0.25);
    }
  };

  E.playEffect = function (name) {
    var fn = EFFECTS[name];
    if (!fn) { console.warn('[engine] 未知特效: ' + name); return; }
    if (!S().tower) return;
    fn();
  };

  // 常驻 idle 动画驱动
  E._tickers = E._tickers || [];
  E._tickers.push(function (dt, t) {
    var fns = E._state._idleFns;
    if (!fns) return;
    for (var i = 0; i < fns.length; i++) fns[i](dt, t);
  });
})();
