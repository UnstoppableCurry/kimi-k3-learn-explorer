/* ============================================================================
 * Token/Embedding 模块 — 文本 → token → embedding → 位置编码 的 3D 可视化
 *
 * 真实结构（非隐喻）：
 *   - 每个 token     = 半径 0.5 的彩色小球（材质带粗糙度/金属感，投影到地面）
 *   - embedding 向量 = 从小球出发的 DIMS 根彩色细线（长度=|值|，颜色=数值热图）
 *                      + 一根 3D 箭头（前 3 维投影，显示向量方向）
 *   - 位置编码       = 环绕每个小球的旋转圆环（角速度/相位 = token position），
 *                      + 一条贯穿序列的正弦波缎带（sin/cos 位置编码曲线），随时间旋转推进
 *   - 数据流         = 发光粒子沿 token 序列流动（embedding 逐 token 进入网络）
 *
 * 接口（MODULE_SPEC.md v1）：
 *   window.TokenModule = { init, show, hide, update, dispose, screenshot }
 *   show(data) 接收: { tokens: [{ id, text, embedding:[...], position }] }
 * ========================================================================== */
(function () {
  'use strict';

  var DIMS = 16;          // 可视化的 embedding 维度数
  var SPACING = 3.2;      // token 沿 X 轴间距
  var RADIUS = 0.5;       // 小球半径（规范要求 0.5）

  // 确定性伪随机（无 embedding 数据时生成可复现的向量）
  function seeded(id) {
    var x = Math.sin(id * 127.1 + 311.7) * 43758.5453;
    return x - Math.floor(x);
  }
  function fakeEmbedding(id) {
    var v = [];
    for (var d = 0; d < DIMS; d++) v.push((seeded(id * 31 + d * 7) - 0.5) * 2);
    return v;
  }

  // 数值 → 热图颜色（负=蓝，零=暗，正=红橙），直观显示 embedding 每个维度的值
  function heatColor(v) {
    var t = Math.max(-1, Math.min(1, v));
    var c = new THREE.Color();
    if (t >= 0) c.setHSL(0.08 * (1 - t), 0.9, 0.25 + 0.35 * t);      // 暗红 → 亮橙
    else c.setHSL(0.62 - 0.08 * -t, 0.85, 0.2 + 0.3 * -t);            // 暗蓝 → 亮蓝
    return c;
  }

  function makeLabel(text) {
    var cv = document.createElement('canvas');
    cv.width = 256; cv.height = 128;
    var ctx = cv.getContext('2d');
    ctx.font = 'bold 64px "PingFang SC", "Helvetica Neue", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = 'rgba(0,0,0,0.6)';
    ctx.shadowBlur = 12;
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, 128, 64);
    var tex = new THREE.CanvasTexture(cv);
    tex.anisotropy = 4;
    var mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
    var sp = new THREE.Sprite(mat);
    sp.scale.set(2.2, 1.1, 1);
    return sp;
  }

  window.TokenModule = {
    _scene: null, _camera: null, _renderer: null,
    _root: null,            // 本模块所有对象的父节点
    _tokens: [],            // { group, sphere, ring, lines[], arrow, label, basePos }
    _wave: null,            // 位置编码正弦波缎带
    _particles: null,       // 数据流粒子
    _particleData: [],
    _time: 0,
    _visible: false,
    _disposables: [],

    init: function (scene, camera, renderer) {
      this._scene = scene;
      this._camera = camera;
      this._renderer = renderer;
      this._root = new THREE.Group();
      this._root.name = 'TokenModule';
      this._root.visible = false;
      scene.add(this._root);

      // ---- 光照（精致写实：主光投影 + 冷色补光 + 环境光） ----
      var amb = new THREE.AmbientLight(0x30364a, 1.2);
      var key = new THREE.DirectionalLight(0xfff2e0, 2.2);
      key.position.set(8, 14, 8);
      key.castShadow = true;
      key.shadow.mapSize.set(2048, 2048);
      key.shadow.camera.left = -20; key.shadow.camera.right = 20;
      key.shadow.camera.top = 20; key.shadow.camera.bottom = -20;
      key.shadow.bias = -0.0004;
      var fill = new THREE.DirectionalLight(0x6a8cff, 0.7);
      fill.position.set(-10, 6, -6);
      this._root.add(amb, key, fill);

      // ---- 地面：接收阴影的暗色磨砂平面 ----
      var groundMat = new THREE.MeshStandardMaterial({ color: 0x11141c, roughness: 0.95, metalness: 0.0 });
      var ground = new THREE.Mesh(new THREE.PlaneGeometry(120, 60), groundMat);
      ground.rotation.x = -Math.PI / 2;
      ground.position.y = -2.6;
      ground.receiveShadow = true;
      this._root.add(ground);
      this._disposables.push(ground.geometry, groundMat);

      var grid = new THREE.GridHelper(120, 60, 0x2a3350, 0x1a2033);
      grid.position.y = -2.59;
      this._root.add(grid);
    },

    show: function (data) {
      this._clearTokens();
      var tokens = (data && data.tokens) || [
        { id: 0, text: '你', position: 0 },
        { id: 1, text: '好', position: 1 },
        { id: 2, text: '，', position: 2 },
        { id: 3, text: '世', position: 3 },
        { id: 4, text: '界', position: 4 }
      ];
      var self = this;
      var n = tokens.length;
      var x0 = -(n - 1) * SPACING / 2;

      tokens.forEach(function (tok, i) {
        var emb = (tok.embedding && tok.embedding.length >= 3)
          ? tok.embedding.slice(0, DIMS)
          : fakeEmbedding(tok.id != null ? tok.id : i);
        var pos = (tok.position != null) ? tok.position : i;
        var base = new THREE.Vector3(x0 + i * SPACING, 0, 0);
        var hue = i / Math.max(1, n);
        var color = new THREE.Color().setHSL(hue * 0.85, 0.75, 0.55);

        var group = new THREE.Group();
        group.position.copy(base);

        // --- token 小球（半径 0.5，写实材质） ---
        var geo = new THREE.SphereGeometry(RADIUS, 48, 32);
        var mat = new THREE.MeshStandardMaterial({
          color: color, roughness: 0.25, metalness: 0.55,
          emissive: color, emissiveIntensity: 0.15
        });
        var sphere = new THREE.Mesh(geo, mat);
        sphere.castShadow = true;
        group.add(sphere);
        self._disposables.push(geo, mat);

        // --- embedding 向量：DIMS 根彩色细线，从小球表面出发向上扇形展开 ---
        var lines = [];
        for (var d = 0; d < DIMS; d++) {
          var v = emb[d] || 0;
          var len = 0.25 + Math.abs(v) * 1.4;
          var angle = (d / DIMS) * Math.PI * 2;
          var dir = new THREE.Vector3(Math.cos(angle) * 0.55, 1, Math.sin(angle) * 0.55).normalize();
          var lg = new THREE.CylinderGeometry(0.02, 0.02, len, 8);
          var lm = new THREE.MeshStandardMaterial({
            color: heatColor(v), roughness: 0.4, metalness: 0.3,
            emissive: heatColor(v), emissiveIntensity: 0.6
          });
          var seg = new THREE.Mesh(lg, lm);
          seg.position.copy(dir.clone().multiplyScalar(RADIUS + len / 2));
          seg.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
          group.add(seg);
          lines.push(seg);
          self._disposables.push(lg, lm);
        }

        // --- embedding 前 3 维投影：3D 箭头显示向量方向 ---
        var v3 = new THREE.Vector3(emb[0] || 0, emb[1] || 0.5, emb[2] || 0);
        if (v3.lengthSq() < 1e-6) v3.set(0, 1, 0);
        var arrow = new THREE.ArrowHelper(
          v3.clone().normalize(), new THREE.Vector3(0, 0, 0),
          1.2 + v3.length() * 0.5, color.getHex(), 0.22, 0.12
        );
        group.add(arrow);

        // --- 位置编码：环绕小球的旋转圆环（相位=position） ---
        var ringGeo = new THREE.TorusGeometry(0.85, 0.035, 12, 64);
        var ringMat = new THREE.MeshStandardMaterial({
          color: 0xffffff, roughness: 0.2, metalness: 0.8,
          emissive: color, emissiveIntensity: 0.5,
          transparent: true, opacity: 0.9
        });
        var ring = new THREE.Mesh(ringGeo, ringMat);
        ring.castShadow = true;
        group.add(ring);
        self._disposables.push(ringGeo, ringMat);

        // --- token 文本标签 ---
        var label = makeLabel(tok.text || String(tok.id != null ? tok.id : i));
        label.position.set(0, 2.6, 0);
        group.add(label);

        self._root.add(group);
        self._tokens.push({ group: group, sphere: sphere, ring: ring, lines: lines, arrow: arrow, label: label, basePos: base, position: pos, color: color });
      });

      this._buildPositionalWave(n, x0);
      this._buildParticles(n, x0);
      this._frameCamera(n);

      this._root.visible = true;
      this._visible = true;
      this._time = 0;
    },

    // 位置编码正弦波缎带：sin/cos(pos) 曲线沿序列展开，随时间旋转
    _buildPositionalWave: function (n, x0) {
      // sin/cos 位置编码曲线：y = sin(kx)，z = cos(kx)（螺旋正弦管）
      var pts = [];
      var width = (n - 1) * SPACING + 4;
      var steps = Math.max(64, n * 24);
      for (var s = 0; s <= steps; s++) {
        var k = s / steps;
        pts.push(new THREE.Vector3(
          x0 - 2 + k * width,
          Math.sin(k * Math.PI * 4) * 0.5,
          Math.cos(k * Math.PI * 2) * 0.35
        ));
      }
      var geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), steps, 0.03, 8, false);
      var mat = new THREE.MeshStandardMaterial({
        color: 0x8fd0ff, roughness: 0.3, metalness: 0.4,
        emissive: 0x3a7bd5, emissiveIntensity: 0.8
      });
      this._wave = new THREE.Mesh(geo, mat);
      this._wave.position.y = -1.6;   // 几何以自身 X 轴为中心，旋转即绕序列下方轴线
      this._wave.userData.width = width;
      this._wave.userData.x0 = x0 - 2;
      this._root.add(this._wave);
      this._disposables.push(geo, mat);
    },

    // 数据流粒子：沿 token 序列流动
    _buildParticles: function (n, x0) {
      var count = 24;
      var geo = new THREE.SphereGeometry(0.09, 12, 8);
      var mat = new THREE.MeshStandardMaterial({
        color: 0xffffff, emissive: 0xffe9a8, emissiveIntensity: 1.6, roughness: 0.3
      });
      this._particles = new THREE.Group();
      this._particleData = [];
      for (var i = 0; i < count; i++) {
        var m = new THREE.Mesh(geo, mat);
        this._particles.add(m);
        this._particleData.push({ mesh: m, t: i / count });
      }
      this._particles.userData = { x0: x0 - 1, x1: x0 + (n - 1) * SPACING + 1 };
      this._root.add(this._particles);
      this._disposables.push(geo, mat);
    },

    _frameCamera: function (n) {
      var width = (n - 1) * SPACING;
      if (this._camera) {
        this._camera.position.set(0, 4.5, Math.max(10, width * 0.9 + 7));
        this._camera.lookAt(0, 0, 0);
      }
    },

    hide: function () {
      this._root.visible = false;
      this._visible = false;
    },

    update: function (delta) {
      if (!this._visible) return;
      this._time += delta;
      var t = this._time;

      // 位置编码旋转动画：每个 token 的圆环以 position 决定的相位/速度旋转
      for (var i = 0; i < this._tokens.length; i++) {
        var tk = this._tokens[i];
        var phase = tk.position * (Math.PI / 4);          // 正弦位置编码的相位差
        tk.ring.rotation.x = t * (0.8 + tk.position * 0.15) + phase;
        tk.ring.rotation.y = Math.sin(t * 0.5 + phase) * 0.6;
        // 小球轻微浮动（呼吸感，不干扰结构）
        tk.sphere.position.y = Math.sin(t * 1.2 + phase) * 0.08;
        tk.label.position.y = 2.6 + Math.sin(t * 1.2 + phase) * 0.08;
      }

      // 正弦位置编码波：绕序列下方的 X 轴缓慢旋转（位置编码旋转动画的一部分）
      if (this._wave) {
        this._wave.rotation.x = t * 0.6;
      }

      // 数据流：粒子沿序列匀速流动，经过每个 token 时短暂增亮
      if (this._particles) {
        var u = this._particles.userData;
        for (var p = 0; p < this._particleData.length; p++) {
          var pd = this._particleData[p];
          pd.t = (pd.t + delta * 0.12) % 1;
          var px = u.x0 + pd.t * (u.x1 - u.x0);
          pd.mesh.position.set(px, -0.9 + Math.sin(pd.t * Math.PI * 6 + t) * 0.15, 0.8);
          var s = 1 + 0.5 * Math.sin(t * 4 + pd.t * 20);
          pd.mesh.scale.setScalar(s);
        }
      }
    },

    _clearTokens: function () {
      for (var i = 0; i < this._tokens.length; i++) {
        this._root.remove(this._tokens[i].group);
      }
      this._tokens = [];
      if (this._wave) { this._root.remove(this._wave); this._wave = null; }
      if (this._particles) { this._root.remove(this._particles); this._particles = null; this._particleData = []; }
    },

    dispose: function () {
      this._clearTokens();
      for (var i = 0; i < this._disposables.length; i++) {
        var d = this._disposables[i];
        if (d && typeof d.dispose === 'function') d.dispose();
      }
      this._disposables = [];
      if (this._root && this._scene) this._scene.remove(this._root);
      this._root = null;
      this._visible = false;
    },

    screenshot: function () {
      if (this._renderer && this._scene && this._camera) {
        this._renderer.render(this._scene, this._camera);
        return this._renderer.domElement.toDataURL('image/png');
      }
      return null;
    }
  };
})();
