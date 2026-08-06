/**
 * CORE_CAMERA — 相机控制（轨道拖拽 / 滚轮缩放 / 平移 / 飞行聚焦 / 复位）
 * 接口：init / show / hide / update / dispose / screenshot
 *
 * 不依赖 OrbitControls（three.min.js 不含 addon），手写球坐标轨道控制：
 * - 左键拖拽：绕目标旋转
 * - 右键/中键拖拽：平移目标点
 * - 滚轮：缩放（对数步进）
 * - flyTo(target, radius)：平滑飞行聚焦
 * - reset()：回到初始机位
 * 所有运动带阻尼（指数趋近），update(delta) 驱动。
 */
window.CORE_CAMERA = {
  camera: null,
  _dom: null,

  // 当前值与目标值（阻尼动画）
  _cur: null,   // {theta, phi, radius, tx, ty, tz}
  _goal: null,
  _home: null,
  _drag: null,
  _enabled: true,

  DAMP: 8,          // 阻尼系数（越大越跟手）
  MIN_R: 2, MAX_R: 90,
  MIN_PHI: 0.08, MAX_PHI: Math.PI - 0.08,
  MOVE_SPEED: 2,    // WASD 移动速度（单位/秒）
  _keys: {},        // 当前按下的键

  init: function (opts) {
    opts = opts || {};
    var aspect = opts.aspect || (window.innerWidth / window.innerHeight);
    var dom = opts.domElement || null;

    var cam = new THREE.PerspectiveCamera(45, aspect, 0.1, 500);
    this.camera = cam;
    this._dom = dom;

    this._home = { theta: 0.6, phi: 1.05, radius: 26, tx: 0, ty: 3, tz: 0 };
    this._cur  = JSON.parse(JSON.stringify(this._home));
    this._goal = JSON.parse(JSON.stringify(this._home));
    this._apply(this._cur);

    if (dom) this._bind(dom);
    return cam;
  },

  _bind: function (dom) {
    var self = this;
    dom.style.touchAction = 'none';

    dom.addEventListener('pointerdown', function (e) {
      if (!self._enabled) return;
      dom.setPointerCapture(e.pointerId);
      self._drag = { x: e.clientX, y: e.clientY, button: e.button };
    });
    dom.addEventListener('pointermove', function (e) {
      if (!self._drag || !self._enabled) return;
      var dx = e.clientX - self._drag.x;
      var dy = e.clientY - self._drag.y;
      self._drag.x = e.clientX; self._drag.y = e.clientY;
      var g = self._goal;
      if (self._drag.button === 0) {
        g.theta -= dx * 0.005;
        g.phi   -= dy * 0.005;
        g.phi = Math.max(self.MIN_PHI, Math.min(self.MAX_PHI, g.phi));
      } else {
        // 平移：沿相机右/上方向移动目标点
        var s = g.radius * 0.0012;
        var right = new THREE.Vector3(), up = new THREE.Vector3();
        self.camera.matrix.extractBasis(right, up, new THREE.Vector3());
        g.tx -= right.x * dx * s; g.ty += up.y * dy * s; g.tz -= right.z * dx * s;
        g.ty -= up.x * 0; // up 分量已含在 ty
      }
    });
    var end = function () { self._drag = null; };
    dom.addEventListener('pointerup', end);
    dom.addEventListener('pointercancel', end);

    dom.addEventListener('wheel', function (e) {
      if (!self._enabled) return;
      e.preventDefault();
      var g = self._goal;
      g.radius *= Math.pow(1.0015, e.deltaY);
      g.radius = Math.max(self.MIN_R, Math.min(self.MAX_R, g.radius));
    }, { passive: false });

    dom.addEventListener('contextmenu', function (e) { e.preventDefault(); });

    // WASD + QE 键盘控制
    window.addEventListener('keydown', function (e) {
      if (!self._enabled) return;
      self._keys[e.key.toLowerCase()] = true;
    });
    window.addEventListener('keyup', function (e) {
      self._keys[e.key.toLowerCase()] = false;
    });
  },

  /** 平滑飞到某个目标点。target: THREE.Vector3 或 {x,y,z} */
  flyTo: function (target, radius, phi, theta) {
    var g = this._goal;
    g.tx = target.x; g.ty = target.y; g.tz = target.z;
    if (radius != null) g.radius = Math.max(this.MIN_R, Math.min(this.MAX_R, radius));
    if (phi != null) g.phi = Math.max(this.MIN_PHI, Math.min(this.MAX_PHI, phi));
    if (theta != null) g.theta = theta;
  },

  /** 聚焦一个 Object3D：按包围球自动选距离 */
  focus: function (object3d) {
    var box = new THREE.Box3().setFromObject(object3d);
    if (box.isEmpty()) return;
    var sphere = box.getBoundingSphere(new THREE.Sphere());
    var dist = sphere.radius / Math.tan((this.camera.fov * Math.PI / 180) / 2) * 1.35;
    this.flyTo(sphere.center, dist);
  },

  reset: function () {
    this._goal = JSON.parse(JSON.stringify(this._home));
  },

  _apply: function (s) {
    var sinPhi = Math.sin(s.phi);
    this.camera.position.set(
      s.tx + s.radius * sinPhi * Math.sin(s.theta),
      s.ty + s.radius * Math.cos(s.phi),
      s.tz + s.radius * sinPhi * Math.cos(s.theta)
    );
    this.camera.lookAt(s.tx, s.ty, s.tz);
  },

  update: function (delta) {
    if (!this.camera) return;

    // WASD + QE 键盘移动
    var keys = this._keys;
    if (keys['w'] || keys['s'] || keys['a'] || keys['d'] || keys['q'] || keys['e']) {
      var g = this._goal;
      var speed = this.MOVE_SPEED * delta;
      var forward = new THREE.Vector3();
      var right = new THREE.Vector3();
      var up = new THREE.Vector3(0, 1, 0);

      // 计算相机朝向
      this.camera.getWorldDirection(forward);
      right.crossVectors(forward, up).normalize();

      // W/S: 前进/后退
      if (keys['w']) { g.tx += forward.x * speed; g.ty += forward.y * speed; g.tz += forward.z * speed; }
      if (keys['s']) { g.tx -= forward.x * speed; g.ty -= forward.y * speed; g.tz -= forward.z * speed; }

      // A/D: 左移/右移
      if (keys['a']) { g.tx -= right.x * speed; g.tz -= right.z * speed; }
      if (keys['d']) { g.tx += right.x * speed; g.tz += right.z * speed; }

      // Q/E: 上升/下降
      if (keys['q']) { g.ty += speed; }
      if (keys['e']) { g.ty -= speed; }
    }

    var k = 1 - Math.exp(-this.DAMP * delta); // 帧率无关阻尼
    var c = this._cur, g = this._goal;
    c.theta  += (g.theta  - c.theta)  * k;
    c.phi    += (g.phi    - c.phi)    * k;
    c.radius += (g.radius - c.radius) * k;
    c.tx += (g.tx - c.tx) * k;
    c.ty += (g.ty - c.ty) * k;
    c.tz += (g.tz - c.tz) * k;
    this._apply(c);
  },

  show: function () { this._enabled = true; },
  hide: function () { this._enabled = false; },

  dispose: function () {
    if (this._dom) {
      var clone = this._dom.cloneNode(false);
      // 不替换 DOM，仅清空引用；事件随页面生命周期结束
    }
    this._dom = null;
    this.camera = null;
    this._drag = null;
  },

  screenshot: function () {
    return window.CORE_RENDERER ? window.CORE_RENDERER.screenshot() : null;
  }
};
