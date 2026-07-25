/* camera.js — 自实现轨道相机：拖拽旋转 / 滚轮缩放 / 阻尼；flyTo 缓动飞行
 * 暴露: ENGINE.camera = {focusOn(componentId), reset(), flyTo(pos, target, duration, onDone)}
 */
window.ENGINE = window.ENGINE || {};
(function () {
  var E = window.ENGINE;

  function OrbitCam(camera, dom) {
    this.cam = camera;
    this.dom = dom;
    this.target = new THREE.Vector3(0, 8, 0);
    this.sph = { r: 46, theta: 0.6, phi: 1.05 };       // 当前球坐标
    this.want = { r: 46, theta: 0.6, phi: 1.05 };      // 阻尼目标
    this.flight = null;                                 // flyTo 动画
    this._bind();
  }

  OrbitCam.prototype._bind = function () {
    var self = this, dom = this.dom;
    var dragging = false, px = 0, py = 0, moved = 0;
    var pinch = 0;

    dom.addEventListener('pointerdown', function (e) {
      if (e.pointerType === 'touch') return; // touch 走 touch 事件
      dragging = true; moved = 0; px = e.clientX; py = e.clientY;
      try { dom.setPointerCapture(e.pointerId); } catch (err) {} // 合成事件/旧浏览器容错
      self.flight = null; // 用户接管
    });
    dom.addEventListener('pointermove', function (e) {
      if (!dragging || e.pointerType === 'touch') return;
      var dx = e.clientX - px, dy = e.clientY - py;
      px = e.clientX; py = e.clientY;
      moved += Math.abs(dx) + Math.abs(dy);
      self.want.theta -= dx * 0.0052;
      self.want.phi = clamp(self.want.phi - dy * 0.0052, 0.08, Math.PI - 0.08);
    });
    dom.addEventListener('pointerup', function () { dragging = false; });
    dom.addEventListener('wheel', function (e) {
      e.preventDefault();
      self.flight = null;
      self.want.r = clamp(self.want.r * (1 + e.deltaY * 0.0011), 4, 320);
    }, { passive: false });

    // 触屏：单指旋转，双指捏合缩放
    dom.addEventListener('touchstart', function (e) {
      self.flight = null;
      if (e.touches.length === 1) { px = e.touches[0].clientX; py = e.touches[0].clientY; }
      if (e.touches.length === 2) pinch = dist(e.touches[0], e.touches[1]);
    }, { passive: true });
    dom.addEventListener('touchmove', function (e) {
      e.preventDefault();
      if (e.touches.length === 1) {
        var t = e.touches[0];
        self.want.theta -= (t.clientX - px) * 0.006;
        self.want.phi = clamp(self.want.phi - (t.clientY - py) * 0.006, 0.08, Math.PI - 0.08);
        px = t.clientX; py = t.clientY;
      } else if (e.touches.length === 2) {
        var d = dist(e.touches[0], e.touches[1]);
        if (pinch > 0) self.want.r = clamp(self.want.r * (pinch / d), 4, 320);
        pinch = d;
      }
    }, { passive: false });

    function dist(a, b) { var dx = a.clientX - b.clientX, dy = a.clientY - b.clientY; return Math.sqrt(dx * dx + dy * dy); }
  };

  OrbitCam.prototype.flyTo = function (pos, target, duration, onDone) {
    this.flight = {
      t: 0, dur: Math.max(duration || 1.2, 0.01),
      p0: this.cam.position.clone(), p1: new THREE.Vector3(pos[0], pos[1], pos[2]),
      t0: this.target.clone(), t1: new THREE.Vector3(target[0], target[1], target[2]),
      onDone: onDone || null
    };
  };

  OrbitCam.prototype.update = function (dt) {
    if (this.flight) {
      var f = this.flight;
      f.t += dt;
      var k = easeInOutCubic(Math.min(f.t / f.dur, 1));
      this.cam.position.lerpVectors(f.p0, f.p1, k);
      this.target.lerpVectors(f.t0, f.t1, k);
      if (f.t >= f.dur) {
        this._syncFromCamera();
        var cb = f.onDone; this.flight = null;
        if (cb) cb();
      }
    } else {
      // 阻尼逼近
      var a = 1 - Math.exp(-dt * 7.5);
      this.sph.r += (this.want.r - this.sph.r) * a;
      this.sph.theta += (this.want.theta - this.sph.theta) * a;
      this.sph.phi += (this.want.phi - this.sph.phi) * a;
      var sp = this.sph, st = Math.sin(sp.phi);
      this.cam.position.set(
        this.target.x + sp.r * st * Math.sin(sp.theta),
        this.target.y + sp.r * Math.cos(sp.phi),
        this.target.z + sp.r * st * Math.cos(sp.theta)
      );
    }
    this.cam.lookAt(this.target);
  };

  // 飞行结束后把相机实际位姿写回球坐标，避免跳变
  OrbitCam.prototype._syncFromCamera = function () {
    var off = this.cam.position.clone().sub(this.target);
    var r = Math.max(off.length(), 0.001);
    this.sph.r = this.want.r = r;
    this.sph.theta = this.want.theta = Math.atan2(off.x, off.z);
    this.sph.phi = this.want.phi = Math.acos(clamp(off.y / r, -1, 1));
  };

  OrbitCam.prototype.setPose = function (pos, target) {
    this.target.set(target[0], target[1], target[2]);
    this.cam.position.set(pos[0], pos[1], pos[2]);
    this._syncFromCamera();
  };

  function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }
  function easeInOutCubic(k) { return k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2; }
  E._easeInOutCubic = easeInOutCubic;
  E._clamp = clamp;

  // 通过 init 钩子挂载（与 engine 各文件加载顺序无关）
  E._initHooks = E._initHooks || [];
  E._initHooks.push(function (canvasEl) {
    var s = E._state;
    if (s.orbit) return;
    s.orbit = new OrbitCam(s.camera, canvasEl);
    E._tickers.push(function (dt) { s.orbit.update(dt); });
  });

  E.camera = {
    flyTo: function (pos, target, dur, onDone) { E._state.orbit && E._state.orbit.flyTo(pos, target, dur, onDone); },
    setPose: function (pos, target) { E._state.orbit && E._state.orbit.setPose(pos, target); },
    focusOn: function (componentId) {
      var s = E._state, c = E._componentCenter(componentId);
      if (!c || !s.orbit) return;
      // 从当前相机方向退到组件前
      var dir = s.camera.position.clone().sub(c);
      if (dir.lengthSq() < 0.01) dir.set(1, 0.5, 1);
      dir.normalize();
      var entry = (s.components[componentId] || [])[0];
      var d = (entry ? entry.radius : 2) * 4.5 + 5;
      var pos = c.clone().add(dir.multiplyScalar(d));
      pos.y = Math.max(pos.y, c.y + 1.5);
      s.orbit.flyTo([pos.x, pos.y, pos.z], [c.x, c.y, c.z], 1.1);
    },
    reset: function () {
      var s = E._state;
      if (s.home && s.orbit) s.orbit.flyTo(s.home.pos, s.home.target, 1.4);
    }
  };
})();
