/* flow.js — 发光粒子沿 spec.flow 路径循环运动，途经组件时触发其 pulse
 * 暴露: ENGINE.flow = {start(), stop()}
 */
window.ENGINE = window.ENGINE || {};
(function () {
  var E = window.ENGINE;
  var N = 130;          // 粒子数
  var SPEED = 0.055;    // 曲线参数速度（圈/秒）

  var flow = {
    running: false,
    _obj: null,   // THREE.Points
    _curve: null,
    _offsets: null,
    _tickBound: false
  };

  // 生成径向光斑贴图
  function glowTexture() {
    var c = document.createElement('canvas');
    c.width = c.height = 64;
    var g = c.getContext('2d');
    var grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, 'rgba(255,225,180,1)');
    grad.addColorStop(0.4, 'rgba(255,205,140,0.55)');
    grad.addColorStop(1, 'rgba(255,200,120,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 64, 64);
    var tex = new THREE.CanvasTexture(c);
    return tex;
  }

  function buildCurve() {
    var s = E._state;
    // 优先用 scene-builder 等比映射后的场景坐标路径
    var pts = s.flowScene || (s.spec && s.spec.flow) || null;
    var v = [];
    if (pts && pts.length >= 2) {
      for (var i = 0; i < pts.length; i++) v.push(new THREE.Vector3(pts[i][0], pts[i][1], pts[i][2]));
    } else {
      // 默认路径：沿塔心向上，外部弧回底部，闭环
      var h = s.towerH || 30;
      v = [
        new THREE.Vector3(0, 1.2, 0), new THREE.Vector3(0, h * 0.3, 0),
        new THREE.Vector3(0, h * 0.7, 0), new THREE.Vector3(0, h + 1.5, 0),
        new THREE.Vector3(9, h + 2.5, 4), new THREE.Vector3(11, h * 0.5, 7),
        new THREE.Vector3(9, 2, 4), new THREE.Vector3(4, 0.8, 1.5)
      ];
      return new THREE.CatmullRomCurve3(v, true, 'catmullrom', 0.35);
    }
    // 首尾相近则闭环
    var closed = v[0].distanceTo(v[v.length - 1]) < 1.0;
    return new THREE.CatmullRomCurve3(v, closed, 'catmullrom', 0.2);
  }

  function buildParticles() {
    var s = E._state;
    removeParticles();
    flow._curve = buildCurve();
    var geo = new THREE.BufferGeometry();
    var pos = new Float32Array(N * 3);
    flow._offsets = new Float32Array(N);
    for (var i = 0; i < N; i++) flow._offsets[i] = i / N;
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    var mat = new THREE.PointsMaterial({
      size: 0.5, map: flow._tex || (flow._tex = glowTexture()),
      transparent: true, depthWrite: false,
      blending: THREE.AdditiveBlending, color: 0xffcf99, sizeAttenuation: true
    });
    var pts = new THREE.Points(geo, mat);
    pts.frustumCulled = false;
    pts.renderOrder = 5;
    (s.tower || s.scene).add(pts);
    flow._obj = pts;
  }

  function removeParticles() {
    var s = E._state;
    if (flow._obj) {
      flow._obj.parent && flow._obj.parent.remove(flow._obj);
      flow._obj.geometry.dispose();
      flow._obj.material.dispose();
      flow._obj = null;
    }
  }

  var sampleTmp = new THREE.Vector3();
  function tick(dt, t) {
    if (!flow.running || !flow._obj) return;
    var attr = flow._obj.geometry.getAttribute('position');
    var curve = flow._curve;
    for (var i = 0; i < N; i++) {
      var u = (flow._offsets[i] + t * SPEED) % 1;
      curve.getPointAt(u, sampleTmp);
      attr.setXYZ(i, sampleTmp.x, sampleTmp.y, sampleTmp.z);
    }
    attr.needsUpdate = true;

    // 途经检测：抽 8 个采样点，靠近组件中心则触发 pulse
    var comps = E._state.components;
    var now = t;
    for (var k = 0; k < 8; k++) {
      curve.getPointAt((t * SPEED + k / 8) % 1, sampleTmp);
      for (var type in comps) {
        var list = comps[type];
        for (var j = 0; j < list.length; j++) {
          var c = list[j];
          if (now - (c._lastFlowPulse || -9) < 1.2) continue;
          if (sampleTmp.distanceTo(c.center) < (c.radius + 1.2)) {
            c._lastFlowPulse = now;
            if (c.pulse) c.pulse(0.55);
          }
        }
      }
    }
  }

  flow.start = function () {
    flow.running = true;
    buildParticles();
  };
  flow.stop = function () {
    flow.running = false;
    removeParticles();
  };
  // buildScene 重建塔后由 scene-builder 调用：粒子挂靠到新塔
  flow._rebuild = function () { if (flow.running) buildParticles(); };

  // 常驻 tick（未运行时秒退），与文件加载顺序无关
  E._tickers = E._tickers || [];
  E._tickers.push(tick);

  E.flow = flow;
})();
