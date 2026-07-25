/* renderer.js — 渲染器 + 雾效 + #171310 底色 + 蓝图网格地面
 * 暴露: ENGINE.init(canvasEl)；内部维护主循环与 _tickers。
 * 所有 engine 文件只做 window.ENGINE 增量挂载，跨文件引用全部发生在运行时。
 */
window.ENGINE = window.ENGINE || {};
(function () {
  var E = window.ENGINE;
  E.BG = 0x171310; // 暖炭背景（契约色）
  E.COLORS = { kda: 0xe8a94d, mla: 0x7aa2f7, moe: 0x46c8ae, attnres: 0xe0699b, generic: 0xa89a83 };
  E._state = {
    renderer: null, scene: null, camera: null, canvas: null,
    tower: null,          // 当前模型塔根节点
    pickables: [],        // 可点 mesh（userData.componentId）
    components: {},       // type -> [{group, center:V3, radius, pulse}]
    anims: [],            // 进行中的特效动画 {delay,dur,t,start,update,done}
    spec: null, home: null, time: 0, _idleFns: []
  };
  E._tickers = E._tickers || [];      // 每帧回调 fn(dt, t)；各文件加载顺序无关
  E._pickCbs = E._pickCbs || [];
  E._initHooks = E._initHooks || [];  // init 时执行的扩展钩子（camera/picking 注册）

  E.init = function (canvasEl) {
    var s = E._state;
    if (s.renderer) return; // 防重复 init
    s.canvas = canvasEl;
    var renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(E.BG, 1);
    if (THREE.SRGBColorSpace) renderer.outputColorSpace = THREE.SRGBColorSpace;
    s.renderer = renderer;

    var scene = new THREE.Scene();
    scene.background = new THREE.Color(E.BG);
    scene.fog = new THREE.FogExp2(E.BG, 0.006);
    s.scene = scene;

    s.camera = new THREE.PerspectiveCamera(46, 1, 0.1, 800);
    s.camera.position.set(26, 24, 36);

    // 灯光：暖主光 + 冷轮廓补光
    scene.add(new THREE.AmbientLight(0xffe9d2, 0.5));
    var key = new THREE.DirectionalLight(0xfff2df, 1.15);
    key.position.set(22, 44, 26);
    scene.add(key);
    var rim = new THREE.DirectionalLight(0x7aa2f7, 0.35);
    rim.position.set(-26, 16, -22);
    scene.add(rim);

    // 蓝图网格地面
    var grid = new THREE.GridHelper(90, 45, 0x3d525e, 0x27333b);
    grid.position.y = -0.02;
    grid.material.transparent = true;
    grid.material.opacity = 0.55;
    scene.add(grid);

    // 尺寸自适应
    function resize() {
      var w = canvasEl.clientWidth || window.innerWidth;
      var h = canvasEl.clientHeight || window.innerHeight;
      renderer.setSize(w, h, false);
      s.camera.aspect = w / h;
      s.camera.updateProjectionMatrix();
    }
    window.addEventListener('resize', resize);
    resize();

    // 扩展钩子（camera 轨道、拾取绑定等，与文件加载顺序无关）
    var hooks = E._initHooks || [];
    for (var h = 0; h < hooks.length; h++) hooks[h](canvasEl);

    // 主循环
    var clock = new THREE.Clock();
    function loop() {
      requestAnimationFrame(loop);
      var dt = Math.min(clock.getDelta(), 0.05);
      s.time += dt;
      for (var i = 0; i < E._tickers.length; i++) E._tickers[i](dt, s.time);
      // 推进特效动画
      for (var j = s.anims.length - 1; j >= 0; j--) {
        var a = s.anims[j];
        if (a.delay > 0) { a.delay -= dt; continue; }
        if (!a._started) { a._started = true; if (a.start) a.start(); }
        a.t += dt;
        var k = Math.min(a.t / a.dur, 1);
        if (a.update) a.update(k);
        if (k >= 1) { if (a.done) a.done(); s.anims.splice(j, 1); }
      }
      renderer.render(scene, s.camera);
    }
    loop();
  };

  // 供其它模块：注册每帧回调
  E._addTicker = function (fn) { E._tickers.push(fn); };

  // 组件查询（scene-builder 建塔时填充）
  E._componentCenter = function (type) {
    var list = E._state.components[type];
    return (list && list.length) ? list[0].center : null;
  };
  E._pulseComponent = function (type, strength) {
    var list = E._state.components[type] || [];
    for (var i = 0; i < list.length; i++) if (list[i].pulse) list[i].pulse(strength || 1);
  };
})();
