/* main.js — 主控制器：加载所有模块，协调显示 */
(function () {
  'use strict';

  var scene, camera, renderer;
  var modules = {};
  var currentModule = null;
  var playing = false;
  var animationId = null;

  // 初始化
  function init() {
    // 初始化 core
    scene = CORE_SCENE.init();
    camera = CORE_CAMERA.init({ domElement: document.getElementById('scene') });
    renderer = CORE_RENDERER.init({
      canvas: document.getElementById('scene'),
      scene: scene,
      camera: camera
    });

    // 初始化模块
    modules.token = window.TokenModule;
    modules.attention = window.ATTENTION_MODULE;
    modules.ffn = window.FFN;
    modules.moe = window.MOE;
    modules.output = window.OUTPUT_MODULE;
    modules.training = window.MODULE_TRAINING;

    // 初始化所有模块
    Object.keys(modules).forEach(function (key) {
      if (modules[key] && typeof modules[key].init === 'function') {
        modules[key].init(scene, camera, renderer);
      }
    });

    // 初始化 UI
    UI_PANEL.init();
    UI_CONTROLS.init({
      onPlay: onPlay,
      onPause: onPause,
      onStep: onStep,
      onReset: onReset
    });

    // 默认显示 Token 模块
    showModule('token');

    // 开始渲染循环
    animate();
  }

  // 显示模块
  function showModule(name) {
    if (currentModule) {
      modules[currentModule].hide();
    }
    currentModule = name;
    modules[name].show();
  }

  // 播放/暂停
  function onPlay() {
    playing = true;
  }

  function onPause() {
    playing = false;
  }

  // 单步
  function onStep() {
    // 切换到下一个模块
    var moduleNames = ['token', 'attention', 'ffn', 'moe', 'output', 'training'];
    var currentIndex = moduleNames.indexOf(currentModule);
    var nextIndex = (currentIndex + 1) % moduleNames.length;
    showModule(moduleNames[nextIndex]);
  }

  // 重置
  function onReset() {
    CORE_CAMERA.reset();
    if (currentModule) {
      showModule(currentModule);
    }
  }

  // 动画循环
  var lastTime = performance.now();
  function animate() {
    animationId = requestAnimationFrame(animate);

    var now = performance.now();
    var delta = (now - lastTime) / 1000;
    lastTime = now;

    // 更新当前模块
    if (currentModule && modules[currentModule]) {
      modules[currentModule].update(delta);
    }

    // 更新相机
    CORE_CAMERA.update(delta);

    // 渲染
    CORE_RENDERER.update();
  }

  // 启动
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
