/* main.js — 装配：core（场景/相机/渲染）+ PIPELINE（整体流水线）
 * 不再是模块切换：整个网络一次建出来，token 从输入流到输出，相机自动跟随。
 */
(function () {
  'use strict';

  var scene, camera, renderer;

  function init() {
    var canvas = document.getElementById('scene');
    scene = CORE_SCENE.init();
    camera = CORE_CAMERA.init({ domElement: canvas });
    renderer = CORE_RENDERER.init({ canvas: canvas, scene: scene, camera: camera });

    PIPELINE.init(scene);

    // 控件
    var btnPlay = document.getElementById('btnPlay');
    btnPlay.addEventListener('click', function () {
      if (PIPELINE.isPlaying()) { PIPELINE.pause(); btnPlay.textContent = '▶ 播放'; }
      else { PIPELINE.play(); btnPlay.textContent = '⏸ 暂停'; }
    });
    document.getElementById('btnRestart').addEventListener('click', function () {
      PIPELINE.restart();
      PIPELINE.play();
      btnPlay.textContent = '⏸ 暂停';
    });
    var chk = document.getElementById('chkFollow');
    chk.addEventListener('change', function () { PIPELINE.setFollow(chk.checked); });

    // 用户拖拽相机时，短暂让出跟随权
    canvas.addEventListener('pointerdown', function () { PIPELINE.notifyUserDrag(); });
    canvas.addEventListener('wheel', function () { PIPELINE.notifyUserDrag(); }, { passive: true });

    // 键盘：空格播放/暂停，R 重播
    window.addEventListener('keydown', function (e) {
      if (e.code === 'Space') { e.preventDefault(); btnPlay.click(); }
      if (e.code === 'KeyR') document.getElementById('btnRestart').click();
    });

    animate();
    window.__ready = true;
  }

  var lastTime = performance.now();
  function animate() {
    requestAnimationFrame(animate);
    var now = performance.now();
    var delta = Math.min(0.1, (now - lastTime) / 1000);
    lastTime = now;
    PIPELINE.update(delta);
    CORE_CAMERA.update(delta);
    CORE_RENDERER.update();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
