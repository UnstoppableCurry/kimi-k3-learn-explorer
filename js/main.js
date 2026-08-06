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
    function setPlaying(v) {
      if (v) { PIPELINE.play(); btnPlay.textContent = '⏸ 暂停'; }
      else { PIPELINE.pause(); btnPlay.textContent = '▶ 播放'; }
    }
    btnPlay.addEventListener('click', function () { setPlaying(!PIPELINE.isPlaying()); });
    document.getElementById('btnRestart').addEventListener('click', function () {
      PIPELINE.restart();
      setPlaying(true);
    });
    var chk = document.getElementById('chkFollow');
    chk.addEventListener('change', function () { PIPELINE.setFollow(chk.checked); });

    // 移动速度滑块（0.5 - 10 单位/秒，默认 2）
    var speedRange = document.getElementById('speedRange');
    var speedVal = document.getElementById('speedVal');
    speedRange.addEventListener('input', function () {
      var v = parseFloat(speedRange.value);
      CORE_CAMERA.setMoveSpeed(v);
      speedVal.textContent = v.toFixed(1);
    });

    // 阶段导航条：可点击跳转，当前阶段高亮
    var nav = document.getElementById('stageNav');
    var chips = PIPELINE.stageNames().map(function (name, i) {
      var c = document.createElement('span');
      c.className = 'chip';
      c.textContent = name;
      c.addEventListener('click', function () {
        setPlaying(false); // 手动跳转后暂停，让用户慢慢看
        PIPELINE.goTo(i);
      });
      nav.appendChild(c);
      return c;
    });
    PIPELINE.setStageListener(function (idx) {
      chips.forEach(function (c, i) {
        c.classList.toggle('active', i === idx);
        c.classList.toggle('done', i < idx);
      });
    });
    // init 时已 enter 过 stage 0，手动补一次高亮
    chips.forEach(function (c, i) { c.classList.toggle('active', i === PIPELINE.currentStage()); });

    // 手动步进：上一步 / 下一步（点击后暂停自动播放）
    document.getElementById('btnNext').addEventListener('click', function () {
      setPlaying(false);
      PIPELINE.next();
    });
    document.getElementById('btnPrev').addEventListener('click', function () {
      setPlaying(false);
      PIPELINE.prev();
    });

    // 用户拖拽相机时，短暂让出跟随权
    canvas.addEventListener('pointerdown', function () { PIPELINE.notifyUserDrag(); });
    canvas.addEventListener('wheel', function () { PIPELINE.notifyUserDrag(); }, { passive: true });

    // 键盘：空格播放/暂停，R 重播，→/← 步进
    window.addEventListener('keydown', function (e) {
      if (e.code === 'Space') { e.preventDefault(); btnPlay.click(); }
      if (e.code === 'KeyR') document.getElementById('btnRestart').click();
      if (e.code === 'ArrowRight') document.getElementById('btnNext').click();
      if (e.code === 'ArrowLeft') document.getElementById('btnPrev').click();
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
