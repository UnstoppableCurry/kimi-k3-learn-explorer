/* tour.js — 导览播放机：按 acts 顺序执行
 * 每幕：switchModel → onActStart(i) → 相机飞行 → focusComponent 脉冲 → playEffect → onSubtitle
 * 暴露: ENGINE.tour = {play(acts, hooks), pause(), resume(), stop(), step()}
 * hooks = {onSubtitle(text), onActStart(i)}
 * 注：focusComponent 命中时，相机目标改用组件实际中心（cam.pos 仍用剧本机位），保证精确取景。
 */
window.ENGINE = window.ENGINE || {};
(function () {
  var E = window.ENGINE;

  var tour = {
    _acts: null, _hooks: null, _i: -1,
    _t: 0, _dur: 0, _pending: null,
    _playing: false, _paused: false
  };

  tour.play = function (acts, hooks) {
    this.stop();
    if (!acts || !acts.length) return;
    this._acts = acts;
    this._hooks = hooks || {};
    this._playing = true;
    this._paused = false;
    this._goto(0);
  };

  tour.pause = function () { if (this._playing) this._paused = true; };
  tour.resume = function () { this._paused = false; };

  tour.stop = function () {
    var h = this._hooks;
    this._playing = false;
    this._paused = false;
    this._acts = null;
    this._i = -1;
    this._pending = null;
    if (h && h.onSubtitle) h.onSubtitle('');
  };

  // 单步：未播则从头开始（停在第 0 幕），已播则跳到下一幕
  tour.step = function () {
    if (!this._acts) return;
    if (!this._playing) { this._playing = true; this._paused = true; this._goto(0); return; }
    if (this._i < this._acts.length - 1) this._goto(this._i + 1);
  };

  tour._goto = function (i) {
    var act = this._acts[i];
    if (!act) { this.stop(); return; }
    this._i = i;
    this._t = 0;
    this._dur = act.duration || 5;
    var hooks = this._hooks || {};
    var pending = [];

    // 1) 换模型（契约行为；外壳若经 onActStart 再换一次，幂等无害）
    if (act.switchModel && window.MODEL_SPECS && window.MODEL_SPECS[act.switchModel]) {
      E.buildScene(window.MODEL_SPECS[act.switchModel]);
    }
    if (hooks.onActStart) hooks.onActStart(i);

    // 2) 相机飞行：focusComponent 命中则目标锁到组件实际中心
    var target = act.cam && act.cam.target;
    var c = act.focusComponent ? E._componentCenter(act.focusComponent) : null;
    if (c) target = [c.x, c.y, c.z];
    if (act.cam && act.cam.pos && E.camera && E.camera.flyTo) {
      E.camera.flyTo(act.cam.pos, target || [0, 20, 0], Math.min(1.6, this._dur * 0.35));
    }

    // 3) 字幕尽早出；4) 组件脉冲与特效在落位后触发
    if (hooks.onSubtitle) pending.push({ at: 0.15, fn: function () { hooks.onSubtitle(act.subtitle || act.title || ''); } });
    if (act.focusComponent) pending.push({
      at: 0.8, fn: function () { E._pulseComponent(act.focusComponent, 1); }
    });
    if (act.effect) pending.push({
      at: 1.1, fn: function () { E.playEffect(act.effect); }
    });
    this._pending = pending;
  };

  // 计时驱动（pause 只停幕推进与待触发事件，不打断相机飞行）
  E._tickers = E._tickers || [];
  E._tickers.push(function (dt) {
    if (!tour._playing || tour._paused) return;
    tour._t += dt;
    var p = tour._pending;
    if (p) {
      for (var i = 0; i < p.length; i++) {
        if (!p[i]._fired && tour._t >= p[i].at) {
          p[i]._fired = true;
          try { p[i].fn(); } catch (e) { console.warn('[tour] act 事件失败', e); }
        }
      }
    }
    if (tour._t >= tour._dur) {
      if (tour._i < tour._acts.length - 1) tour._goto(tour._i + 1);
      else tour.stop(); // 自然播完
    }
  });

  E.tour = tour;
})();
