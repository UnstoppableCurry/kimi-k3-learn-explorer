/**
 * UI_CONTROLS — 控制按钮（播放 / 暂停 / 单步 / 重置）
 * 接口：init / show / hide / update / dispose / screenshot
 *
 * init(opts)：
 * {
 *   onPlay: fn, onPause: fn, onStep: fn, onReset: fn  // 全部可选
 * }
 * 状态机：paused 为 true 时播放按钮显示 ▶，false 时显示 ⏸（播放/暂停同一个按钮）。
 * 样式内联注入，模块自包含。
 */
window.UI_CONTROLS = {
  _el: null,
  _playBtn: null,
  paused: true,
  _handlers: null,

  init: function (opts) {
    opts = opts || {};
    this._handlers = opts;
    if (this._el) return this._el;
    this._injectStyle();

    var el = document.createElement('div');
    el.id = 'ui-controls';
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', '播放控制');

    var self = this;
    this._playBtn = this._btn('▶', '播放 / 暂停', function () {
      self.setPaused(!self.paused);
    });
    var stepBtn = this._btn('⏭', '单步', function () {
      if (opts.onStep) opts.onStep();
    });
    var resetBtn = this._btn('⟲', '重置', function () {
      self.setPaused(true);
      if (opts.onReset) opts.onReset();
    });

    el.appendChild(this._playBtn);
    el.appendChild(stepBtn);
    el.appendChild(resetBtn);
    document.body.appendChild(el);

    this._el = el;
    this.setPaused(true);
    return el;
  },

  /** 切播放/暂停状态并触发回调 */
  setPaused: function (paused) {
    this.paused = paused;
    if (this._playBtn) this._playBtn.textContent = paused ? '▶' : '⏸';
    var h = this._handlers || {};
    if (paused && h.onPause) h.onPause();
    if (!paused && h.onPlay) h.onPlay();
  },

  _btn: function (label, title, onclick) {
    var b = document.createElement('button');
    b.className = 'ui-ctl-btn';
    b.textContent = label;
    b.title = title;
    b.setAttribute('aria-label', title);
    b.addEventListener('click', onclick);
    return b;
  },

  show: function () { if (this._el) this._el.style.display = ''; },
  hide: function () { if (this._el) this._el.style.display = 'none'; },

  update: function () { /* 静态 DOM，无需逐帧更新 */ },

  dispose: function () {
    if (this._el) this._el.remove();
    var st = document.getElementById('ui-controls-style');
    if (st) st.remove();
    this._el = null;
    this._playBtn = null;
    this._handlers = null;
  },

  screenshot: function () {
    return window.CORE_RENDERER ? window.CORE_RENDERER.screenshot() : null;
  },

  _injectStyle: function () {
    if (document.getElementById('ui-controls-style')) return;
    var css = [
      '#ui-controls{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);',
      'display:flex;gap:10px;padding:8px 10px;border-radius:999px;',
      'background:rgba(10,13,19,.72);backdrop-filter:blur(12px);',
      'border:1px solid rgba(255,255,255,.08);z-index:40}',
      '.ui-ctl-btn{width:42px;height:42px;border:0;border-radius:50%;cursor:pointer;',
      'background:rgba(255,255,255,.07);color:#e6ecf6;font-size:16px;line-height:1;',
      'transition:background .18s ease,transform .12s ease}',
      '.ui-ctl-btn:hover{background:rgba(255,255,255,.16)}',
      '.ui-ctl-btn:active{transform:scale(.92)}'
    ].join('\n');
    var style = document.createElement('style');
    style.id = 'ui-controls-style';
    style.textContent = css;
    document.head.appendChild(style);
  }
};
