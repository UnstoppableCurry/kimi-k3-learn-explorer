/**
 * UI_PANEL — 右侧信息面板（极简，点击组件时才显示）
 * 接口：init / show / hide / update / dispose / screenshot
 *
 * show(data) 接受：
 * {
 *   title: 'Attention 权重',
 *   subtitle: '第 3 层 · 头 5',
 *   text: '一段说明文字',
 *   rows: [{ label: 'softmax 和', value: '1.000' }],
 *   bars: [{ label: '你', value: 0.62 }]   // value ∈ [0,1]，画成横条
 * }
 * 样式内联注入，模块自包含，不依赖外部 CSS。
 */
window.UI_PANEL = {
  _el: null,
  _body: null,
  _visible: false,

  init: function () {
    if (this._el) return this._el;
    this._injectStyle();

    var el = document.createElement('aside');
    el.id = 'ui-panel';
    el.setAttribute('aria-live', 'polite');

    var close = document.createElement('button');
    close.className = 'ui-panel-close';
    close.textContent = '×';
    close.title = '关闭';
    var self = this;
    close.addEventListener('click', function () { self.hide(); });

    var body = document.createElement('div');
    body.className = 'ui-panel-body';

    el.appendChild(close);
    el.appendChild(body);
    document.body.appendChild(el);

    this._el = el;
    this._body = body;
    return el;
  },

  show: function (data) {
    if (!this._el) this.init();
    data = data || {};
    var b = this._body;
    b.innerHTML = '';

    if (data.title) {
      var h = document.createElement('h2');
      h.textContent = data.title;
      b.appendChild(h);
    }
    if (data.subtitle) {
      var s = document.createElement('p');
      s.className = 'ui-panel-sub';
      s.textContent = data.subtitle;
      b.appendChild(s);
    }
    if (data.text) {
      var t = document.createElement('p');
      t.className = 'ui-panel-text';
      t.textContent = data.text;
      b.appendChild(t);
    }
    (data.rows || []).forEach(function (r) {
      var div = document.createElement('div');
      div.className = 'ui-panel-row';
      var l = document.createElement('span'); l.textContent = r.label;
      var v = document.createElement('span'); v.className = 'ui-panel-val'; v.textContent = r.value;
      div.appendChild(l); div.appendChild(v);
      b.appendChild(div);
    });
    (data.bars || []).forEach(function (bar) {
      var div = document.createElement('div');
      div.className = 'ui-panel-bar';
      var l = document.createElement('span'); l.textContent = bar.label;
      var track = document.createElement('div'); track.className = 'ui-panel-track';
      var fill = document.createElement('div'); fill.className = 'ui-panel-fill';
      var pct = Math.max(0, Math.min(1, bar.value));
      fill.style.width = (pct * 100).toFixed(1) + '%';
      if (bar.color) fill.style.background = bar.color;
      var v = document.createElement('span'); v.className = 'ui-panel-val';
      v.textContent = typeof bar.value === 'number' ? bar.value.toFixed(3) : bar.value;
      track.appendChild(fill);
      div.appendChild(l); div.appendChild(track); div.appendChild(v);
      b.appendChild(div);
    });

    this._el.classList.add('open');
    this._visible = true;
  },

  hide: function () {
    if (this._el) this._el.classList.remove('open');
    this._visible = false;
  },

  update: function () { /* 面板是静态 DOM，无需逐帧更新 */ },

  dispose: function () {
    if (this._el) this._el.remove();
    var st = document.getElementById('ui-panel-style');
    if (st) st.remove();
    this._el = null;
    this._body = null;
    this._visible = false;
  },

  screenshot: function () {
    return window.CORE_RENDERER ? window.CORE_RENDERER.screenshot() : null;
  },

  _injectStyle: function () {
    if (document.getElementById('ui-panel-style')) return;
    var css = [
      '#ui-panel{position:fixed;top:0;right:0;height:100%;width:300px;box-sizing:border-box;',
      'padding:56px 22px 22px;background:rgba(10,13,19,.82);backdrop-filter:blur(14px);',
      'border-left:1px solid rgba(255,255,255,.07);color:#dfe6f2;',
      'font:13px/1.6 -apple-system,"SF Pro Text","PingFang SC",sans-serif;',
      'transform:translateX(105%);transition:transform .38s cubic-bezier(.22,1,.36,1);z-index:40;}',
      '#ui-panel.open{transform:translateX(0)}',
      '#ui-panel h2{margin:0 0 2px;font-size:17px;font-weight:600;letter-spacing:.02em;color:#fff}',
      '.ui-panel-sub{margin:0 0 14px;font-size:11px;color:#7d8aa3;letter-spacing:.06em;text-transform:uppercase}',
      '.ui-panel-text{margin:0 0 16px;color:#aeb9cc}',
      '.ui-panel-row{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.06)}',
      '.ui-panel-val{font-family:"SF Mono",Menlo,monospace;color:#9fd0ff;font-variant-numeric:tabular-nums}',
      '.ui-panel-bar{display:grid;grid-template-columns:64px 1fr 52px;align-items:center;gap:8px;padding:5px 0}',
      '.ui-panel-bar .ui-panel-val{text-align:right;font-size:11px}',
      '.ui-panel-track{height:6px;border-radius:3px;background:rgba(255,255,255,.08);overflow:hidden}',
      '.ui-panel-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,#4f8dff,#7be0c3);transition:width .3s ease}',
      '.ui-panel-close{position:absolute;top:14px;right:14px;width:28px;height:28px;border:0;border-radius:50%;',
      'background:rgba(255,255,255,.08);color:#cfd8e6;font-size:15px;cursor:pointer;line-height:1}',
      '.ui-panel-close:hover{background:rgba(255,255,255,.16)}'
    ].join('\n');
    var style = document.createElement('style');
    style.id = 'ui-panel-style';
    style.textContent = css;
    document.head.appendChild(style);
  }
};
