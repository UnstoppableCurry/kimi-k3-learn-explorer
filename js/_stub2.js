/* 集成自测桩：仅桩 ENGINE（该部分未交付），models/content 用真实文件。测试后删除。 */
window.ENGINE = {
  init: function (canvas) {
    var ctx = canvas.getContext('2d');
    function draw() {
      canvas.width = innerWidth; canvas.height = innerHeight;
      ctx.fillStyle = '#171310'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      var comps = [['kda','#e8a94d',300,300],['experts','#46c8ae',450,300],['attnres','#e0699b',600,300]];
      comps.forEach(function (c) {
        ctx.fillStyle = c[1]; ctx.fillRect(c[2], c[3], 90, 90);
      });
      ctx.fillStyle = '#a89a83'; ctx.font = '13px -apple-system';
      ctx.fillText('[ENGINE 桩] 点击色块模拟 onPick', 300, 260);
      canvas._comps = comps;
    }
    draw(); addEventListener('resize', draw);
    var self = this;
    canvas.addEventListener('click', function (e) {
      (canvas._comps||[]).forEach(function (c) {
        if (e.clientX>=c[2]&&e.clientX<=c[2]+90&&e.clientY>=c[3]&&e.clientY<=c[3]+90) {
          if (self._pickCb) self._pickCb(c[0]);
        }
      });
    });
  },
  buildScene: function (spec) { console.log('[stub] buildScene', spec.id, 'layers=' + spec.layers.length, 'blocks=' + spec.blockCount); },
  onPick: function (cb) { this._pickCb = cb; },
  playEffect: function (n) { console.log('[stub] playEffect', n); },
  tour: {
    play: function (acts, hooks) {
      var i = 0, self = this;
      this._t = setInterval(function () {
        if (i >= acts.length) { self.stop(); return; }
        hooks.onActStart(i); hooks.onSubtitle(acts[i].subtitle); i++;
      }, 2000);
    },
    pause: function () { clearInterval(this._t); },
    resume: function () {},
    stop: function () { clearInterval(this._t); },
    step: function () {}
  },
  camera: { focusOn: function () {}, reset: function () {} },
  flow: { start: function () {}, stop: function () {} }
};
setTimeout(function () {
  document.getElementById('scene').dispatchEvent(new MouseEvent('click', { clientX: 345, clientY: 345 })); // kda
}, 800);
setTimeout(function () {
  document.getElementById('scene').dispatchEvent(new MouseEvent('click', { clientX: 645, clientY: 345 })); // attnres
}, 2600);
setTimeout(function () {
  document.getElementById('btnTourPlay').click();   // 真实 CONTENT_TOUR
}, 4400);
