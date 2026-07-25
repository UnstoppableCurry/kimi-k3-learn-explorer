/* picking.js — Raycaster 拾取：点击组件回调 userData.componentId
 * 暴露: ENGINE.onPick(cb)；点击命中后附带组件脉冲反馈。
 */
window.ENGINE = window.ENGINE || {};
(function () {
  var E = window.ENGINE;

  E.onPick = function (cb) { if (typeof cb === 'function') E._pickCbs.push(cb); };

  function findComponentId(obj) {
    var o = obj;
    while (o) {
      if (o.userData && o.userData.componentId) return o.userData.componentId;
      o = o.parent;
    }
    return null;
  }

  function raycastAt(clientX, clientY) {
    var s = E._state;
    if (!s.camera || !s.pickables.length) return null;
    var rect = s.canvas.getBoundingClientRect();
    var ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    var ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, s.camera);
    var hits = ray.intersectObjects(s.pickables, false);
    for (var i = 0; i < hits.length; i++) {
      var id = findComponentId(hits[i].object);
      if (id) return { id: id, point: hits[i].point, instanceId: hits[i].instanceId, object: hits[i].object };
    }
    return null;
  }

  E._initHooks = E._initHooks || [];
  E._initHooks.push(function (canvasEl) {
    if (canvasEl._pickBound) return;
    canvasEl._pickBound = true;

    var downX = 0, downY = 0, downT = 0;
    canvasEl.addEventListener('pointerdown', function (e) {
      downX = e.clientX; downY = e.clientY; downT = performance.now();
    });
    canvasEl.addEventListener('pointerup', function (e) {
      // 位移小、时长短才算点击（区别于拖拽旋转）
      var dx = e.clientX - downX, dy = e.clientY - downY;
      if (dx * dx + dy * dy > 36 || performance.now() - downT > 500) return;
      var hit = raycastAt(e.clientX, e.clientY);
      if (!hit) return;
      E._pulseComponent(hit.id, 1); // 点击反馈
      for (var i = 0; i < E._pickCbs.length; i++) E._pickCbs[i](hit.id, hit);
    });

    // 悬停手型（节流）
    var last = 0;
    canvasEl.addEventListener('pointermove', function (e) {
      var now = performance.now();
      if (now - last < 90) return;
      last = now;
      canvasEl.style.cursor = raycastAt(e.clientX, e.clientY) ? 'pointer' : 'grab';
    });
  });
})();
