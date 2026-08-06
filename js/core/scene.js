/**
 * CORE_SCENE — 场景管理（光照、阴影、背景）
 * 接口：init / show / hide / update / dispose / screenshot（screenshot 委托给 CORE_RENDERER）
 *
 * 设计：三点布光（主光带阴影 + 补光 + 轮廓光）+ 半球环境光 + 深色雾背景 + 接影地面。
 * 不依赖其他模块，返回的 scene 交给 CORE_RENDERER 渲染。
 */
window.CORE_SCENE = {
  scene: null,
  _disposables: [],

  init: function () {
    var scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0d13);
    scene.fog = new THREE.Fog(0x0a0d13, 55, 140);

    // 半球环境光：天空微蓝 / 地面微暖，给暗部一点层次
    var hemi = new THREE.HemisphereLight(0x8ea6c8, 0x1a1410, 0.55);
    scene.add(hemi);

    // 主光（key）：暖白，右上前方，唯一投影光源
    var key = new THREE.DirectionalLight(0xfff2e0, 2.2);
    key.position.set(14, 22, 10);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 1;
    key.shadow.camera.far = 80;
    key.shadow.camera.left = -25;
    key.shadow.camera.right = 25;
    key.shadow.camera.top = 25;
    key.shadow.camera.bottom = -25;
    key.shadow.bias = -0.0004;
    key.shadow.radius = 4;
    scene.add(key);

    // 补光（fill）：冷色，左侧，压暗部对比
    var fill = new THREE.DirectionalLight(0x6f8fc0, 0.5);
    fill.position.set(-16, 8, 6);
    scene.add(fill);

    // 轮廓光（rim）：背后上方，把物体从背景里勾出来
    var rim = new THREE.DirectionalLight(0x9fc0ff, 0.9);
    rim.position.set(-4, 14, -18);
    scene.add(rim);

    // 地面：大圆盘，只负责接阴影，近乎黑色
    var groundGeo = new THREE.CircleGeometry(90, 64);
    var groundMat = new THREE.MeshStandardMaterial({
      color: 0x11151d, roughness: 0.95, metalness: 0.0
    });
    var ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    scene.add(ground);
    this._disposables.push(groundGeo, groundMat);

    // 极淡的参考网格，帮助读出空间尺度
    var grid = new THREE.GridHelper(60, 30, 0x233046, 0x161d2b);
    grid.position.y = 0;
    grid.material.transparent = true;
    grid.material.opacity = 0.35;
    scene.add(grid);
    this._disposables.push(grid.geometry, grid.material);

    this.scene = scene;
    return scene;
  },

  show: function () { if (this.scene) this.scene.visible = true; },
  hide: function () { if (this.scene) this.scene.visible = false; },

  update: function () { /* 静态光照，无需逐帧更新 */ },

  dispose: function () {
    this._disposables.forEach(function (d) { if (d && d.dispose) d.dispose(); });
    this._disposables = [];
    this.scene = null;
  },

  screenshot: function () {
    return window.CORE_RENDERER ? window.CORE_RENDERER.screenshot() : null;
  }
};
