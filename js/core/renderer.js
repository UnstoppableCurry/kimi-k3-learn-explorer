/**
 * CORE_RENDERER — 渲染器（WebGL、阴影、色调映射、自适应尺寸、截图）
 * 接口：init / show / hide / update / dispose / screenshot
 *
 * - antialias + PCFSoftShadowMap + ACESFilmicToneMapping
 * - pixelRatio 封顶 2，避免高分屏掉帧
 * - 监听 resize，自动同步 camera aspect
 * - update(delta) 即渲染一帧（渲染循环由调用方驱动）
 * - screenshot() 先强制渲染再取 dataURL，无需 preserveDrawingBuffer
 */
window.CORE_RENDERER = {
  renderer: null,
  scene: null,
  camera: null,
  _onResize: null,

  /**
   * opts: { canvas?: HTMLCanvasElement, container?: HTMLElement,
   *         scene?: THREE.Scene, camera?: THREE.Camera }
   */
  init: function (opts) {
    opts = opts || {};
    var canvas = opts.canvas;
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.id = 'core-canvas';
      (opts.container || document.body).appendChild(canvas);
    }

    var renderer = new THREE.WebGLRenderer({
      canvas: canvas, antialias: true, alpha: false
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.renderer = renderer;
    this.scene = opts.scene || null;
    this.camera = opts.camera || null;

    var self = this;
    this._onResize = function () { self._resize(); };
    window.addEventListener('resize', this._onResize);
    this._resize();

    return renderer;
  },

  /** 绑定要渲染的场景与相机（main.js 装配用） */
  attach: function (scene, camera) {
    this.scene = scene;
    this.camera = camera;
  },

  _resize: function () {
    if (!this.renderer) return;
    var canvas = this.renderer.domElement;
    var w = canvas.clientWidth || window.innerWidth;
    var h = canvas.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    if (this.camera && this.camera.isPerspectiveCamera) {
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
  },

  /** 每帧调用：渲染一帧 */
  update: function () {
    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  },

  show: function () {
    if (this.renderer) this.renderer.domElement.style.display = '';
  },
  hide: function () {
    if (this.renderer) this.renderer.domElement.style.display = 'none';
  },

  dispose: function () {
    if (this._onResize) window.removeEventListener('resize', this._onResize);
    if (this.renderer) {
      this.renderer.dispose();
      if (this.renderer.domElement && this.renderer.domElement.id === 'core-canvas') {
        this.renderer.domElement.remove();
      }
    }
    this.renderer = null;
    this.scene = null;
    this.camera = null;
  },

  /** 截图：先强制渲染当前帧再读像素，返回 PNG dataURL */
  screenshot: function () {
    if (!this.renderer) return null;
    if (this.scene && this.camera) this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL('image/png');
  }
};
