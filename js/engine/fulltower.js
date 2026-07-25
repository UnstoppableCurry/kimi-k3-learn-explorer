/* fulltower.js — 全塔模式："模型城市"布局
 * 暴露: window.FullTower = { build(spec), dispose(), isActive() }
 *
 * 设计语义（用户裁决）：MoE 不是一栋摩天楼，而是一群摩天楼。
 *   - 60 个 Block = 60 个街区（10×6 蛇形排布），街区之间以茜红光路相连 = 深度流（AttnRes）
 *   - 每个街区：中央低层裙楼 = 注意力栈（norm/kda/mla 薄板）+ router 枢纽（亮薄荷八面体）
 *   - 896 个专家 = 896 栋独立高楼（宽基座 up → 收腰激活 → 宽塔冠 down），楼高各异读出天际线
 *   - 稀疏激活：16/896 栋楼点亮成亮薄荷，其余压暗成休眠海；共享专家 = 蓝色常亮高楼（街区边缘）
 *   - token/embedding 是城市入口，softmax/output 是出口
 * 性能：3 个 53,760 实例的 InstancedMesh（楼体三段）+ 若干 60 实例网格，~20 draw calls。
 * 拾取：每街区一个隐形代理盒（material.visible=false），点中 = experts，O(1)。
 */
window.FullTower = (function () {
  var E = window.ENGINE;
  function S() { return E._state; }
  function COL() { return E.COLORS || { kda: 0xe8a94d, mla: 0x7aa2f7, moe: 0x46c8ae, attnres: 0xe0699b, generic: 0xa89a83 }; }

  // ---------- 城市布局常量 ----------
  var EXP_W = 64, EXP_D = 44;            // 街区占地（专家楼群范围，留足街道）
  var PX = EXP_W + 18, PZ = EXP_D + 18;  // 街区间距（含街道）
  var COLS_B = 10;                       // 每行 10 个街区，蛇形
  var SLAB_W = 16, SLAB_D = 10.5;        // 注意力裙楼截面
  var MARGIN = 0.10;
  var TOKEN_N = 8;
  var SLOT = { norm: 0.45, kda: 0.85, mla: 0.85, router: 0.65, attnres: 0.85, _unknown: 0.65 };
  var GLOBAL_TYPES = { tokenizer: 1, embedding: 1, output: 1, softmax: 1 };
  var EXP_COLS = 34, EXP_ROWS_G = 30;    // 候选格 34×30，跳过裙楼 footprint 后取前 896
  // 摩天楼分段（FFN 形状语言：宽-窄-宽）
  var UP_H = 10.0, ACT_H = 7.5, DN_H = 10.0, GAP_S = 0.5;   // 楼高夸张化：远景也读得出垂直线条
  var FULL_FOG = 0.0009;   // 前排清晰、纵深渐隐
  var savedFogDensity = null;
  var savedFar = null;   // 原始 far 只捕获一次（重建沿用，避免快照链丢失原值）
  var fullHud = null;    // 全景塔简化数值 HUD

  function slotH(t) { return SLOT[t] || SLOT._unknown; }
  function fmt(v, n) { return (v >= 0 ? ' ' : '') + v.toFixed(n == null ? 2 : n); }

  // 与 scene-builder.reg 同语义（其内部函数不可达，这里按既有做法复刻）
  function reg(type, group, meshes, center, radius, pulse) {
    var s = S();
    for (var i = 0; i < meshes.length; i++) {
      meshes[i].userData.componentId = type;
      s.pickables.push(meshes[i]);
    }
    if (!s.components[type]) s.components[type] = [];
    s.components[type].push({ group: group, center: center, radius: radius, pulse: pulse || null });
    return s.components[type][s.components[type].length - 1];
  }

  function disposeObj(root) {
    root.traverse(function (o) {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        var mats = Array.isArray(o.material) ? o.material : [o.material];
        for (var i = 0; i < mats.length; i++) {
          if (mats[i].map) mats[i].map.dispose();
          mats[i].dispose();
        }
      }
    });
  }

  function typeColor(type) {
    var c = COL();
    if (type === 'kda') return c.kda;
    if (type === 'mla') return c.mla;
    if (type === 'router') return 0xa8f5e0;   // 路由=独立阶段，亮薄荷色与楼群分离
    if (type === 'attnres') return c.attnres;
    return c.generic;
  }

  // ---------- 全景塔数值 HUD：城市尺度下的关键数字 ----------
  function hudTexture() {
    var SIM = window.SIM;
    var c = document.createElement('canvas');
    c.width = 640; c.height = 420;
    var g = c.getContext('2d');
    g.fillStyle = 'rgba(23,19,16,0.88)'; g.fillRect(0, 0, c.width, c.height);
    g.strokeStyle = '#e8a94d'; g.lineWidth = 4; g.strokeRect(2, 2, c.width - 4, c.height - 4);
    g.font = 'bold 26px Menlo, monospace';
    g.textBaseline = 'top';
    g.fillStyle = '#efe6d8'; g.fillText('全景数值流 · MoE FFN 层', 18, 18);
    g.font = '18px Menlo, monospace';
    g.fillStyle = '#46c8ae'; g.fillText('896 个 FFN 专家，每 token 激活 16 + 共享', 18, 48);

    if (!SIM || !SIM.data || !SIM.data.length) {
      g.fillStyle = '#a89a83'; g.fillText('等待输入…', 18, 60);
      return new THREE.CanvasTexture(c);
    }
    var cur = SIM.data[SIM.data.length - 1];
    var y = 56;
    function line(label, value, color) {
      g.font = '22px Menlo, monospace';
      g.fillStyle = '#6f6350'; g.fillText(label, 18, y);
      g.font = 'bold 22px Menlo, monospace';
      g.fillStyle = color || '#c9bba6'; g.fillText(value, 220, y);
      y += 34;
    }
    line('当前 token', '「' + cur.ch + '」 id=' + cur.id, '#e8a94d');
    line('KDA 状态迹', fmt(cur.S.reduce(function(a,b){return a+b;},0), 2), '#e8a94d');
    line('注意力峰值', fmt(Math.max.apply(null, cur.attn), 3), '#7aa2f7');
    line('Top-1 专家', '#' + cur.top[0] + '  w=' + fmt(cur.w[0], 3), '#46c8ae');

    // Top-8 专家权重条
    y += 8;
    g.fillStyle = '#6f6350'; g.font = '18px Menlo, monospace'; g.fillText('Top-8 专家权重：', 18, y); y += 28;
    for (var i = 0; i < Math.min(8, cur.top.length); i++) {
      var barW = Math.max(2, cur.w[i] * 280);
      g.fillStyle = 'rgba(70,200,174,0.25)'; g.fillRect(18, y + 4, 280, 20);
      g.fillStyle = '#46c8ae'; g.fillRect(18, y + 4, barW, 20);
      g.fillStyle = '#efe6d8'; g.font = '16px Menlo, monospace';
      g.fillText('#' + cur.top[i] + ' ' + fmt(cur.w[i], 3), 310, y + 5);
      y += 26;
    }

    // 输出概率 Top-4
    var sp = (E._state._sampler) || { temperature: 1, topP: 1 };
    var p = SIM.probs(sp.temperature, sp.topP);
    var order = p.map(function(v,i){return i;}).sort(function(a,b){return p[b]-p[a];});
    y += 6;
    g.fillStyle = '#6f6350'; g.font = '18px Menlo, monospace'; g.fillText('输出采样 T=' + fmt(sp.temperature,2) + ' p=' + fmt(sp.topP,2), 18, y); y += 28;
    for (var j = 0; j < 4; j++) {
      var idx = order[j];
      g.fillStyle = j === 0 ? '#e8a94d' : '#a89a83'; g.font = 'bold 18px Menlo, monospace';
      g.fillText('「' + SIM.outCh[idx] + '」 ' + (p[idx]*100).toFixed(1) + '%', 18, y);
      y += 24;
    }

    var tex = new THREE.CanvasTexture(c);
    if (THREE.SRGBColorSpace) tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  function buildHud() {
    var s = S();
    if (!s || !s.scene) return;
    var tex = hudTexture();
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    sp.scale.set(100, 65, 1);
    // 放在城市中心上方偏前，航拍机位下足够大且可读
    sp.position.set(0, 70, 120);
    s.scene.add(sp);
    fullHud = { sprite: sp, update: function() {
      var t = hudTexture();
      sp.material.map.dispose();
      sp.material.map = t;
      sp.material.needsUpdate = true;
    }};
  }

  function updateHud() {
    if (fullHud && fullHud.update) fullHud.update();
  }

  function disposeHud() {
    var s = S();
    if (!fullHud) return;
    if (fullHud.sprite.parent) fullHud.sprite.parent.remove(fullHud.sprite);
    if (fullHud.sprite.material.map) fullHud.sprite.material.map.dispose();
    fullHud.sprite.material.dispose();
    fullHud = null;
  }

  // 街区蛇形坐标：b=0 在西北角，奇数行反向，终点在西南角
  function plazaPos(b) {
    var row = Math.floor(b / COLS_B);
    var colRaw = b % COLS_B;
    var col = (row % 2 === 1) ? (COLS_B - 1 - colRaw) : colRaw;
    var rows = Math.ceil(60 / COLS_B);
    return {
      x: (col - (COLS_B - 1) / 2) * PX,
      z: (row - (rows - 1) / 2) * PZ,
      row: row, col: col
    };
  }

  function build(spec, keepEnv) {
    var s = S();
    if (!s || !s.scene) { console.warn('[fulltower] 请先调用 ENGINE.init(canvas)'); return; }
    if (s._full) dispose(); // 重复 build 先回滚
    hookBuildScene();
    spec = spec || s.spec || {};

    var N = Math.max(1, (spec.blockCount | 0) || 60);

    // ---------- 快照 detail 现场 ----------
    var prev = {
      pickables: s.pickables.slice(),
      components: {},
      home: s.home,
      tower: s.tower,
      towerH: s.towerH,
      mode: s._mode || 'detail'
    };
    for (var k in s.components) prev.components[k] = s.components[k].slice();
    if (s.tower) s.tower.visible = false;   // 隐藏 detail 塔，不销毁（dispose 原样还原）
    s.pickables = [];
    s.components = {};

    // numbers 面板兼容①：细节模式的数值牌是直接挂在 s.scene 上的 Sprite，不随 tower 隐藏。
    // 此刻 components 已清空 —— 调一次 refreshNumbers：clearPanels 清掉细节残留，
    // attach 端因 comps 缺失全部静默跳过（numbers.js 既有语义，无需改它）。
    hookRefreshNumbers();
    if (typeof E.refreshNumbers === 'function') { try { E.refreshNumbers(); } catch (e) {} }

    if (s.scene.fog) {
      if (!keepEnv || savedFogDensity == null) savedFogDensity = s.scene.fog.density;
      s.scene.fog.density = FULL_FOG;
    }

    var city = new THREE.Group();
    var full = { group: city, prev: prev };
    s._full = full;

    // ---------- block 内裙楼层序列（experts 单独建楼群） ----------
    var blockLayers = [], globals = {};
    (spec.layers || []).forEach(function (l) {
      if (!l || !l.type) return;
      if (GLOBAL_TYPES[l.type]) globals[l.type] = true;
      else if (l.type !== 'experts' && l.type !== 'attnres') blockLayers.push(l.type);
    });
    if (!blockLayers.length) blockLayers = ['norm', 'kda', 'norm', 'mla', 'router'];

    var podiumH = 0.45;                    // 地块板厚度
    var slabY = podiumH;
    blockLayers.forEach(function (t) { slabY += slotH(t) + MARGIN; });
    podiumH = slabY;                       // 裙楼总高（含 router）

    var dummy = new THREE.Object3D();
    var moeC = new THREE.Color(COL().moe);
    var mintC = new THREE.Color(0xa8f5e0);
    var sharedC = new THREE.Color(0x6f9ff0);

    // ---------- 裙楼薄板：每个 type 一个 InstancedMesh（N 实例，非光照满色） ----------
    var seen = {};
    blockLayers.forEach(function (t) {
      if (seen[t]) return;
      seen[t] = 1;
      var h = slotH(t);
      var off = 0.45;
      for (var i = 0; i < blockLayers.length; i++) {
        if (blockLayers[i] === t) break;
        off += slotH(blockLayers[i]) + MARGIN;
      }
      var mat = new THREE.MeshBasicMaterial({ color: typeColor(t) });
      var mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(SLAB_W, h, SLAB_D), mat, N);
      for (var b = 0; b < N; b++) {
        var p = plazaPos(b);
        dummy.position.set(p.x, off + h / 2, p.z);
        dummy.rotation.set(0, 0, 0); dummy.scale.set(1, 1, 1);
        dummy.updateMatrix();
        mesh.setMatrixAt(b, dummy.matrix);
      }
      mesh.instanceMatrix.needsUpdate = true;
      city.add(mesh);
      reg(t, city, [mesh], new THREE.Vector3(0, 3, 0), 900, null);
    });

    // ---------- 地面：不铺地皮（俯视时大黑板丑），楼群直接立在夜色里 ----------

    // ---------- 896 栋专家高楼：三段（up 基座 / 激活腰 / down 塔冠）× 60 街区 ----------
    var moe = spec.moe || { experts: 896, active: 16, shared: 1 };
    var perBlock = Math.max(1, moe.experts | 0);
    var activeK = Math.max(1, (moe.active | 0) || 16);
    var total = perBlock * N;
    var gx = EXP_W / EXP_COLS, gz = EXP_D / EXP_ROWS_G;
    var fp = Math.min(gx, gz) * 0.45;      // 楼的截面：纤细，楼间留街
    var POD_TOP = 0.35;                    // 楼从地皮上起
    var bodyC = new THREE.Color(0x51759b); // 休眠楼体：蓝灰（夜景但可读）
    var warmC = new THREE.Color(0xffc46b); // 休眠楼腰：暖黄窗光

    // 候选格：跳过中央裙楼 footprint，取前 perBlock 个
    var cells = [];
    for (var cy = 0; cy < EXP_ROWS_G && cells.length < perBlock; cy++) {
      for (var cxI = 0; cxI < EXP_COLS && cells.length < perBlock; cxI++) {
        var ccx = (cxI - (EXP_COLS - 1) / 2) * gx;
        var ccz = (cy - (EXP_ROWS_G - 1) / 2) * gz;
        if (Math.abs(ccx) < SLAB_W / 2 + 1.5 && Math.abs(ccz) < SLAB_D / 2 + 1.5) continue; // 裙楼占位
        cells.push([ccx, ccz]);
      }
    }
    while (cells.length < perBlock) cells.push([0, EXP_D / 2 + 2]); // 兜底（理论不到这）

    // 楼体外墙纹理：Canvas 画的窗格（亮窗随机暖黄/冷蓝），一个纹理全体复用，零额外 draw call
    function facadeTexture() {
      var cv = document.createElement('canvas'); cv.width = 64; cv.height = 256;
      var g = cv.getContext('2d');
      g.fillStyle = '#263646'; g.fillRect(0, 0, 64, 256);
      for (var wy = 5; wy < 250; wy += 9) {
        for (var wx = 5; wx < 59; wx += 9) {
          var r = Math.random();
          g.fillStyle = r < 0.22 ? '#ffd9a0' : (r < 0.3 ? '#bfe8ff' : '#31465a');
          g.fillRect(wx, wy, 6, 6);
        }
      }
      var tx = new THREE.CanvasTexture(cv);
      tx.wrapS = tx.wrapT = THREE.RepeatWrapping;
      return tx;
    }
    var facade = facadeTexture();

    // ---------- 896 栋专家高楼：单段简化（性能：3× 几何降到 1×，仍保留天际线） ----------
    var TOTAL_H = UP_H + GAP_S + ACT_H + GAP_S + DN_H; // 28.5
    var bldGeo = new THREE.BoxGeometry(fp, TOTAL_H, fp);
    var bldMat = new THREE.MeshBasicMaterial({ color: 0xffffff, map: facade });
    var bldMesh = new THREE.InstancedMesh(bldGeo, bldMat, total);

    var idx = 0;
    for (var b2 = 0; b2 < N; b2++) {
      var pp = plazaPos(b2);
      for (var e = 0; e < perBlock; e++) {
        var ex = pp.x + cells[e][0], ez = pp.z + cells[e][1];
        var hf = 0.6 + (((e * 2246822519) >>> 0) % 100) / 100 * 1.0; // 楼高 0.6~1.6×，天际线起伏
        if (e % 89 === 0) hf *= 1.5;                                  // 每街区几栋地标超高楼
        // 确定性伪随机 top-K：16/896 点亮，其余压暗成休眠海
        var act = ((e * 2654435761 + b2 * 97) >>> 0) % perBlock < activeK;
        var j = 0.45 + (((e * 2654435761) >>> 0) % 100) / 100 * 0.4; // 楼体明度 0.45~0.85
        var c = act ? mintC.clone() : new THREE.Color(j, j * 0.95, j * 0.88);
        dummy.rotation.set(0, 0, 0); dummy.scale.set(1, hf, 1);
        dummy.position.set(ex, POD_TOP + TOTAL_H * hf / 2, ez); dummy.updateMatrix();
        bldMesh.setMatrixAt(idx, dummy.matrix); bldMesh.setColorAt(idx, c);
        idx++;
      }
    }
    bldMesh.instanceMatrix.needsUpdate = true;
    if (bldMesh.instanceColor) bldMesh.instanceColor.needsUpdate = true;
    city.add(bldMesh);

    // ---------- router 枢纽（裙楼顶亮薄荷八面体）+ 共享专家（街区边缘蓝色高楼） ----------
    var hubMesh = new THREE.InstancedMesh(
      new THREE.OctahedronGeometry(1.4),
      new THREE.MeshBasicMaterial({ color: 0xa8f5e0 }), N);
    var sharedMesh = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1.6, 4.2, 1.6),
      new THREE.MeshBasicMaterial({ color: sharedC }), N);
    // 隐形拾取代理盒：点中街区任意处 = experts，O(1) raycast
    var proxyMesh = new THREE.InstancedMesh(
      new THREE.BoxGeometry(EXP_W + 6, 16, EXP_D + 6),
      new THREE.MeshBasicMaterial({ visible: false }), N);
    for (var b3 = 0; b3 < N; b3++) {
      var ph = plazaPos(b3);
      dummy.rotation.set(0, 0, 0); dummy.scale.set(1, 1, 1);
      dummy.position.set(ph.x, podiumH + 1.2, ph.z); dummy.updateMatrix();
      hubMesh.setMatrixAt(b3, dummy.matrix);
      dummy.position.set(ph.x + EXP_W / 2 - 1.5, POD_TOP + 2.1, ph.z); dummy.updateMatrix();
      sharedMesh.setMatrixAt(b3, dummy.matrix);
      dummy.position.set(ph.x, 7, ph.z); dummy.updateMatrix();
      proxyMesh.setMatrixAt(b3, dummy.matrix);
    }
    [hubMesh, sharedMesh, proxyMesh].forEach(function (m) {
      m.instanceMatrix.needsUpdate = true; city.add(m);
    });
    reg('experts', city, [proxyMesh], new THREE.Vector3(0, 5, 0), 900, null);
    reg('router', city, [hubMesh], new THREE.Vector3(0, 5, 0), 900, null);

    // ---------- 深度流光路（AttnRes）：蛇形连接 60 街区的茜红发光路径 ----------
    var pathMat = new THREE.MeshBasicMaterial({
      color: COL().attnres, transparent: true, opacity: 0.45,
      blending: THREE.AdditiveBlending, depthWrite: false
    });
    var pathGeo = new THREE.BoxGeometry(1, 0.22, 2.4);
    var pathMesh = new THREE.InstancedMesh(pathGeo, pathMat, N - 1);
    for (var b4 = 0; b4 < N - 1; b4++) {
      var a = plazaPos(b4), c2 = plazaPos(b4 + 1);
      var mx = (a.x + c2.x) / 2, mz = (a.z + c2.z) / 2;
      var dx = c2.x - a.x, dz = c2.z - a.z;
      var len = Math.sqrt(dx * dx + dz * dz);
      dummy.position.set(mx, 0.5, mz);
      dummy.rotation.set(0, Math.atan2(dx, dz) + Math.PI / 2, 0);
      dummy.scale.set(len + 2, 1, 1);
      dummy.updateMatrix();
      pathMesh.setMatrixAt(b4, dummy.matrix);
    }
    pathMesh.instanceMatrix.needsUpdate = true;
    city.add(pathMesh);
    reg('attnres', city, [pathMesh], new THREE.Vector3(0, 0.5, 0), 900, null);

    // ---------- 城市入口：token 行 + embedding 柱；出口：logits + softmax ----------
    var p0 = plazaPos(0);
    var entryX = p0.x - PX / 2 - 18;
    var tokGroup = new THREE.Group();
    var tokMat = new THREE.MeshBasicMaterial({ color: 0xe8a94d });
    for (var ti = 0; ti < TOKEN_N; ti++) {
      var tc = new THREE.Mesh(new THREE.BoxGeometry(2.0, 2.0, 2.0), tokMat);
      tc.position.set(entryX, 1.6, p0.z - 10.5 + ti * 3.0);
      tokGroup.add(tc);
    }
    city.add(tokGroup);
    reg('tokenizer', city, tokGroup.children.slice(), new THREE.Vector3(entryX, 1.6, p0.z), 12, null);

    var embCore = new THREE.Mesh(
      new THREE.BoxGeometry(3.0, 5.4, 3.0),
      new THREE.MeshBasicMaterial({ color: 0xe8a94d }));
    embCore.position.set(entryX + 7, 3.0, p0.z);
    var embGlow = new THREE.Mesh(
      new THREE.BoxGeometry(3.8, 6.2, 3.8),
      new THREE.MeshBasicMaterial({ color: 0xe8a94d, transparent: true, opacity: 0.25, blending: THREE.AdditiveBlending, depthWrite: false }));
    embGlow.position.copy(embCore.position);
    city.add(embCore); city.add(embGlow);
    reg('embedding', city, [embCore], embCore.position.clone(), 4.5, null);

    var pN = plazaPos(N - 1);
    var exitX = pN.x - PX / 2 - 18;
    if (globals.output) {
      var outGroup = new THREE.Group();
      for (var oi = 0; oi < TOKEN_N; oi++) {
        var oc = new THREE.Mesh(
          new THREE.BoxGeometry(1.5, 1.5, 1.5),
          new THREE.MeshBasicMaterial({ color: 0xa89a83 }));
        oc.position.set(exitX, 1.4, pN.z - 8.75 + oi * 2.5);
        outGroup.add(oc);
      }
      city.add(outGroup);
      reg('output', city, outGroup.children.slice(), new THREE.Vector3(exitX, 1.4, pN.z), 9, null);
    }
    if (globals.softmax) {
      var smGroup = new THREE.Group();
      var bars = [];
      for (var si = 0; si < TOKEN_N; si++) {
        var bar = new THREE.Mesh(
          new THREE.BoxGeometry(1.3, 1, 1.3),
          new THREE.MeshBasicMaterial({ color: 0xa89a83 }));
        var bh = 0.8 + Math.random() * 3.0;
        bar.scale.y = bh;
        bar.position.set(exitX + 6, bh / 2 + 0.4, pN.z - 8.05 + si * 2.3);
        smGroup.add(bar); bars.push(bar);
      }
      city.add(smGroup);
      reg('softmax', city, bars, new THREE.Vector3(exitX + 6, 2.4, pN.z), 9, null);
    }

    s.towerH = 16;
    s.scene.add(city);

    // ---------- 机位：城市航拍视角 ----------
    if (s.camera && s.camera.far < 5000) {
      if (!keepEnv || savedFar == null) savedFar = s.camera.far;
      s.camera.far = 5000;
      s.camera.updateProjectionMatrix();
    }
    var cityW = COLS_B * PX, cityD = Math.ceil(N / COLS_B) * PZ;
    // 默认机位：街区尺度（约 2~3 个街区入画），楼体可读；全城由环绕/缩放探索
    s.home = {
      pos: [cityW * 0.08, 110, cityD * 0.5 + 240],
      target: [0, 18, 30]
    };
    if (E.camera) {
      if (E.camera.flyTo) E.camera.flyTo(s.home.pos, s.home.target, 1.6);
      else if (E.camera.setPose) E.camera.setPose(s.home.pos, s.home.target);
    }

    s._mode = 'full';
    buildHud();
    return full;
  }

  function dispose() {
    var s = S();
    if (!s || !s._full) return;
    var full = s._full;
    var prev = full.prev;

    if (full.group) {
      s.scene.remove(full.group);
      disposeObj(full.group);   // 释放 geometry / material / instanceColor
    }
    disposeHud();
    // 精确回滚 detail 现场
    s.pickables = prev.pickables;
    s.components = prev.components;
    s.home = prev.home;
    s.towerH = prev.towerH;
    if (prev.tower) prev.tower.visible = true;
    if (savedFar != null && s.camera) { s.camera.far = savedFar; s.camera.updateProjectionMatrix(); savedFar = null; }
    if (s.scene.fog && savedFogDensity != null) s.scene.fog.density = savedFogDensity;
    savedFogDensity = null;
    s._full = null;
    s._mode = prev.mode === 'full' ? 'detail' : (prev.mode || 'detail');
    // numbers 面板兼容③：细节现场已还原（s._full=null 后包装层直通原实现），数值牌重新挂回细节塔
    if (typeof E.refreshNumbers === 'function') { try { E.refreshNumbers(); } catch (e) {} }
    // 相机飞回 detail 机位：城市机位距离 ~800，切回细节后塔会被 far 面裁掉（黑屏）
    if (E.camera && s.home) {
      if (E.camera.flyTo) { try { E.camera.flyTo(s.home.pos, s.home.target, 1.2); } catch (e) {} }
      else if (E.camera.setPose) { try { E.camera.setPose(s.home.pos, s.home.target); } catch (e) {} }
    }
  }

  function isActive() {
    var s = S();
    return !!(s && s._full);
  }

  // detail 场景被重建（switchModel / single 内部会走 E.buildScene；tour 换模型同）后，
  // 旧快照里的 tower/pickables 已失效：丢弃旧全塔，基于新 detail 现场重建。
  function rebuildAfterSceneChange(spec) {
    var s = S();
    var full = s._full;
    if (full && full.group) {
      s.scene.remove(full.group);
      disposeObj(full.group);
    }
    s._full = null;
    build(spec || s.spec, true); // keepEnv：雾密度/far 沿用首次捕获的 detail 值
  }

  // numbers 面板兼容②：全景模式下数值牌（细节塔尺度坐标）不应挂出。
  // 但 numbers 的 attach 只在 comps 缺失时跳过，而全景城市也注册了 kda/router 等 components，
  // simStep/setSampler/buildScene 任一触发都会把面板误挂到城市中心。
  // 包装 refreshNumbers：全景期间临时置空 components → 原实现退化为「只清不挂」。
  var refreshHooked = false;
  function hookRefreshNumbers() {
    if (refreshHooked || typeof E.refreshNumbers !== 'function') return; // numbers.js 未加载则无需兼容
    refreshHooked = true;
    var origRefresh = E.refreshNumbers;
    E.refreshNumbers = function () {
      var s = S();
      if (!s || !s._full) return origRefresh.apply(E, arguments);
      // 全景塔下不挂细节数值牌，只刷新城市 HUD
      var keep = s.components;
      s.components = {};
      try { origRefresh.apply(E, arguments); } finally { s.components = keep; }
      updateHud();
    };
  }

  // pipeline.compare 持有 scene-builder 原始 buildScene 的引用（origBuildScene），
  // 绕过全部包装：全景期间调 compare 会在城市之上直接重建双细节塔，快照/pickables 脱节。
  // 与外壳 switchModel 同一约定：compare 前先退出全景（single 走 E.buildScene，已被上面的包装覆盖）。
  var compareHooked = false;
  function hookCompare() {
    if (compareHooked || typeof E.compare !== 'function') return;
    compareHooked = true;
    var origCompare = E.compare;
    E.compare = function () {
      if (isActive()) dispose();
      return origCompare.apply(E, arguments);
    };
  }

  // 包装 E.buildScene（与 pipeline.js 相同的增量挂载手法）：
  // 全塔激活期间任何 buildScene 调用都会重建 detail 现场并覆盖
  // home/pickables/components/tower —— 必须在其后重建全塔，否则状态脱节黑屏。
  var buildSceneHooked = false;
  function hookBuildScene() {
    if (buildSceneHooked || typeof E.buildScene !== 'function') return;
    buildSceneHooked = true;
    var orig = E.buildScene;
    E.buildScene = function (spec) {
      var wasFull = isActive();
      var r = orig.apply(E, arguments);
      if (wasFull) rebuildAfterSceneChange(spec);
      return r;
    };
  }
  hookBuildScene(); // 加载时即可用则立即挂；否则 build() 里补挂
  hookRefreshNumbers();
  hookCompare();

  return { build: build, dispose: dispose, isActive: isActive };
})();
