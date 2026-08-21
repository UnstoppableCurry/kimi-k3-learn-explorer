/* pipeline.js — v2 扩展：采样 / 新特效 / 训练阶段 / 双塔对比 / CV 横向流水线
 * 增量挂载 window.ENGINE，不改既有引擎文件：
 *   setSampler({temperature,topP}) / setStage('pretrain'|'sft'|'rl'|'off')
 *   compare(idA,idB) / single(id)
 *   playEffect 新增 'tokenize' 'rope' 'gate' 'aggregate' 'sample'；覆盖 'attnres'（顿悟两段式光束）
 *   质感打磨：tokenize 碎块翻转+落位弹跳 / gate 分数线 60ms 依次点亮 / aggregate 冲击波环+Σ脉冲
 *   buildScene 包装：spec.category==='cv' 走横向流水线 buildCv
 */
window.ENGINE = window.ENGINE || {};
window.PIPELINE_VERSION = 'v2-timeline';
(function () {
  var E = window.ENGINE;
  function S() { return E._state; }
  function COL() { return E.COLORS; }

  // ---------- 本地小工具（scene-builder 内部函数不可达，这里自给自足） ----------
  function pulseAnim(apply, dur, delay, done) {
    S().anims.push({
      t: 0, dur: dur || 0.6, delay: delay || 0,
      update: function (k) { apply(Math.sin(Math.PI * k)); },
      done: done || null
    });
  }

  function textSprite(text, px, color, scaleX, scaleY) {
    var c = document.createElement('canvas');
    c.width = 512; c.height = 128;
    var g = c.getContext('2d');
    g.font = (px || 48) + 'px "Songti SC", "PingFang SC", serif';
    g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillStyle = color || '#c9bba6';
    g.fillText(text, 256, 66);
    var tex = new THREE.CanvasTexture(c);
    if (THREE.SRGBColorSpace) tex.colorSpace = THREE.SRGBColorSpace;
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    sp.scale.set(scaleX || 9, scaleY || 2.3, 1);
    return sp;
  }

  // 与 scene-builder.reg 同语义：挂 componentId + 入 pickables/components
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

  function compCenter(type, idx) {
    var list = S().components[type];
    if (!list || !list.length) return null;
    return (list[idx] || list[0]).center;
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

  // ======================================================================
  // 1) setSampler：softmax 条按 temperature / topP 重塑
  // ======================================================================
  var TOP1 = 0xe8a94d;
  E.setSampler = function (opts) {
    var rec = S()._fx && S()._fx.softmax;
    if (!rec || !opts) return;
    var T = Math.max(0.05, opts.temperature == null ? 1 : opts.temperature);
    var topP = Math.min(1, Math.max(0.05, opts.topP == null ? 1 : opts.topP));
    var n = rec.bars.length;

    // 当前高度当 logit：z = ln(h)
    var z = [];
    for (var i = 0; i < n; i++) z[i] = Math.log(Math.max(rec.heights[i], 1e-4)) / T;
    var mx = -Infinity;
    for (i = 0; i < n; i++) if (z[i] > mx) mx = z[i];
    var p = [], sum = 0;
    for (i = 0; i < n; i++) { p[i] = Math.exp(z[i] - mx); sum += p[i]; }
    for (i = 0; i < n; i++) p[i] /= sum;

    // top-p：按概率降序截断尾部再归一化
    var order = [];
    for (i = 0; i < n; i++) order.push(i);
    order.sort(function (a, b) { return p[b] - p[a]; });
    var cum = 0, keep = {};
    for (i = 0; i < n; i++) {
      keep[order[i]] = true;
      cum += p[order[i]];
      if (cum >= topP) break;
    }
    var top1 = order[0];
    sum = 0;
    for (i = 0; i < n; i++) { if (!keep[i]) p[i] = 0; sum += p[i]; }
    for (i = 0; i < n; i++) p[i] /= sum;

    var from = rec.heights.slice();
    var to = [];
    for (i = 0; i < n; i++) to[i] = 0.25 + p[i] * 5.6;
    S().anims.push({
      t: 0, dur: 0.8, delay: 0,
      update: function (k) {
        var e = 1 - Math.pow(1 - k, 3);
        for (var i = 0; i < n; i++) {
          var h = from[i] + (to[i] - from[i]) * e;
          rec.bars[i].scale.y = h;
          rec.bars[i].position.y = rec.baseY + h / 2;
          rec.heights[i] = h;
        }
      },
      done: function () {
        for (var i = 0; i < n; i++) {
          rec.bars[i].material.color.setHex(i === top1 ? TOP1 : 0x8a7d68);
          rec.bars[i].material.emissiveIntensity = i === top1 ? 0.9 : 0.4;
        }
      }
    });
    S()._sampler = { temperature: T, topP: topP };
  };

  // 按当前条高分布随机抽一根（sample 特效用）
  function sampleFromBars(rec) {
    var sum = 0;
    for (var i = 0; i < rec.heights.length; i++) sum += rec.heights[i];
    var r = Math.random() * sum, acc = 0;
    for (i = 0; i < rec.heights.length; i++) {
      acc += rec.heights[i];
      if (r <= acc) return i;
    }
    return rec.heights.length - 1;
  }

  // ======================================================================
  // 2) 新特效
  // ======================================================================
  var NEW_EFFECTS = {

    // 塔底上方一句中文 → 碎成 8 块飞向 token 立方体
    'tokenize': function () {
      var s = S();
      var fx = s._fx.tokens;
      if (!fx) return;
      var sentence = '月之暗面发布了k3开源大模型';
      var full = textSprite(sentence, 52, '#e8dcc8', 12, 3);
      full.position.set(0, 4.6, 0);
      s.tower.add(full);
      s.anims.push({
        t: 0, dur: 0.9, delay: 0,
        update: function (k) { full.material.opacity = Math.min(1, k * 2); },
        done: function () {
          s.tower.remove(full);
          disposeObj(full);
          // 8 个碎片 = 8 个字符块，各自飞向对应 token
          var chars = ['月', '之', '暗', '面', '发', '布', 'K', '3'];
          for (var i = 0; i < 8; i++) {
            (function (i) {
              var frag = textSprite(chars[i], 64, '#e8a94d', 2.2, 2.2);
              frag.position.set(-4.4 + i * 1.25, 4.6, 0);
              s.tower.add(frag);
              var tgt = fx.cubes[i].position.clone();
              var from = frag.position.clone();
              var spin = (i % 2 ? 1 : -1) * (0.35 + (i % 3) * 0.12); // 小角度翻转，交替方向
              s.anims.push({
                t: 0, dur: 0.5, delay: i * 0.04,
                update: function (k) {
                  var e = 1 - Math.pow(1 - k, 3);   // 快速 easeOutCubic：碎裂要「脆」
                  frag.position.lerpVectors(from, tgt, e);
                  var sc = 1 - 0.55 * e;
                  frag.scale.set(2.2 * sc, 2.2 * sc, 1);
                  frag.material.opacity = 1 - 0.6 * e;
                  frag.material.rotation = spin * Math.sin(Math.PI * k); // 翻出再回正
                },
                done: function () {
                  s.tower.remove(frag);
                  disposeObj(frag);
                  // 落位弹跳：overshoot 峰值 1.06 倍，带一次轻微回弹后归位
                  var cube = fx.cubes[i];
                  s.anims.push({
                    t: 0, dur: 0.3, delay: 0,
                    update: function (kk) {
                      cube.scale.setScalar(1 + 0.10 * Math.sin(kk * Math.PI * 2.2) * Math.exp(-2.5 * kk));
                    },
                    done: function () { cube.scale.setScalar(1); }
                  });
                  pulseAnim(function (kk) {
                    fx.mats[i].emissiveIntensity = 0.3 + 1.5 * kk;
                  }, 0.4);
                }
              });
            })(i);
          }
        }
      });
    },

    // embedding 柱周围 3 个旋转光环，2 秒
    'rope': function () {
      var s = S();
      var c = compCenter('embedding') || new THREE.Vector3(0, 6.8, 0);
      var rings = [];
      for (var i = 0; i < 3; i++) {
        var ring = new THREE.Mesh(
          new THREE.TorusGeometry(2.4 + i * 0.5, 0.07, 8, 48),
          new THREE.MeshBasicMaterial({ color: COL().mla, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })
        );
        ring.position.copy(c);
        ring.rotation.x = Math.PI / 2 + (i - 1) * 0.4;
        s.tower.add(ring);
        rings.push(ring);
      }
      s.anims.push({
        t: 0, dur: 2.0, delay: 0,
        update: function (k) {
          var fade = Math.sin(Math.PI * Math.min(k * 1.15, 1));
          for (var i = 0; i < 3; i++) {
            rings[i].rotation.z += 0.06 + i * 0.03;
            rings[i].position.y = c.y + Math.sin(k * Math.PI * 4 + i * 2) * 0.9;
            rings[i].material.opacity = 0.65 * fade;
          }
        },
        done: function () {
          for (var i = 0; i < 3; i++) { s.tower.remove(rings[i]); disposeObj(rings[i]); }
        }
      });
    },

    // router → 专家阵顶部：16 条分数线「依次点亮」（间隔 60ms）+ 线上方小分数条，2.2 秒
    'gate': function () {
      var s = S();
      var from = compCenter('router');
      var expC = compCenter('experts');
      if (!from || !expC) return;
      var start = from.clone(); start.y += 0.8;
      var objs = [], lines = [], scores = [];
      var DUR = 2.2, STAG = 0.06;   // 60ms  stagger：爽感来自「一条接一条」
      for (var i = 0; i < 16; i++) {
        scores.push(0.2 + Math.random() * 0.8);
        var end = new THREE.Vector3(-5.6 + i * 0.75, expC.y + 2.2, (Math.random() - 0.5) * 3);
        // 每条线独立 Line + 独立材质，才能逐条点亮（原实现单 LineSegments 只能同亮同灭）
        var line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([start.clone(), end]),
          new THREE.LineBasicMaterial({ color: 0x9ff5e2, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })
        );
        lines.push(line);
        s.tower.add(line);
        // 小分数条：立在终点上方，高度∝分数，跟随本线的点亮节奏
        var bar = new THREE.Mesh(
          new THREE.BoxGeometry(0.28, 1, 0.28),
          new THREE.MeshBasicMaterial({ color: COL().moe, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })
        );
        bar.position.set(end.x, end.y + 0.5, end.z);
        objs.push(bar);
        s.tower.add(bar);
      }
      s.anims.push({
        t: 0, dur: DUR, delay: 0,
        update: function (k) {
          var t = k * DUR;
          var fadeOut = t > 1.75 ? Math.max(1 - (t - 1.75) / 0.45, 0) : 1;
          for (var i = 0; i < 16; i++) {
            // 第 i 条线延迟 i*60ms 后 0.2s 内冲亮
            var lp = Math.min(Math.max((t - i * STAG) / 0.2, 0), 1);
            var le = 1 - Math.pow(1 - lp, 2);
            lines[i].material.opacity = 0.9 * le * fadeOut;
            // 分数条紧跟本线节奏生长
            var bp = Math.min(Math.max((t - i * STAG - 0.06) / 0.35, 0), 1);
            var be = 1 - Math.pow(1 - bp, 3);
            var h = 0.3 + scores[i] * 1.6 * be;
            objs[i].scale.y = h;
            objs[i].position.y = expC.y + 2.2 + h / 2;
            objs[i].material.opacity = 0.85 * (bp > 0 ? 1 : 0) * fadeOut;
          }
        },
        done: function () {
          for (var i = 0; i < 16; i++) {
            s.tower.remove(lines[i]); disposeObj(lines[i]);
            s.tower.remove(objs[i]); disposeObj(objs[i]);
          }
        }
      });
    },

    // 点亮专家（错峰 + 冲击波环）→ Σ 球（汇聚完成后脉冲一次）→ 合并单柱向上，2.8 秒
    'aggregate': function () {
      var s = S();
      var list = s._fx.experts;
      if (!list || !list.length) return;
      var entry = list[0], rec = entry.expertsRec;
      var mesh = rec.mesh, n = rec.n;
      var cnt = Math.min(rec.active, n);
      var used = {}, ids = [];
      while (ids.length < cnt) {
        var id = Math.floor(Math.random() * n);
        if (!used[id]) { used[id] = 1; ids.push(id); }
      }
      var expC = compCenter('experts');
      var sigmaPos = new THREE.Vector3(expC.x, expC.y + 4.2, expC.z);
      var DUR = 2.8, STAG = 0.07;   // 专家错峰点亮相隔 70ms

      // Σ 球
      var sigma = new THREE.Mesh(
        new THREE.SphereGeometry(0.85, 20, 16),
        new THREE.MeshBasicMaterial({ color: COL().moe, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })
      );
      sigma.position.copy(sigmaPos);
      var sigmaLabel = textSprite('Σ', 72, '#9ff5e2', 2.6, 2.6);
      sigmaLabel.position.copy(sigmaPos);
      s.tower.add(sigma); s.tower.add(sigmaLabel);

      // 专家 → Σ 的圆柱线（半径∝随机权重）
      var tmpM = new THREE.Matrix4(), tmpP = new THREE.Vector3();
      var beams = [], starts = [], lit = [];
      mesh.updateMatrixWorld(true);
      for (var i = 0; i < ids.length; i++) {
        mesh.getMatrixAt(ids[i], tmpM);
        tmpP.setFromMatrixPosition(tmpM);
        var from = tmpP.clone();
        starts.push(from);
        lit.push(false);
        var w = 0.25 + Math.random();          // 随机权重
        var dir = sigmaPos.clone().sub(from);
        var len = dir.length();
        var cyl = new THREE.Mesh(
          new THREE.CylinderGeometry(0.03 * w, 0.03 * w, len, 6),
          new THREE.MeshBasicMaterial({ color: 0x9ff5e2, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })
        );
        cyl.position.copy(from).addScaledVector(dir, 0.5);
        cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
        s.tower.add(cyl);
        beams.push(cyl);
      }
      // 合并单柱
      var col = new THREE.Mesh(
        new THREE.CylinderGeometry(0.35, 0.35, 1, 10),
        new THREE.MeshBasicMaterial({ color: COL().moe, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })
      );
      col.position.copy(sigmaPos);
      s.tower.add(col);

      var convT = (ids.length - 1) * STAG + 0.35;   // 汇聚完成时刻（秒）

      s.anims.push({
        t: 0, dur: DUR, delay: 0,
        update: function (k) {
          var t = k * DUR;
          var fadeOut = t > DUR - 0.45 ? Math.max(1 - (t - (DUR - 0.45)) / 0.45, 0) : 1;
          for (var i = 0; i < beams.length; i++) {
            var bp = Math.min(Math.max((t - i * STAG) / 0.3, 0), 1);
            var be = 1 - Math.pow(1 - bp, 2);
            beams[i].material.opacity = 0.75 * be * fadeOut;
            if (!lit[i] && bp > 0) {
              lit[i] = true;
              // 专家被点亮的瞬间：脚下炸开一个向外扩散的小冲击波环
              var ring = new THREE.Mesh(
                new THREE.TorusGeometry(0.42, 0.06, 8, 40),
                new THREE.MeshBasicMaterial({ color: COL().moe, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false })
              );
              ring.position.copy(starts[i]);
              ring.rotation.x = Math.PI / 2;
              s.tower.add(ring);
              s.anims.push({
                t: 0, dur: 0.55, delay: 0, _ring: ring,
                update: function (kk) {
                  var re = 1 - Math.pow(1 - kk, 3);
                  this._ring.scale.setScalar(1 + 2.6 * re);
                  this._ring.material.opacity = 0.85 * (1 - re);
                },
                done: function () {
                  s.tower.remove(this._ring); disposeObj(this._ring);
                }
              });
            }
          }
          // Σ 亮度随汇聚爬升；汇聚完成后脉冲一次（涨 1.32 倍再收回）
          var ramp = Math.min(t / 1.1, 1);
          var pulse = t > convT ? Math.sin(Math.PI * Math.min((t - convT) / 0.5, 1)) : 0;
          sigma.material.opacity = (0.75 * ramp + 0.25 * pulse) * fadeOut;
          sigma.scale.setScalar(1 + 0.32 * pulse);
          sigmaLabel.material.opacity = ramp * (1 - Math.max((t - (DUR - 0.6)) / 0.6, 0));
          // 合并单柱：汇聚接近尾声时开始生长
          var k2 = Math.min(Math.max((t - convT * 0.8) / 0.9, 0), 1);
          var h = 0.2 + k2 * 3.4;
          col.scale.y = h;
          col.position.y = sigmaPos.y + h / 2;
          col.material.opacity = 0.9 * fadeOut * (k2 > 0 ? 1 : 0);
        },
        done: function () {
          s.tower.remove(sigma); disposeObj(sigma);
          s.tower.remove(sigmaLabel); disposeObj(sigmaLabel);
          s.tower.remove(col); disposeObj(col);
          for (var i = 0; i < beams.length; i++) { s.tower.remove(beams[i]); disposeObj(beams[i]); }
        }
      });
    },

    // softmax 条轮盘闪选 1.5 秒，停在按当前分布选中的条
    'sample': function () {
      var s = S();
      var rec = s._fx.softmax;
      if (!rec) return;
      var n = rec.bars.length;
      var winner = sampleFromBars(rec);
      var last = -1, flipT = 0;
      s.anims.push({
        t: 0, dur: 1.5, delay: 0,
        update: function (k) {
          flipT -= 1 / 60;
          if (k < 0.75) {
            if (flipT <= 0) {
              flipT = 0.04 + k * 0.12;  // 越转越慢
              if (last >= 0) rec.bars[last].material.color.setHex(0x8a7d68);
              last = Math.floor(Math.random() * n);
              rec.bars[last].material.color.setHex(TOP1);
            }
          } else {
            if (last >= 0 && last !== winner) rec.bars[last].material.color.setHex(0x8a7d68);
            rec.bars[winner].material.color.setHex(TOP1);
            rec.bars[winner].material.emissiveIntensity = 0.9 + 0.4 * Math.sin(k * 30);
            last = winner;
          }
        },
        done: function () {
          for (var i = 0; i < n; i++) {
            rec.bars[i].material.color.setHex(i === winner ? TOP1 : 0x8a7d68);
            rec.bars[i].material.emissiveIntensity = i === winner ? 0.9 : 0.4;
          }
        }
      });
    },

    // AttnRes「顿悟」：光束先暗下去 → 一缕微光加速下探、触及下层闪一下 → 亮涌自底部全亮回流
    // （覆盖 scene-builder 的匀速脉冲版；经 playEffect 包装器优先命中本表）
    'attnres': function () {
      var s = S();
      var beams = s._fx.beams;
      if (!beams || !beams.length) return;
      for (var bi = 0; bi < beams.length; bi++) {
        (function (rec, bi) {
          var beam = rec.beam, mat = rec.mat;
          var DUR = 2.3;
          var top = beam.position.y + 2.2, bot = beam.position.y - 2.2; // 柱高 4.4
          // 下探微光
          var probe = new THREE.Mesh(
            new THREE.CylinderGeometry(0.16, 0.16, 1.1, 10),
            new THREE.MeshBasicMaterial({ color: COL().attnres, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })
          );
          probe.position.set(beam.position.x, top, beam.position.z);
          // 回流亮涌
          var surge = new THREE.Mesh(
            new THREE.CylinderGeometry(0.72, 0.72, 1.5, 14, 1, true),
            new THREE.MeshBasicMaterial({ color: 0xff9ec4, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide })
          );
          surge.position.set(beam.position.x, bot, beam.position.z);
          s.tower.add(probe); s.tower.add(surge);
          rec._fxTmp = { probe: probe, surge: surge };   // 供外部断言读取运行态
          s.anims.push({
            t: 0, dur: DUR, delay: bi * 0.25,
            start: function () { rec._busy = true; },
            update: function (k) {
              var t = k * DUR;
              if (t < 1.05) {
                // 第一段：光束暗下去（0.3 → 0.12），微光 0.15s 起加速下探
                var dim = 0.3 - 0.18 * Math.min(t / 0.35, 1);
                var pp = Math.min(Math.max((t - 0.15) / 0.8, 0), 1);
                var pe = pp * pp;                       // easeIn：试探着加速下探
                probe.position.y = top - pe * (top - bot);
                probe.material.opacity = pp > 0 ? 0.24 : 0;
                // 触及下层的一瞬：微光闪一下，光束跟着回一点亮
                if (pp >= 1) {
                  var flash = Math.sin(Math.PI * Math.min((t - 0.95) / 0.25, 1));
                  probe.material.opacity = 0.24 + 0.4 * flash;
                  dim += 0.16 * flash;
                }
                mat.opacity = dim;
              } else {
                // 第二段：亮涌自底部升回塔身，光束全亮回流
                var q = Math.min((t - 1.05) / 0.8, 1);
                var qe = q < 0.5 ? 2 * q * q : 1 - Math.pow(-2 * q + 2, 2) / 2; // easeInOut
                surge.position.y = bot + qe * (top - bot);
                surge.material.opacity = 0.85 * Math.sin(Math.PI * Math.min(q * 1.08, 1));
                var bright = 0.28 + 0.67 * Math.min((t - 1.05) / 0.75, 1);
                if (t > 1.95) bright = 0.95 - 0.65 * Math.min((t - 1.95) / 0.35, 1); // 沉淀回 idle
                mat.opacity = bright;
                var rr = 1 + 0.18 * Math.sin(Math.PI * Math.min((t - 1.05) / 0.9, 1));
                beam.scale.set(rr, 1, rr);
              }
            },
            done: function () {
              s.tower.remove(probe); disposeObj(probe);
              s.tower.remove(surge); disposeObj(surge);
              rec._fxTmp = null;
              mat.opacity = 0.3;
              beam.scale.set(1, 1, 1);
              rec._busy = false;
            }
          });
        })(beams[bi], bi);
      }
    }
  };

  // 包装 playEffect：先查新表，未命中回落原实现
  var origPlayEffect = E.playEffect;
  E.playEffect = function (name) {
    if (NEW_EFFECTS[name]) {
      if (!S().tower) return;
      NEW_EFFECTS[name]();
      return;
    }
    if (origPlayEffect) return origPlayEffect(name);
    console.warn('[engine] 未知特效: ' + name);
  };

  // ======================================================================
  // 3) setStage：训练阶段 3D 时间线
  // ======================================================================
  var STAGE_TEXT = {
    pretrain: '预训练：全部参数更新',
    sft: 'SFT：QAT 量化感知训练',
    rl: 'RL：奖励信号反向塑形'
  };
  var STAGE_COLOR = { pretrain: 0x7aa2f7, sft: 0xe8a94d, rl: 0xe0699b };
  var STAGE_X = { pretrain: -24, sft: 0, rl: 24 };

  function clearStage() {
    var s = S();
    // 清理训练时间线场景
    if (s._trainingGroup) {
      s.scene.remove(s._trainingGroup);
      disposeObj(s._trainingGroup);
      s._trainingGroup = null;
    }
    s._trainingStage = null;
    s.inTraining = false;
    // 恢复推理塔可见，并触发数值面板重挂
    if (s.tower) s.tower.visible = true;
    if (typeof E.refreshNumbers === 'function') { try { E.refreshNumbers(); } catch (e) {} }
  }

  // 小型塔缩影：6 层薄板堆叠
  function miniTower(stage, active) {
    var g = new THREE.Group();
    var layers = ['norm', 'kda', 'mla', 'router', 'experts', 'norm'];
    var y = 0;
    var h = 0.7;
    var color = STAGE_COLOR[stage];
    var cBase = new THREE.Color(active ? color : 0x4a4036);
    layers.forEach(function (t, i) {
      var mat = new THREE.MeshStandardMaterial({
        color: cBase.clone().lerp(new THREE.Color(0x171310), i / 8),
        roughness: 0.7, emissive: color, emissiveIntensity: active ? 0.35 : 0.05
      });
      var slab = new THREE.Mesh(new THREE.BoxGeometry(2.4, h, 1.6), mat);
      slab.position.y = y + h / 2;
      g.add(slab);
      y += h + 0.12;
    });
    return g;
  }

  // 阶段曲线：pretrain/sft loss 下降，rl reward 上升
  function stageCurve(stage, active) {
    var g = new THREE.Group();
    var pts = [];
    var n = 24;
    var color = STAGE_COLOR[stage];
    for (var i = 0; i <= n; i++) {
      var x = -5 + (i / n) * 10;
      var y;
      if (stage === 'rl') y = Math.pow(i / n, 0.7) * 4.5;            // reward 上升
      else y = 4.5 * (1 - Math.pow(i / n, 0.55));                    // loss 下降
      pts.push(new THREE.Vector3(x, y, 0));
    }
    var geo = new THREE.BufferGeometry().setFromPoints(pts);
    var line = new THREE.Line(geo, new THREE.LineBasicMaterial({
      color: color, transparent: true, opacity: active ? 0.9 : 0.35,
      blending: THREE.AdditiveBlending, depthWrite: false, linewidth: 2
    }));
    g.add(line);
    // 轴标签
    var yLabel = textSprite(stage === 'rl' ? 'reward ↑' : 'loss ↓', 28, active ? '#c9bba6' : '#6f6350', 4, 1.6);
    yLabel.position.set(-6, 2.5, 0);
    g.add(yLabel);
    return g;
  }

  // 数据/样本/奖励粒子流
  function addFlow(group, fromX, toX, y, z, color, reverse) {
    var N = 48;
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    var pts = new THREE.Points(geo, new THREE.PointsMaterial({
      color: color, size: 0.55, transparent: true, opacity: 0.85,
      blending: THREE.AdditiveBlending, depthWrite: false
    }));
    pts.frustumCulled = false;
    pts.userData = { fromX: fromX, toX: toX, y: y, z: z, N: N, t0: S().time };
    group.add(pts);

    // 常驻驱动：每帧更新粒子位置
    var tick = function (dt, t) {
      if (!pts.parent) { E._tickers = E._tickers.filter(function (f) { return f !== tick; }); return; }
      var attr = pts.geometry.getAttribute('position');
      var dir = reverse ? -1 : 1;
      var span = Math.abs(toX - fromX);
      for (var i = 0; i < N; i++) {
        var u = ((i / N + (t - (pts.userData.t0 || 0)) * 0.15 * dir) % 1 + 1) % 1;
        var x = fromX + (toX - fromX) * u;
        var yy = y + Math.sin(u * Math.PI * 2 + i) * 0.6;
        var zz = z + Math.cos(u * Math.PI * 3 + i * 0.7) * 0.8;
        attr.setXYZ(i, x, yy, zz);
      }
      attr.needsUpdate = true;
    };
    E._tickers.push(tick);
  }

  // SFT 量化闪光：小方块围绕塔闪烁
  function addQatSparkle(group, x, y, z) {
    var N = 20;
    var sparks = [];
    for (var i = 0; i < N; i++) {
      var mesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.25, 0.25, 0.25),
        new THREE.MeshBasicMaterial({ color: 0xe8a94d, transparent: true, opacity: 0 })
      );
      var ang = (i / N) * Math.PI * 2;
      mesh.position.set(x + Math.cos(ang) * 3, y + Math.sin(i * 1.3), z + Math.sin(ang) * 2);
      group.add(mesh);
      sparks.push({ mesh: mesh, ang: ang, speed: 0.5 + Math.random() * 0.8 });
    }
    var tick = function (dt, t) {
      if (!sparks[0].mesh.parent) { E._tickers = E._tickers.filter(function (f) { return f !== tick; }); return; }
      sparks.forEach(function (s, i) {
        var p = (t * s.speed + i / N) % 1;
        s.mesh.material.opacity = Math.sin(p * Math.PI) * 0.9;
        s.mesh.rotation.x = t + i; s.mesh.rotation.y = t * 0.7 + i;
      });
    };
    E._tickers.push(tick);
  }

  E.setStage = function (stage) {
    var s = S();
    clearStage();
    if (!stage || stage === 'off' || !s.tower) return;
    if (!STAGE_TEXT[stage]) { console.warn('[engine] 未知阶段: ' + stage); return; }

    // 隐藏推理塔，切换到训练剧场
    s.tower.visible = false;
    s.inTraining = true;
    s._trainingStage = stage;

    var tg = new THREE.Group();
    s._trainingGroup = tg;
    s.scene.add(tg);

    var STAGES = ['pretrain', 'sft', 'rl'];
    var BASE_Y = 2.5;

    // 三阶段平台 + 小塔 + 曲线
    STAGES.forEach(function (st) {
      var active = st === stage;
      var x = STAGE_X[st];
      var col = STAGE_COLOR[st];

      // 平台
      var plat = new THREE.Mesh(
        new THREE.BoxGeometry(16, 0.35, 11),
        new THREE.MeshStandardMaterial({ color: 0x2a231d, roughness: 0.85, transparent: true, opacity: active ? 0.9 : 0.45 })
      );
      plat.position.set(x, BASE_Y - 0.18, 0);
      tg.add(plat);

      // 发光边框
      var edge = new THREE.LineSegments(
        new THREE.EdgesGeometry(plat.geometry),
        new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: active ? 0.6 : 0.2 })
      );
      edge.position.copy(plat.position);
      tg.add(edge);

      // 小塔
      var mt = miniTower(st, active);
      mt.position.set(x, BASE_Y, 0);
      tg.add(mt);

      // 曲线
      var cv = stageCurve(st, active);
      cv.position.set(x, BASE_Y + 0.5, 4);
      tg.add(cv);

      // 标签
      var lbl = textSprite(STAGE_TEXT[st], 36, active ? '#efe6d8' : '#6f6350', 13, 3);
      lbl.position.set(x, BASE_Y + 7.5, 0);
      tg.add(lbl);
    });

    // 进入训练模式后清掉推理数值面板
    if (typeof E.refreshNumbers === 'function') { try { E.refreshNumbers(); } catch (e) {} }

    // 当前阶段特效
    if (stage === 'pretrain') {
      addFlow(tg, STAGE_X['pretrain'] - 12, STAGE_X['pretrain'] - 2, BASE_Y + 1, 0, STAGE_COLOR['pretrain']);
    } else if (stage === 'sft') {
      addFlow(tg, STAGE_X['sft'] - 12, STAGE_X['sft'] - 2, BASE_Y + 1, 0, STAGE_COLOR['sft']);
      addQatSparkle(tg, STAGE_X['sft'], BASE_Y + 3.5, 0);
    } else if (stage === 'rl') {
      addFlow(tg, STAGE_X['rl'] + 12, STAGE_X['rl'] + 2, BASE_Y + 1, 0, STAGE_COLOR['rl'], true);
    }

    // 相机拉远，让三阶段时间线并排可见，注视整个时间线中心
    if (E.camera && E.camera.flyTo) {
      E.camera.flyTo([STAGE_X[stage] * 0.12, 26, 68], [0, 5, 0], 1.0);
    }
  };

  // 顺序播放三阶段训练时间线
  var trainTimer = null;
  E.playTrainingTimeline = function () {
    if (trainTimer) { clearTimeout(trainTimer); trainTimer = null; }
    var stages = ['pretrain', 'sft', 'rl'];
    var i = 0;
    function next() {
      if (i >= stages.length) {
        E.setStage('off');
        trainTimer = null;
        return;
      }
      E.setStage(stages[i]);
      i++;
      trainTimer = setTimeout(next, 5500);
    }
    next();
  };

  E.stopTrainingTimeline = function () {
    if (trainTimer) { clearTimeout(trainTimer); trainTimer = null; }
    E.setStage('off');
  };

  // ======================================================================
  // 4) compare / single 双塔
  // ======================================================================
  function specOf(id) {
    var sp = (window.MODEL_SPECS || {})[id];
    if (!sp) console.warn('[engine] MODEL_SPECS 缺少模型: ' + id);
    return sp;
  }

  E.compare = function (idA, idB) {
    var s = S();
    var spA = specOf(idA), spB = specOf(idB);
    if (!spA || !spB) return;
    // 清掉上一轮对比残留
    if (s._towerA) { s.scene.remove(s._towerA); disposeObj(s._towerA); s._towerA = null; }

    origBuildScene(spA);
    if (!s.tower) return;
    s.tower.position.x = -15;
    s.tower.scale.setScalar(0.72);
    s._towerA = s.tower;
    s.tower = null;               // 防止第二次 buildScene 销毁它

    origBuildScene(spB);
    if (!s.tower) return;
    s.tower.position.x = 15;
    s.tower.scale.setScalar(0.72);
    // pickables 以第二次 build 为准（契约接受）
    if (E.camera && E.camera.setPose) E.camera.setPose([0, 24, 52], [0, 11, 0]);
  };

  E.single = function (id) {
    var s = S();
    if (s._towerA) {
      s.scene.remove(s._towerA);
      disposeObj(s._towerA);
      s._towerA = null;
    }
    var sp = specOf(id);
    if (!sp) return;
    E.buildScene(sp);   // 走包装后的入口，CV 模型也正确
  };

  // ======================================================================
  // 5) CV 横向流水线
  // ======================================================================
  function checkerTexture() {
    var c = document.createElement('canvas');
    c.width = 64; c.height = 48;
    var g = c.getContext('2d');
    for (var y = 0; y < 12; y++) {
      for (var x = 0; x < 16; x++) {
        var on = (x + y) % 2 === 0;
        g.fillStyle = on ? '#3a3129' : '#241d15';
        g.fillRect(x * 4, y * 4, 4, 4);
      }
    }
    var t = new THREE.CanvasTexture(c);
    t.magFilter = THREE.NearestFilter;
    if (THREE.SRGBColorSpace) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }

  function buildCv(spec) {
    var s = S();
    s.spec = spec;
    // 与 buildScene 同语义的重置
    if (s.tower) { disposeObj(s.tower); s.scene.remove(s.tower); }
    s.anims.length = 0;
    s.pickables = [];
    s.components = {};
    s._idleFns = [];
    s._fx = { tokens: null, kda: [], mla: [], experts: [], beams: [], softmax: null, arc: null };

    var tower = new THREE.Group();
    s.tower = tower;

    var LINE_Y = 4;
    var comps = (spec.cv && spec.cv.components) || [];
    var n = comps.length;
    // x 轴 -20 → +20：输入平面在最左，组件依次排开
    var x0 = -20, x1 = 20;
    var span = n > 0 ? (x1 - x0) / (n + 1) : 0;

    // 输入图像平面（棋盘格）
    var img = new THREE.Mesh(
      new THREE.PlaneGeometry(5.3, 4),
      new THREE.MeshStandardMaterial({ map: checkerTexture(), roughness: 0.8, emissive: 0x332b1e, emissiveIntensity: 0.25 })
    );
    img.position.set(x0, LINE_Y, 0);
    tower.add(img);
    reg('input', tower, [img], new THREE.Vector3(x0, LINE_Y, 0), 3.4, function () {
      pulseAnim(function (k) { img.material.emissiveIntensity = 0.25 + 1.1 * k; }, 0.7);
    });

    // 组件块：统一暖灰块 + 标签（未知 type 同样回退，不报错）
    for (var i = 0; i < n; i++) {
      (function (comp, i) {
        var type = comp.type || 'block';
        var labelText = comp.label || type;
        var x = x0 + span * (i + 1);
        var group = new THREE.Group();
        var mat = new THREE.MeshStandardMaterial({ color: 0x8a7d68, roughness: 0.75, emissive: 0x332b1e, emissiveIntensity: 0.3 });
        var box = new THREE.Mesh(new THREE.BoxGeometry(3.2, 2.2, 2.2), mat);
        box.position.set(x, LINE_Y, 0);
        group.add(box);
        var edge = new THREE.LineSegments(
          new THREE.EdgesGeometry(box.geometry),
          new THREE.LineBasicMaterial({ color: 0xa89a83, transparent: true, opacity: 0.5 })
        );
        edge.position.copy(box.position);
        group.add(edge);
        var label = textSprite(labelText, 40, '#c9bba6', 4.6, 2.3);
        label.position.set(x, LINE_Y + 2.2, 0);
        group.add(label);
        tower.add(group);
        reg(type, group, [box], new THREE.Vector3(x, LINE_Y, 0), 2.6, function () {
          pulseAnim(function (k) { mat.emissiveIntensity = 0.3 + 1.3 * k; }, 0.7);
        });
      })(comps[i], i);
    }

    // 水平引导线
    var lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(x0, LINE_Y - 1.4, 0), new THREE.Vector3(x1, LINE_Y - 1.4, 0)
    ]);
    tower.add(new THREE.Line(lineGeo, new THREE.LineBasicMaterial({
      color: 0xffd9a0, transparent: true, opacity: 0.25, blending: THREE.AdditiveBlending, depthWrite: false
    })));

    // flow 粒子路径 = 这条水平线（闭环回到起点）
    s.flowScene = [
      [x0, LINE_Y, 0], [x1, LINE_Y, 0], [x1 + 3, LINE_Y - 2, 4], [x0 - 3, LINE_Y - 2, 4]
    ];
    s.towerH = LINE_Y + 4;
    s.scene.add(tower);

    s.home = { pos: [0, 10, 26], target: [0, LINE_Y, 0] };
    if (!s._builtOnce && E.camera && E.camera.setPose) E.camera.setPose(s.home.pos, s.home.target);
    s._builtOnce = true;
    if (E.flow && E.flow._rebuild) E.flow._rebuild();
  }

  // 包装 buildScene：CV 走横向流水线，其余保持原逻辑
  var origBuildScene = E.buildScene;
  E.buildScene = function (spec) {
    var s = S();
    if (!s.scene) { console.warn('[engine] 请先调用 ENGINE.init(canvas)'); return; }
    // 直接切模型时退出对比态（compare 内部走 origBuildScene，不受影响）
    if (s._towerA) { s.scene.remove(s._towerA); disposeObj(s._towerA); s._towerA = null; }
    if (spec && spec.category === 'cv') {
      if (typeof window.validateSpec === 'function') {
        var v = window.validateSpec(spec);
        if (v) spec = v;
      }
      buildCv(spec);
      return;
    }
    return origBuildScene(spec);
  };
})();
