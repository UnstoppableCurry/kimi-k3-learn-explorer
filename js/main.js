/* ============ LLM/CV 架构 3D 探索器 · 装配层 v2 ============ */
/* 职责：初始化引擎、模型切换（LLM/CV 分组下拉）、点击取件、导览控制、
   阶段切换（推理/训练）、模型对比、采样控制、面板渲染 */
(function () {
  'use strict';

  /* ---------- 组件 type → 中文名 / 语义色（与契约视觉语言一致） ---------- */
  var TYPE_META = {
    tokenizer: { label: '分词器',          color: '#a89a83' },
    bpe:       { label: 'BPE 分词',        color: '#a89a83' },
    embedding: { label: '嵌入层',          color: '#a89a83' },
    rope:      { label: 'RoPE 旋转位置编码', color: '#7aa2f7' },
    kda:       { label: 'KDA 线性注意力',  color: '#e8a94d' },
    mla:       { label: 'MLA 潜注意力',    color: '#7aa2f7' },
    attn:      { label: '全注意力',        color: '#7aa2f7' },
    'lightning-attn': { label: 'Lightning 线性注意力', color: '#e8a94d' },
    'nsa-attn': { label: 'NSA 稀疏注意力', color: '#7aa2f7' },
    router:    { label: '专家路由',        color: '#46c8ae' },
    gate:      { label: '门控',            color: '#46c8ae' },
    experts:   { label: 'MoE 专家层',      color: '#46c8ae' },
    aggregate: { label: '专家聚合',        color: '#46c8ae' },
    attnres:   { label: 'AttnRes 注意力残差', color: '#e0699b' },
    norm:      { label: '归一化',          color: '#a89a83' },
    output:    { label: '输出头',          color: '#a89a83' },
    softmax:   { label: 'Softmax 采样',    color: '#a89a83' },
    sampler:   { label: '采样器',          color: '#e8a94d' },
    training:  { label: '训练流程',        color: '#e0699b' },
    'vit-patch':   { label: 'ViT 图像分块',   color: '#7aa2f7' },
    'yolo-head':   { label: 'YOLO 检测头',    color: '#e8a94d' },
    'sam-decoder': { label: 'SAM 掩码解码器', color: '#46c8ae' }
  };

  var TAG_CHIPS = {
    official:   '<span class="chip chip-official">官方</span>',
    inferred:   '<span class="chip chip-inferred">待确认</span>',
    simplified: '<span class="chip chip-simplified">简化</span>',
    estimated:  '<span class="chip chip-estimated">估算</span>'
  };

  /* ---------- 模型分组与排序（聚焦版：只做 K3 / K2） ---------- */
  var GROUPS = [
    { key: 'llm', label: '语言模型', order: ['k3', 'k2'] }
  ];

  /* ---------- 硬编码：v1 模型 overview 兜底文案（CONTENT_MODELS 缺失时使用） ---------- */
  var OVERVIEWS = {
    k3: {
      name: 'Kimi K3',
      stats: { total: '2.8T', active: '16/896 专家', context: '1M', quant: 'MXFP4' },
      notes: [
        '开源状态：开放权重（以官方模型卡为准）',
        '基准模型：KDA 线性注意力 + MLA 全注意力混合架构',
        'MoE 896 专家选 16，含 AttnRes 注意力残差结构，1M 上下文'
      ]
    },
    k2: {
      name: 'Kimi K2',
      stats: { total: '约 1T', active: '32B', context: '128K', quant: '—' },
      notes: [
        '开源状态：开放权重（Modified MIT）',
        '与 K3 最大差异：无 KDA 线性注意力，全部层使用 MLA 全注意力',
        '无 AttnRes 结构 —— 对比 K3 时可直观看到这两类组件缺失'
      ]
    },
    glm52: {
      name: 'GLM-5.2',
      stats: { total: '约 700B（估算）', active: '约 40B（估算）', context: '200K', quant: '—' },
      notes: [
        '开源状态：开放权重（数值为估算，待官方确认）',
        '与 K3 最大差异：无 KDA，使用标准全注意力；无 AttnRes 结构',
        '本模型结构为社区估算，橙色「待确认」标记的内容请谨慎采信'
      ]
    },
    minimax3: {
      name: 'MiniMax M3',
      stats: { total: '约 2.7T（估算）', active: '约 50B（估算）', context: '1M', quant: '—' },
      notes: [
        '开源状态：开放权重（数值为估算，待官方确认）',
        '与 K3 最大差异：以 Lightning Attention 类线性注意力替代 KDA',
        '与 K3 同样走 1M 超长上下文路线，但线性注意力实现不同'
      ]
    },
    deepseekv4: {
      name: 'DeepSeek V4',
      stats: { total: '约 1.6T（估算）', active: '约 49B（估算）', context: '1M', quant: '—' },
      notes: [
        '开源状态：开放权重（数值为估算，待官方确认）',
        '与 K3 最大差异：保留 MLA 并引入 NSA 稀疏注意力，但无 KDA',
        '无 AttnRes 结构'
      ]
    }
  };

  /* ---------- 状态 ---------- */
  var currentModel = 'k3';
  var currentSpec = null;
  var tourState = 'stopped';   // stopped | playing | paused
  var flowOn = false;
  var viewMode = 'full';       // full（全景塔） | detail（细节）——默认全景，整塔权重可见

  /* ---------- 全景塔 / 细节 双模式 ---------- */
  function fullAvailable() { return !!(window.FullTower && typeof FullTower.build === 'function'); }

  function enterFull() {
    if (!fullAvailable() || !currentSpec) return;
    if (!FullTower.isActive()) FullTower.build(currentSpec);
    viewMode = 'full';
    syncViewModeUI();
  }

  function enterDetail(focusId) {
    if (fullAvailable() && FullTower.isActive()) FullTower.dispose(); // dispose 自动还原细节场景与拾取表
    viewMode = 'detail';
    syncViewModeUI();
    if (focusId) {
      var cam = engine().camera;
      if (cam && typeof cam.focusOn === 'function') {
        try { cam.focusOn(focusId); } catch (e) { /* 聚焦失败不阻塞 */ }
      }
    }
  }

  function syncViewModeUI() {
    var bf = $('btnViewFull'), bd = $('btnViewDetail');
    if (bf) bf.classList.toggle('active', viewMode === 'full');
    if (bd) bd.classList.toggle('active', viewMode === 'detail');
  }
  var subtitleEnabled = true;
  var stageMode = 'infer';     // infer | train
  var trainStage = 'pretrain'; // pretrain | sft | rl
  var compareOn = false;
  var compareTarget = null;
  // 引擎 v2 API 能力探测（boot 时确定）
  var hasSetStage = false;
  var hasCompare = false;
  var hasSetSampler = false;

  /* ---------- DOM ---------- */
  var $ = function (id) { return document.getElementById(id); };
  var panelBody, subtitleBar, subtitleText;
  var modelSelect, compareSelect, samplerGroup, btnCompare;

  /* ---------- 容错工具 ---------- */
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  // 引擎可能尚未就绪（文件未加载完），全部调用走安全包装
  function engine() { return window.ENGINE || {}; }
  function call(name, args) {
    var fn = engine()[name];
    if (typeof fn === 'function') {
      try { return fn.apply(engine(), args || []); }
      catch (e) { console.warn('[shell] ENGINE.' + name + ' 调用失败', e); }
    }
  }
  function callObj(obj, name, args) {
    if (obj && typeof obj[name] === 'function') {
      try { return obj[name].apply(obj, args || []); }
      catch (e) { console.warn('[shell] ' + name + ' 调用失败', e); }
    }
  }

  /* ---------- 模型切换 ---------- */
  function getSpec(id) {
    var specs = window.MODEL_SPECS || {};
    var spec = specs[id];
    if (!spec) { console.warn('[shell] MODEL_SPECS 缺少模型: ' + id); return null; }
    if (typeof window.validateSpec === 'function') {
      var v = window.validateSpec(spec);
      if (v) spec = v;   // validateSpec 可能返回修补后的副本，也可能原地补默认值
    }
    return spec;
  }

  // 按 GROUPS 把 MODEL_SPECS 里的模型分桶排序
  function modelGroups() {
    var specs = window.MODEL_SPECS || {};
    var buckets = { llm: [], cv: [] };
    Object.keys(specs).forEach(function (id) {
      var cat = (specs[id] && specs[id].category === 'cv') ? 'cv' : 'llm';
      buckets[cat].push(id);
    });
    return GROUPS.map(function (g) {
      g.order.forEach(function () {});
      var ids = buckets[g.key].slice().sort(function (a, b) {
        var ia = g.order.indexOf(a), ib = g.order.indexOf(b);
        if (ia === -1 && ib === -1) return a < b ? -1 : 1;
        if (ia === -1) return 1;
        if (ib === -1) return -1;
        return ia - ib;
      });
      return { label: g.label, ids: ids };
    });
  }

  // 用 optgroup 填充下拉；excludeId 用于对比下拉剔除当前主模型
  function fillSelect(sel, groups, excludeId) {
    sel.innerHTML = '';
    groups.forEach(function (g) {
      var og = document.createElement('optgroup');
      og.label = g.label;
      g.ids.forEach(function (id) {
        if (excludeId && id === excludeId) return;
        var spec = (window.MODEL_SPECS || {})[id] || {};
        var opt = document.createElement('option');
        opt.value = id;
        opt.textContent = (spec.name || id) + (spec.sourceTag === 'estimated' ? '（估算）' : '');
        og.appendChild(opt);
      });
      if (og.children.length) sel.appendChild(og);
    });
  }

  function switchModel(id) {
    var spec = getSpec(id);
    if (!spec) return;
    currentModel = id;
    currentSpec = spec;
    if (window.FullTower && FullTower.isActive()) FullTower.dispose(); // 换模型前先退出全塔，避免快照失效
    call('buildScene', [spec]);
    if (viewMode === 'full' && window.FullTower) FullTower.build(spec); // 全景模式下换模型直接重建全塔
    renderOverview(spec);
    if (modelSelect && modelSelect.value !== id) modelSelect.value = id;
    syncSamplerFromSpec(spec);
    applyStage();            // 场景重建后重放阶段状态
    refreshCompareOptions(); // 对比下拉剔除新的主模型
    applyCompare();
  }

  /* ---------- v2：阶段切换（推理 / 训练 pretrain·sft·rl） ---------- */
  function applyStage() {
    if (!hasSetStage) return;
    call('setStage', [stageMode === 'train' ? trainStage : 'off']);
  }

  function updateStageUI() {
    $('btnStageInfer').classList.toggle('active', stageMode === 'infer');
    $('btnStageTrain').classList.toggle('active', stageMode === 'train');
    var tabs = $('stageSubTabs');
    tabs.classList.toggle('v2-hidden', stageMode !== 'train');
    var subs = tabs.querySelectorAll('[data-stage]');
    for (var i = 0; i < subs.length; i++) {
      subs[i].classList.toggle('active', subs[i].getAttribute('data-stage') === trainStage);
    }
    // 采样组只在推理模式显示（且引擎支持 setSampler）
    samplerGroup.classList.toggle('v2-hidden', !(hasSetSampler && stageMode === 'infer'));
  }

  function bindStage() {
    $('btnStageInfer').addEventListener('click', function () {
      if (stageMode === 'infer') return;
      stageMode = 'infer';
      updateStageUI();
      applyStage();
      onSamplerInput();      // 回到推理模式时恢复采样参数
      callCameraReset();     // 相机回到推理塔默认视角
    });
    $('btnStageTrain').addEventListener('click', function () {
      if (stageMode === 'train') return;
      stageMode = 'train';
      updateStageUI();
      applyStage();
    });
    var subs = $('stageSubTabs').querySelectorAll('[data-stage]');
    for (var i = 0; i < subs.length; i++) {
      (function (btn) {
        btn.addEventListener('click', function () {
          trainStage = btn.getAttribute('data-stage');
          updateStageUI();
          applyStage();
        });
      })(subs[i]);
    }
    var btnTrainPlay = $('btnTrainPlay');
    if (btnTrainPlay) {
      btnTrainPlay.addEventListener('click', function () {
        if (stageMode !== 'train') { stageMode = 'train'; updateStageUI(); }
        call('playTrainingTimeline');
      });
    }
  }

  /* ---------- v2：模型对比 ---------- */
  function refreshCompareOptions() {
    if (!hasCompare) return;
    var prev = compareTarget;
    fillSelect(compareSelect, modelGroups(), currentModel);
    if (prev && prev !== currentModel && (window.MODEL_SPECS || {})[prev]) {
      compareSelect.value = prev;
    }
    compareTarget = compareSelect.value || null;
  }

  function applyCompare() {
    if (!hasCompare) return;
    if (compareOn && compareTarget && compareTarget !== currentModel) {
      call('compare', [currentModel, compareTarget]);
    } else {
      call('single', [currentModel]);
    }
  }

  function bindCompare() {
    btnCompare.addEventListener('click', function () {
      compareOn = !compareOn;
      btnCompare.classList.toggle('on', compareOn);
      compareSelect.classList.toggle('v2-hidden', !compareOn);
      if (compareOn && !compareTarget && compareSelect.options.length) {
        compareSelect.selectedIndex = 0;
        compareTarget = compareSelect.value;
      }
      applyCompare();
    });
    compareSelect.addEventListener('change', function () {
      compareTarget = this.value;
      if (compareOn) applyCompare();
    });
  }

  /* ---------- v2：采样控制（temperature / top-p） ---------- */
  function readSampler() {
    return { temperature: parseFloat($('sldTemp').value), topP: parseFloat($('sldTopP').value) };
  }

  function onSamplerInput() {
    $('valTemp').textContent = parseFloat($('sldTemp').value).toFixed(2);
    $('valTopP').textContent = parseFloat($('sldTopP').value).toFixed(2);
    if (hasSetSampler) call('setSampler', [readSampler()]);
  }

  // 模型自带 sampler 默认值时同步到滑块
  function syncSamplerFromSpec(spec) {
    var s = spec.sampler || {};
    if (typeof s.temperature === 'number') $('sldTemp').value = s.temperature;
    if (typeof s.topP === 'number') $('sldTopP').value = s.topP;
    $('valTemp').textContent = parseFloat($('sldTemp').value).toFixed(2);
    $('valTopP').textContent = parseFloat($('sldTopP').value).toFixed(2);
    if (hasSetSampler && stageMode === 'infer') call('setSampler', [readSampler()]);
  }

  function bindSampler() {
    $('sldTemp').addEventListener('input', onSamplerInput);
    $('sldTopP').addEventListener('input', onSamplerInput);
  }

  /* ---------- 面板：模型 overview 卡 ---------- */
  function renderOverview(spec) {
    var id = spec.id || currentModel;
    var mc = (window.CONTENT_MODELS || {})[id] || null;   // v2 模型卡
    var ov = OVERVIEWS[id] || { name: spec.name || id, stats: {}, notes: [] };
    var p = spec.params || {};
    var stats = {
      total:   p.total   || ov.stats.total   || '—',
      active:  p.active  || ov.stats.active  || '—',
      context: spec.context || ov.stats.context || '—',
      quant:   spec.quant || ov.stats.quant || '—'
    };
    var srcChip = spec.sourceTag === 'estimated' ? TAG_CHIPS.estimated : TAG_CHIPS.official;

    var html = '';
    html += '<div class="p-head"><span class="p-title">' + esc(spec.name || ov.name) + '</span>' + srcChip + '</div>';

    if (mc && mc.overview) {
      html += '<p class="layer-one">' + esc(mc.overview) + '</p>';
    } else {
      html += '<p class="p-sub">点击场景中的组件可查看逐层讲解</p>';
    }

    html += '<div class="card"><h3>关键指标</h3><div class="stat-grid">'
      + stat('总参数量', stats.total)
      + stat('激活参数', stats.active)
      + stat('上下文', fmtCtx(stats.context))
      + stat('量化', stats.quant)
      + '</div></div>';

    // v2：facts 档案表（每行 ['指标','值','official'|'estimated']，带 chip）
    if (mc && mc.facts && mc.facts.length) {
      html += '<div class="card"><h3>模型档案</h3><table class="facts-table">'
        + mc.facts.map(function (f) {
            var chip = f[2] === 'official' ? TAG_CHIPS.official : TAG_CHIPS.estimated;
            return '<tr><td>' + esc(f[0]) + '</td><td>' + esc(f[1]) + '</td><td>' + chip + '</td></tr>';
          }).join('')
        + '</table></div>';
    }

    html += '<div class="card"><h3>架构要点</h3>' + archFlow(spec) + '</div>';

    if (mc && mc.strengths && mc.strengths.length) {
      html += '<div class="card"><h3>优势</h3><ul class="notes">'
        + mc.strengths.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('')
        + '</ul></div>';
    }
    if (mc && mc.vs_k3) {
      html += '<div class="card"><h3>对比 K3</h3><p class="card-text">' + esc(mc.vs_k3) + '</p></div>';
    }

    // v2：训练阶段说明
    if (spec.training && spec.training.stages && spec.training.stages.length) {
      html += '<div class="card"><h3>训练阶段</h3><ul class="notes">'
        + spec.training.stages.map(function (st) {
            return '<li><b>' + esc(st.id) + '</b> — ' + esc(st.desc) + '</li>';
          }).join('')
        + '</ul></div>';
    }

    // v1 兜底：无 CONTENT_MODELS 时用硬编码 notes
    if (!mc && ov.notes.length) {
      html += '<div class="card"><h3>说明</h3><ul class="notes">'
        + ov.notes.map(function (n) { return '<li>' + esc(n) + '</li>'; }).join('')
        + '</ul></div>';
    }
    panelBody.innerHTML = html;
  }

  function stat(k, v) {
    return '<div class="stat"><div class="k">' + esc(k) + '</div><div class="v">' + esc(v) + '</div></div>';
  }
  function fmtCtx(c) {
    if (typeof c !== 'number') return c;
    if (c % 1000000 === 0) return (c / 1000000) + 'M';   // 1000000 → 1M
    if (c % 1000 === 0) return (c / 1000) + 'K';         // 200000 → 200K
    if (c >= 1024) return Math.round(c / 1024) + 'K';    // 131072 → 128K
    return String(c);
  }

  // 组件 type 元信息；CV 模型可在 spec.cv.components 里自带 label
  function typeMeta(t) {
    if (TYPE_META[t]) return TYPE_META[t];
    var cv = currentSpec && currentSpec.cv && currentSpec.cv.components;
    if (cv) {
      for (var i = 0; i < cv.length; i++) {
        if (cv[i] && cv[i].type === t && cv[i].label) {
          return { label: cv[i].label, color: '#a89a83' };
        }
      }
    }
    return { label: t, color: '#a89a83' };
  }

  // 架构组件序列：列出 spec.layers（CV 模型退回 spec.cv.components）出现的组件；
  // K3 有而本模型没有的划线标出（架构差异可视化，仅 LLM 组参与对比）
  function archFlow(spec) {
    var seen = {};
    var order = [];
    var layers = spec.layers || (spec.cv && spec.cv.components) || [];
    layers.forEach(function (l) {
      if (l && l.type && !seen[l.type]) { seen[l.type] = true; order.push(l.type); }
    });
    if (!order.length) return '<p class="p-sub">层结构数据待补充</p>';

    var html = '<div class="arch-flow">';
    order.forEach(function (t, i) {
      var m = typeMeta(t);
      if (i) html += '<span class="arch-arrow">→</span>';
      html += '<span class="arch-node" style="border-color:' + m.color + '55;color:' + m.color + '">'
        + esc(m.label) + '</span>';
    });
    // 与 K3 对比缺失项（仅 LLM 组）
    var isLlm = (spec.category || 'llm') === 'llm';
    if (isLlm && spec.id !== 'k3' && window.MODEL_SPECS && window.MODEL_SPECS.k3) {
      var k3seen = {};
      (window.MODEL_SPECS.k3.layers || []).forEach(function (l) { if (l && l.type) k3seen[l.type] = true; });
      Object.keys(k3seen).forEach(function (t) {
        if (!seen[t]) {
          var m = typeMeta(t);
          html += '<span class="arch-node missing" title="K3 有而本模型无" style="color:' + m.color + '">'
            + esc(m.label) + '</span>';
        }
      });
    }
    return html + '</div>';
  }

  /* ---------- 面板：组件四层讲解 ---------- */
  function showComponent(componentId) {
    var m = typeMeta(componentId);
    var contents = window.CONTENT_COMPONENTS || {};
    var c = contents[componentId];

    var html = '<div class="p-head">'
      + '<span class="p-dot" style="background:' + m.color + ';color:' + m.color + '"></span>'
      + '<span class="p-title" style="color:' + m.color + '">' + esc(m.label) + '</span>'
      + (c && TAG_CHIPS[c.tag] ? TAG_CHIPS[c.tag] : '')
      + '</div>';

    if (!c) {
      // 容错：内容缺失 → 待补充提示 + 回退到模型 overview
      html += '<div class="empty-note">「' + esc(m.label) + '」内容待补充</div>';
      panelBody.innerHTML = html;
      var spec = getSpec(currentModel);
      if (spec) {
        var tmp = document.createElement('div');
        panelBody.appendChild(tmp);
        renderOverviewInto(spec, tmp);
      }
      return;
    }

    // 第一层：一句话（常显）
    if (c.one) html += '<p class="layer-one">' + esc(c.one) + '</p>';
    // 第二~四层：类比 / 原理 / 公式，默认折叠
    if (c.analogy)   html += layer('analogy',   '打个比方', c.analogy);
    if (c.principle) html += layer('principle', '工作原理', c.principle);
    if (c.formula)   html += layer('formula',   '数学表达', c.formula);
    // 官方原文引用
    if (c.quote) {
      html += '<blockquote class="quote">' + esc(c.quote)
        + (c.quoteUrl ? '<a href="' + esc(c.quoteUrl) + '" target="_blank" rel="noopener">原文 ↗</a>' : '')
        + '</blockquote>';
    }
    panelBody.innerHTML = html;
    panelBody.parentElement.scrollTop = 0;
  }

  function layer(cls, title, body) {
    return '<details class="layer layer-' + cls + '"><summary>' + esc(title) + '</summary>'
      + '<div class="layer-body">' + esc(body) + '</div></details>';
  }

  // 在指定容器里渲染 overview（供「待补充」回退拼接用）
  function renderOverviewInto(spec, container) {
    var keep = panelBody;
    panelBody = container;
    renderOverview(spec);
    panelBody = keep;
  }

  /* ---------- 面包屑：pipeline 位置感 ---------- */
  // 提升为 IIFE 顶层：tourHooks 与 boot 内的事件回调都要引用（此前定义在 boot 里，onActStart 调用即 ReferenceError）
  var STAGES = ['输入', '嵌入', '注意力', 'MoE 路由→聚合', '残差', '输出/采样', '自回归'];
  var TYPE2STAGE = {
    tokenizer: 0, bpe: 0, embedding: 1, rope: 1,
    kda: 2, mla: 2, attn: 2, router: 3, gate: 3, experts: 3, aggregate: 3,
    attnres: 4, norm: 4, output: 5, softmax: 5, sampler: 5, autoregressive: 6
  };
  var FX2TYPE = { tokenize: 'tokenizer', rope: 'rope', gate: 'router', aggregate: 'aggregate', sample: 'softmax', autoregress: 'autoregressive' };
  function crumbTo(type) {
    var el = $('crumb');
    if (!el) return;
    var cur = TYPE2STAGE[type];
    if (cur == null) { el.innerHTML = ''; return; }
    el.innerHTML = STAGES.map(function (s, i) {
      return '<span class="crumb-node' + (i === cur ? ' on' : (i < cur ? ' past' : '')) + '">' + s + '</span>';
    }).join('<span class="crumb-sep">→</span>');
  }

  /* ---------- 导览（tour） ---------- */
  var tourHooks = {
    onSubtitle: function (text) {
      if (!subtitleEnabled) return;
      subtitleText.textContent = text || '';
      subtitleBar.classList.remove('hidden');
    },
    onActStart: function (i) {
      var acts = window.CONTENT_TOUR || [];
      var act = acts[i];
      // 优先按本幕特效映射阶段（如 autoregress 幕 focusComponent 是 output，但应高亮『自回归』）
      var crumbType = act && ((act.effect && FX2TYPE[act.effect]) || act.focusComponent);
      if (crumbType) crumbTo(crumbType);
      // act.switchModel：需要换模型时由外壳重建场景（引擎不管模型数据）
      if (act && act.switchModel && act.switchModel !== currentModel) {
        switchModel(act.switchModel);
      }
    }
  };

  function setTourUI(state) {
    tourState = state;
    $('btnTourPlay').disabled = (state === 'playing');
    $('btnTourPause').disabled = (state === 'stopped');
    $('btnTourStop').disabled = (state === 'stopped');
    $('btnTourPlay').classList.toggle('playing', state === 'playing');
    $('btnTourPause').textContent = (state === 'paused') ? '▶ 继续' : '⏸ 暂停';
  }

  function bindTour() {
    $('btnTourPlay').addEventListener('click', function () {
      var acts = window.CONTENT_TOUR || [];
      if (!acts.length) { console.warn('[shell] CONTENT_TOUR 为空'); return; }
      if (fullAvailable() && FullTower.isActive()) enterDetail(); // 导览基于细节场景
      subtitleEnabled = true;
      callObj(engine().tour, 'stop');           // 防止重复播放叠加
      callObj(engine().tour, 'play', [acts, tourHooks]);
      setTourUI('playing');
    });
    $('btnTourPause').addEventListener('click', function () {
      if (tourState === 'playing') {
        callObj(engine().tour, 'pause');
        setTourUI('paused');
      } else if (tourState === 'paused') {
        callObj(engine().tour, 'resume');
        setTourUI('playing');
      }
    });
    $('btnTourStep').addEventListener('click', function () {
      callObj(engine().tour, 'step');
    });
    $('btnTourStop').addEventListener('click', function () {
      callObj(engine().tour, 'stop');
      setTourUI('stopped');
      subtitleBar.classList.add('hidden');
      callCameraReset();
    });
  }

  function callCameraReset() {
    var cam = engine().camera;
    if (cam && typeof cam.reset === 'function') {
      try { cam.reset(); } catch (e) { console.warn('[shell] camera.reset 失败', e); }
    }
  }

  /* ---------- 视图控制 ---------- */
  function bindView() {
    if ($('btnViewFull')) $('btnViewFull').addEventListener('click', enterFull);
    if ($('btnViewDetail')) $('btnViewDetail').addEventListener('click', function () { enterDetail(); });
    $('btnFlow').addEventListener('click', function () {
      flowOn = !flowOn;
      var f = engine().flow;
      if (f) callObj(f, flowOn ? 'start' : 'stop');
      this.classList.toggle('on', flowOn);
    });
    $('btnCamReset').addEventListener('click', callCameraReset);
    $('subtitleClose').addEventListener('click', function () {
      subtitleEnabled = false;
      subtitleBar.classList.add('hidden');
    });
  }

  /* ---------- 启动 ---------- */
  function boot() {
    panelBody = $('panelBody');
    subtitleBar = $('subtitleBar');
    subtitleText = $('subtitleText');
    modelSelect = $('modelSelect');
    compareSelect = $('compareSelect');
    samplerGroup = $('samplerGroup');
    btnCompare = $('btnCompare');

    call('init', [$('scene')]);

    // v2 引擎 API 能力探测：不存在的 API 对应控件整组隐藏
    hasSetStage = typeof engine().setStage === 'function';
    hasCompare = typeof engine().compare === 'function' && typeof engine().single === 'function';
    hasSetSampler = typeof engine().setSampler === 'function';
    // 聚焦版：保留训练阶段入口，下架对比模式
    $('compareGroup').classList.add('v2-hidden');

    // 模型分组下拉（按 MODEL_SPECS 动态填充）
    fillSelect(modelSelect, modelGroups(), null);
    if (!modelSelect.children.length) modelSelect.classList.add('v2-hidden');
    modelSelect.addEventListener('change', function () { switchModel(this.value); });

    bindStage();
    bindCompare();
    bindSampler();
    updateStageUI();

    // 点击组件 → 面板讲解 + 相机聚焦 + 联动数据流动画
    var FX_ON_PICK = {
      tokenizer: 'tokenize', embedding: 'rope',
      softmax: 'sample', output: 'sample',
      kda: 'kda-write', attnres: 'attnres'
    };
    var eng = engine();
    if (typeof eng.onPick === 'function') {
      eng.onPick(function (componentId, hit) {
        crumbTo(componentId); // 面包屑同步（全景/细节两模式通用）
        // 全景模式：原地缩放查看（不跳模式）—— 相机飞到被点结构旁，可继续自由环绕/缩放
        if (fullAvailable() && FullTower.isActive()) {
          showComponent(componentId);
          if (hit && hit.point) {
            var p = hit.point, camF = engine().camera;
            if (camF && typeof camF.flyTo === 'function') {
              try {
                camF.flyTo([p.x + 42, Math.max(p.y, 8) + 32, p.z + 48],
                           [p.x, Math.max(p.y * 0.5, 5), p.z], 1.4);
              } catch (e) { /* 飞行失败不阻塞面板 */ }
            }
          }
          return;
        }
        showComponent(componentId);
        var cam = engine().camera;
        if (cam && typeof cam.focusOn === 'function') {
          try { cam.focusOn(componentId); } catch (e) { /* 聚焦失败不阻塞面板 */ }
        }
        // 联动演示：router/experts 走完整 打分→聚合 链，其余组件映射单个特效
        if (componentId === 'router' || componentId === 'experts' || componentId === 'gate' || componentId === 'aggregate') {
          call('playEffect', ['gate']);
          setTimeout(function () { call('playEffect', ['aggregate']); }, 2300);
        } else if (FX_ON_PICK[componentId]) {
          call('playEffect', [FX_ON_PICK[componentId]]);
        }
      });
    }

    // 数据流演示按钮组（点击直接触发对应动画）
    var fxBtns = document.querySelectorAll('[data-fx]');
    for (var fi = 0; fi < fxBtns.length; fi++) {
      (function (btn) {
        btn.addEventListener('click', function () {
          var fx = btn.getAttribute('data-fx');
          if (fullAvailable() && FullTower.isActive()) enterDetail(); // 演示动画只在细节模式播
          call('playEffect', [fx]);
          if (fx === 'gate') setTimeout(function () { call('playEffect', ['aggregate']); }, 2300);
          crumbTo(FX2TYPE[fx]);
        });
      })(fxBtns[fi]);
    }

    // token 单步推进（真实数值逐 token 流动）；面包屑高亮『注意力』（KDA 是当前 token 流经的核心层）
    var btnPrev = $('btnTokPrev'), btnNext = $('btnTokNext');
    if (btnPrev) btnPrev.addEventListener('click', function () { call('simStep', [-1]); crumbTo('kda'); });
    if (btnNext) btnNext.addEventListener('click', function () { call('simStep', [1]); crumbTo('kda'); });

    bindTour();
    bindView();
    setTourUI('stopped');

    // 初始模型：K3（缺失时退到列表第一个）；默认进入全景塔模式
    var initial = (window.MODEL_SPECS && window.MODEL_SPECS.k3) ? 'k3' : modelSelect.value;
    if (initial) switchModel(initial);
    if (viewMode === 'full') enterFull();
    syncViewModeUI();

    // 聚焦版：加载后自动开播旅程（不点任何按钮也能看懂）；导览会自动切入细节模式
    setTimeout(function () {
      var acts = window.CONTENT_TOUR || [];
      if (!acts.length || tourState !== 'stopped') return;
      if (fullAvailable() && FullTower.isActive()) enterDetail();
      subtitleEnabled = true;
      callObj(engine().tour, 'play', [acts, tourHooks]);
      setTourUI('playing');
    }, 1600);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
