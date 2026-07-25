// models/glm-52.js — GLM-5.2（估算）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.glm52 = {
    id: 'glm52',
    name: 'GLM-5.2',
    sourceTag: 'estimated',
    params: { total: '~700B（估算）', active: '~40B（估算）' },
    context: 200000, // 估算
    blockCount: 92,  // 估算（沿 GLM-4.5 量级）
    d_model: 5120,   // 估算
    // attn 为占位类型：dense（GQA）注意力，非 MLA
    layers: [
      { type: 'attn' },
      { type: 'router' },
      { type: 'experts' },
      { type: 'norm' },
      { type: 'output' },
      { type: 'softmax' }
    ],
    moe: { experts: 256, active: 8, shared: 1 },
    quant: null,
    // ---- v2 字段 ----
    category: 'llm',
    posEncoding: { type: 'rope' }, // 官方：GLM 系列使用 RoPE
    sampler: { temperature: 1.0, topP: 0.95 }, // 估算
    training: {
      stages: [
        { id: 'pretrain', desc: '大规模预训练（估算）' },
        { id: 'sft', desc: '监督微调（估算）' },
        { id: 'rl', desc: '强化学习对齐（估算）' }
      ]
    },
    flow: [
      [0, 0, 0],     // 输入
      [0, 1.5, 0],   // attn（dense 注意力）
      [0, 3, 0],     // router
      [2.4, 4.5, 0], // 路由到专家（右）
      [-2.4, 4.5, 0],// 扫过专家（左）
      [0, 4.5, 0],   // 聚合回主干
      [0, 6, 0],     // norm
      [0, 7.5, 0],   // output
      [0, 9, 0]      // softmax 输出
    ],
    meta: {
      notes: [
        '全部数字为估算（sourceTag: estimated），官方未公开完整配置',
        'attn 是约定占位类型：表示 dense（GQA）注意力，engine 按通用注意力渲染即可',
        '估算依据：~700B 总参 / ~40B 激活 / 256 专家 / 92 层 / context 200K，按 GLM-4.5/GLM-5 前代外推',
        '无 kda / attnres：dense attention + MoE 的传统结构'
      ]
    }
  };
})();
