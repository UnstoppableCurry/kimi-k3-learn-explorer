// models/kimi-k2.js — Kimi K2（官方）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.k2 = {
    id: 'k2',
    name: 'Kimi K2',
    sourceTag: 'official',
    params: { total: '~1T', active: '32B' },
    context: 131072,
    blockCount: 61, // 官方 61 层（含 1 个 dense 层）
    d_model: 7168,
    // 无 kda、无 attnres —— 与 K3 的架构差异点，场景自动缺失
    layers: [
      { type: 'mla' },
      { type: 'router' },
      { type: 'experts' },
      { type: 'norm' },
      { type: 'output' }
    ],
    moe: { experts: 384, active: 8, shared: 1 },
    quant: null, // 官方未做权重量化
    // ---- v2 字段 ----
    category: 'llm',
    posEncoding: { type: 'rope' }, // 官方：RoPE
    sampler: { temperature: 1.0, topP: 0.95 }, // 估算：官方未公布，沿用通用默认
    training: {
      stages: [
        { id: 'pretrain', desc: '大规模 token 预训练（估算：官方报告有描述，细节未逐项核验）' },
        { id: 'sft', desc: '监督微调（估算）' },
        { id: 'rl', desc: 'agentic 强化学习（估算）' }
      ]
    },
    flow: [
      [0, 0, 0],     // 输入
      [0, 1.5, 0],   // mla
      [0, 3, 0],     // router
      [2.4, 4.5, 0], // 路由到专家（右）
      [-2.4, 4.5, 0],// 扫过专家（左）
      [0, 4.5, 0],   // 聚合回主干
      [0, 6, 0],     // norm
      [0, 7.5, 0]    // output
    ],
    meta: {
      notes: [
        '官方（K2 技术报告）：total ~1T（1.04T）、active 32B、384 专家/激活 8+1 共享、d_model 7168、context 131072、blockCount 61',
        'quant 为 null：官方未发布量化方案',
        '无 kda / attnres 组件：纯 MLA + MoE 架构，这是与 K3 对比可视化的关键差异'
      ]
    }
  };
})();
