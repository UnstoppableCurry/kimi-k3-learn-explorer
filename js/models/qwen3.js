// models/qwen3.js — Qwen3（估算）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.qwen3 = {
    id: 'qwen3',
    name: 'Qwen3',
    sourceTag: 'estimated',
    params: { total: '235B', active: '22B' }, // 官方 235B-A22B 旗舰档
    context: 262144, // 估算（YaRN 扩展后）
    blockCount: 94,  // 官方 235B 档：94 层
    d_model: 4096,   // 估算
    layers: [
      { type: 'attn' },   // GQA dense 注意力
      { type: 'router' },
      { type: 'experts' },
      { type: 'norm' },
      { type: 'output' },
      { type: 'softmax' }
    ],
    moe: { experts: 128, active: 8, shared: 0 }, // 官方：128 专家激活 8，无共享专家
    quant: null,
    // ---- v2 字段 ----
    category: 'llm',
    posEncoding: { type: 'rope' }, // 官方：RoPE
    sampler: { temperature: 1.0, topP: 0.95 }, // 估算
    training: {
      stages: [
        { id: 'pretrain', desc: '约 36T token 预训练（估算：官方披露量级）' },
        { id: 'sft', desc: '监督微调 + 思维模式蒸馏（估算）' },
        { id: 'rl', desc: '强化学习对齐（估算）' }
      ]
    },
    flow: [
      [0, 0, 0],     // 输入
      [0, 1.5, 0],   // attn（GQA）
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
        '官方：235B-A22B、128 专家激活 8、无共享专家、94 层、RoPE',
        '估算：d_model 4096、context 262144（YaRN 扩展）、sampler 与训练细节',
        '无 MLA：Qwen3 用 GQA + QK-Norm，attn 为通用占位类型',
        '混合思维模式（thinking / non-thinking）为其产品特色'
      ]
    }
  };
})();
