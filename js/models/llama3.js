// models/llama3.js — Llama 3.1 405B（dense，GQA，官方）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.llama3 = {
    id: 'llama3',
    name: 'Llama 3.1',
    sourceTag: 'official',
    params: { total: '405B', active: '405B' }, // dense
    context: 131072, // 官方
    blockCount: 126, // 官方 405B：126 层
    d_model: 16384,  // 官方 405B
    layers: [
      { type: 'attn' },   // GQA：8 个 KV 头对 64 个查询头
      { type: 'norm' },   // RMSNorm（pre-norm）
      { type: 'output' },
      { type: 'softmax' }
    ],
    moe: { experts: 1, active: 1, shared: 0 },
    quant: null,
    // ---- v2 字段 ----
    category: 'llm',
    posEncoding: { type: 'rope' }, // 官方：RoPE（base 500000）
    sampler: { temperature: 1.0, topP: 0.95 }, // 估算：官方未固定采样参数
    training: {
      stages: [
        { id: 'pretrain', desc: '15.6T token 预训练（官方）' },
        { id: 'sft', desc: '多轮监督微调 + 拒绝采样（官方）' },
        { id: 'rl', desc: 'DPO 直接偏好优化（官方）' }
      ]
    },
    flow: [
      [0, 0, 0],     // 输入
      [0, 1.5, 0],   // attn（GQA）
      [0, 3, 0],     // norm
      [0, 4.5, 0],   // output
      [0, 6, 0]      // softmax 输出
    ],
    meta: {
      notes: [
        '官方（Llama 3.1 报告）：405B、126 层、d_model 16384、context 131072、RoPE',
        'GQA（分组查询注意力）：64 查询头共享 8 组 KV 头，降低 KV cache 显存',
        'dense 架构：官方明确选择 dense 而非 MoE 以保证训练稳定性',
        '后训练用 SFT + 拒绝采样 + DPO，非 PPO'
      ]
    }
  };
})();
