// models/gpt2.js — GPT-2 Small（decoder-only 自回归，官方）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.gpt2 = {
    id: 'gpt2',
    name: 'GPT-2',
    sourceTag: 'official',
    params: { total: '117M', active: '117M' },
    context: 1024,  // 官方
    blockCount: 12, // 官方 small：12 层
    d_model: 768,   // 官方 small
    layers: [
      { type: 'attn' },   // 因果（单向）自注意力
      { type: 'norm' },
      { type: 'output' },
      { type: 'softmax' }
    ],
    moe: { experts: 1, active: 1, shared: 0 },
    quant: null,
    // ---- v2 字段 ----
    category: 'llm',
    posEncoding: { type: 'rope' }, // 注：原作为可学习绝对位置嵌入，按契约统一 rope，见 meta.notes
    sampler: { temperature: 1.0, topP: 0.95 }, // 估算：官方示例用 top-k 采样
    training: {
      stages: [
        { id: 'pretrain', desc: 'WebText 自回归预训练（官方）' },
        { id: 'sft', desc: '无 SFT：主打 zero-shot 能力（官方）' },
        { id: 'rl', desc: '无 RL 阶段（官方）' }
      ]
    },
    flow: [
      [0, 0, 0],     // 输入
      [0, 1.5, 0],   // attn（因果自注意力）
      [0, 3, 0],     // norm
      [0, 4.5, 0],   // output
      [0, 6, 0]      // softmax 输出
    ],
    meta: {
      notes: [
        '官方（Radford et al. 2019）：GPT-2 Small 117M、12 层、d_model 768、context 1024',
        'decoder-only 因果注意力：只能看左侧上下文，与 BERT 双向形成对比',
        '位置编码为可学习绝对位置嵌入；posEncoding 按契约统一填 rope',
        'dense 架构，无 MoE'
      ]
    }
  };
})();
