// models/transformer2017.js — Transformer（Attention Is All You Need, 2017，官方）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.transformer2017 = {
    id: 'transformer2017',
    name: 'Transformer (2017)',
    sourceTag: 'official',
    params: { total: '65M', active: '65M' }, // 官方 base 版 dense
    context: 512,      // 论文训练序列长度
    blockCount: 6,     // 官方 base：6 encoder + 6 decoder 层
    d_model: 512,      // 官方 base
    layers: [
      { type: 'attn' },   // 多头自注意力（全量注意力）
      { type: 'norm' },
      { type: 'output' },
      { type: 'softmax' }
    ],
    moe: { experts: 1, active: 1, shared: 0 }, // 非 MoE 兜底
    quant: null,
    // ---- v2 字段 ----
    category: 'llm',
    posEncoding: { type: 'rope' }, // 注：原始论文为 sinusoidal，此处按契约统一 rope，详见 meta.notes
    sampler: { temperature: 1.0, topP: 0.95 }, // 估算：论文用 beam search，无采样参数
    training: {
      stages: [
        { id: 'pretrain', desc: 'WMT 2014 英德/英法翻译任务训练（官方）' },
        { id: 'sft', desc: '无独立 SFT 阶段：单阶段端到端训练（官方）' },
        { id: 'rl', desc: '无 RL 阶段（官方）' }
      ]
    },
    flow: [
      [0, 0, 0],     // 输入
      [0, 1.5, 0],   // attn（多头自注意力）
      [0, 3, 0],     // norm
      [0, 4.5, 0],   // output
      [0, 6, 0]      // softmax 输出
    ],
    meta: {
      notes: [
        '官方（Vaswani et al. 2017）：base 版 65M 参数、d_model 512、6 层、8 头注意力',
        '原文位置编码为 sinusoidal 绝对位置编码；posEncoding 按契约统一填 rope，渲染差异见备注',
        '无 MoE / 无 RoPE / 无量化：最原始的 dense Transformer，作为架构演进起点',
        'encoder-decoder 结构，此 spec 按单栈简化展示'
      ]
    }
  };
})();
