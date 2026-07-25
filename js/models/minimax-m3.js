// models/minimax-m3.js — MiniMax M3（估算）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.minimax3 = {
    id: 'minimax3',
    name: 'MiniMax M3',
    sourceTag: 'estimated',
    params: { total: '~2.7T（估算）', active: '~50B（估算）' },
    context: 1000000, // 估算（lightning attention 主打长上下文）
    blockCount: 80,   // 估算
    d_model: 6144,    // 估算
    // lightning-attn 为占位类型：MiniMax Lightning Attention（线性注意力），与 mla 混合堆叠
    layers: [
      { type: 'lightning-attn' },
      { type: 'mla' },
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
    posEncoding: { type: 'rope' }, // 估算：按 MiniMax-01 的 RoPE 外推
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
      [0, 1.5, 0],   // lightning-attn
      [0, 3, 0],     // mla
      [0, 4.5, 0],   // router
      [2.4, 6, 0],   // 路由到专家（右）
      [-2.4, 6, 0],  // 扫过专家（左）
      [0, 6, 0],     // 聚合回主干
      [0, 7.5, 0],   // norm
      [0, 9, 0],     // output
      [0, 10.5, 0]   // softmax 输出
    ],
    meta: {
      notes: [
        '全部数字为估算（sourceTag: estimated），官方未公开 M3 完整配置',
        'lightning-attn 是约定占位类型：MiniMax 线性注意力（Lightning Attention），engine 按通用组件渲染即可',
        '结构参考 MiniMax-01 的混合设计：lightning-attn 与 mla 混合 + MoE',
        '估算依据：~2.7T 总参 / 256 专家 / 80 层 / context 1M，按 MiniMax-01/M2 外推'
      ]
    }
  };
})();
