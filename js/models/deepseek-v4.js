// models/deepseek-v4.js — DeepSeek V4（估算）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.deepseekv4 = {
    id: 'deepseekv4',
    name: 'DeepSeek V4',
    sourceTag: 'estimated',
    params: { total: '~1.6T（估算）', active: '~49B（估算）' },
    context: 1000000, // 估算（NSA 主打长上下文）
    blockCount: 61,   // 估算（沿 V3）
    d_model: 7168,    // 估算（沿 V3）
    // nsa-attn 为占位类型：NSA（Native Sparse Attention）原生稀疏注意力
    layers: [
      { type: 'mla' },
      { type: 'nsa-attn' },
      { type: 'router' },
      { type: 'experts' },
      { type: 'norm' },
      { type: 'output' },
      { type: 'softmax' }
    ],
    moe: { experts: 384, active: 8, shared: 1 },
    quant: null,
    // ---- v2 字段 ----
    category: 'llm',
    posEncoding: { type: 'rope' }, // 估算：沿 V3 官方 RoPE
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
      [0, 1.5, 0],   // mla
      [0, 3, 0],     // nsa-attn
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
        '全部数字为估算（sourceTag: estimated），官方未发布 V4',
        'nsa-attn 是约定占位类型：NSA（Native Sparse Attention）原生稀疏注意力，engine 按通用组件渲染即可',
        '估算依据：~1.6T 总参 / 384 专家 / 61 层 / d_model 7168，按 V3（671B/256 专家/61 层）外推',
        'active ~49B：按 V3 的 37B 激活同比例放大的粗估'
      ]
    }
  };
})();
