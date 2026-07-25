// models/rwkv.js — RWKV（RNN 式线性时间架构，官方概念）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.rwkv = {
    id: 'rwkv',
    name: 'RWKV',
    sourceTag: 'official', // 概念与占位组件类型为官方口径，参数按 RWKV-4 14B 档
    params: { total: '14B', active: '14B' },
    context: 0, // 概念上无限（RNN 状态递推），0 表示不适用
    blockCount: 40,  // RWKV-4 14B：40 层
    d_model: 5120,   // RWKV-4 14B
    // time-mix / channel-mix 为占位类型：RWKV 的时间混合与通道混合模块
    layers: [
      { type: 'time-mix' },
      { type: 'channel-mix' },
      { type: 'norm' },
      { type: 'output' },
      { type: 'softmax' }
    ],
    moe: { experts: 1, active: 1, shared: 0 },
    quant: null,
    // ---- v2 字段 ----
    category: 'llm',
    posEncoding: { type: 'rope' }, // 注：RWKV 无位置编码（用 time decay 替代），按契约统一 rope，见 meta.notes
    sampler: { temperature: 1.0, topP: 0.95 }, // 估算
    training: {
      stages: [
        { id: 'pretrain', desc: '自回归预训练，RNN 模式 O(n) 推理（官方）' },
        { id: 'sft', desc: '监督微调（估算：官方有 chat 微调版）' },
        { id: 'rl', desc: '强化学习对齐（估算）' }
      ]
    },
    flow: [
      [0, 0, 0],     // 输入
      [0, 1.5, 0],   // time-mix（时间混合，替代注意力）
      [0, 3, 0],     // channel-mix（通道混合）
      [0, 4.5, 0],   // norm
      [0, 6, 0],     // output
      [0, 7.5, 0]    // softmax 输出
    ],
    meta: {
      notes: [
        '官方概念（Peng et al. RWKV）：time-mix 用线性递推替代注意力，无 KV cache，推理显存恒定',
        'time-mix / channel-mix 是约定占位类型，engine 按通用组件渲染即可',
        'RWKV 不用位置编码：用 time decay（时间衰减）因子表达位置关系；posEncoding 按契约统一填 rope',
        '参数档取 RWKV-4 14B（40 层、d_model 5120）；训练可并行（Transformer 模式）、推理可递推（RNN 模式）'
      ]
    }
  };
})();
