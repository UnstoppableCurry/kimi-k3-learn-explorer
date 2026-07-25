// models/gpt3.js — GPT-3 175B（dense，官方）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.gpt3 = {
    id: 'gpt3',
    name: 'GPT-3',
    sourceTag: 'official',
    params: { total: '175B', active: '175B' }, // dense：激活 = 总参
    context: 2048,   // 官方
    blockCount: 96,  // 官方 175B 版：96 层
    d_model: 12288,  // 官方 175B 版
    layers: [
      { type: 'attn' },   // 因果自注意力（交替 dense / sparse 模式）
      { type: 'norm' },
      { type: 'output' },
      { type: 'softmax' }
    ],
    moe: { experts: 1, active: 1, shared: 0 },
    quant: null,
    // ---- v2 字段 ----
    category: 'llm',
    posEncoding: { type: 'rope' }, // 注：原作为可学习位置嵌入，按契约统一 rope，见 meta.notes
    sampler: { temperature: 1.0, topP: 0.95 }, // 估算：官方 API 默认值附近
    training: {
      stages: [
        { id: 'pretrain', desc: 'Common Crawl 等 300B token 自回归预训练（官方）' },
        { id: 'sft', desc: '无 SFT：主打 few-shot in-context learning（官方）' },
        { id: 'rl', desc: '无 RL 阶段（官方；RLHF 从 InstructGPT 起）' }
      ]
    },
    flow: [
      [0, 0, 0],     // 输入
      [0, 1.5, 0],   // attn
      [0, 3, 0],     // norm
      [0, 4.5, 0],   // output
      [0, 6, 0]      // softmax 输出
    ],
    meta: {
      notes: [
        '官方（Brown et al. 2020）：175B、96 层、d_model 12288、96 头、context 2048',
        'dense 架构：175B 全部参数每个 token 都激活，与 MoE 模型形成规模对比',
        '论文提到交替使用 dense 与 locally banded sparse attention 模式',
        '位置编码为可学习位置嵌入；posEncoding 按契约统一填 rope'
      ]
    }
  };
})();
