// models/bert.js — BERT-Base（encoder-only 双向，官方）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.bert = {
    id: 'bert',
    name: 'BERT-Base',
    sourceTag: 'official',
    params: { total: '110M', active: '110M' },
    context: 512,   // 官方最大序列长度
    blockCount: 12, // 官方 base：12 层 encoder
    d_model: 768,   // 官方 base
    layers: [
      { type: 'attn' },   // 双向全量自注意力（MLM 遮住 token 预测）
      { type: 'norm' },
      { type: 'output' },
      { type: 'softmax' }
    ],
    moe: { experts: 1, active: 1, shared: 0 },
    quant: null,
    // ---- v2 字段 ----
    category: 'llm',
    posEncoding: { type: 'rope' }, // 注：原作为可学习绝对位置嵌入，按契约统一 rope，见 meta.notes
    sampler: { temperature: 1.0, topP: 0.95 }, // 估算：BERT 非生成式，无采样参数
    training: {
      stages: [
        { id: 'pretrain', desc: 'MLM（掩码语言模型）+ NSP 双向预训练（官方）' },
        { id: 'sft', desc: '下游任务微调（GLUE/SQuAD，官方）' },
        { id: 'rl', desc: '无 RL 阶段（官方）' }
      ]
    },
    flow: [
      [0, 0, 0],     // 输入
      [0, 1.5, 0],   // attn（双向自注意力）
      [0, 3, 0],     // norm
      [0, 4.5, 0],   // output
      [0, 6, 0]      // softmax 输出
    ],
    meta: {
      notes: [
        '官方（Devlin et al. 2018）：BERT-Base 110M、12 层、d_model 768、12 头、context 512',
        'encoder-only 双向架构：注意力同时看左右上下文，与 GPT 系单向因果注意力形成对比',
        'MLM 训练：随机遮住 15% token 让模型预测，不是自回归生成',
        '位置编码为可学习绝对位置嵌入；posEncoding 按契约统一填 rope'
      ]
    }
  };
})();
