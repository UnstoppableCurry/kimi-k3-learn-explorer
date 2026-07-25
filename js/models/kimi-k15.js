// models/kimi-k15.js — Kimi K1.5（多模态推理模型，估算）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.k15 = {
    id: 'k15',
    name: 'Kimi K1.5',
    sourceTag: 'estimated',
    params: { total: '未公开', active: '未公开' }, // 官方未披露参数量
    context: 131072, // 估算
    blockCount: 61,  // 估算（沿 K 系公开架构量级）
    d_model: 7168,   // 估算
    layers: [
      { type: 'mla' },   // 估算：沿 K 系 MLA 设计
      { type: 'router' },
      { type: 'experts' },
      { type: 'norm' },
      { type: 'output' },
      { type: 'softmax' }
    ],
    moe: { experts: 384, active: 8, shared: 1 }, // 估算：沿 K2 公开配置
    quant: null,
    // ---- v2 字段 ----
    category: 'llm',
    posEncoding: { type: 'rope' }, // 估算：沿 K 系 RoPE
    sampler: { temperature: 1.0, topP: 0.95 }, // 估算
    training: {
      stages: [
        { id: 'pretrain', desc: '多模态联合预训练（估算）' },
        { id: 'sft', desc: '监督微调（估算）' },
        { id: 'rl', desc: '长 CoT 强化学习 + long2short（官方：技术报告核心）' }
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
      [0, 7.5, 0],   // output
      [0, 9, 0]      // softmax 输出
    ],
    meta: {
      notes: [
        '官方（K1.5 技术报告）：长 CoT 强化学习、long2short 蒸馏、多模态（文本+视觉）联合训练',
        '估算：参数量 / blockCount / d_model / moe 配置均未公开，沿 K2 量级外推',
        'K1.5 是 K 系 RL 推理路线的代表：o1 级数学/代码推理能力',
        '架构细节按 K 系 MLA + MoE 家族惯例占位，sourceTag 标 estimated'
      ]
    }
  };
})();
