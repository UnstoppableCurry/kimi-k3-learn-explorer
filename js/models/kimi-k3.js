// models/kimi-k3.js — Kimi K3（官方数据为主）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.k3 = {
    id: 'k3',
    name: 'Kimi K3',
    sourceTag: 'official',
    params: { total: '2.8T', active: '16/896 专家' },
    context: 1000000,
    blockCount: 60, // 估算，官方未公布（见 meta.notes）
    d_model: 7168,  // 估算，官方未公布
    // 一个 Block 内自底向上组件列表
    layers: [
      { type: 'kda' },
      { type: 'mla' },
      { type: 'router' },
      { type: 'experts' },
      { type: 'attnres' },
      { type: 'norm' },
      { type: 'output' },
      { type: 'softmax' }
    ],
    moe: { experts: 896, active: 16, shared: 1 },
    quant: 'MXFP4 权重 / MXFP8 激活',
    // ---- v2 字段 ----
    category: 'llm',
    posEncoding: { type: 'rope' }, // 官方：RoPE
    sampler: { temperature: 1.0, topP: 0.95 }, // 官方固定采样参数
    training: {
      stages: [
        { id: 'pretrain', desc: '大规模预训练（官方，token 量未公开）' },
        { id: 'sft', desc: '监督微调；QAT 量化感知训练自此阶段开启（官方）' },
        { id: 'rl', desc: '长程 agent 强化学习后训练（官方）' }
      ]
    },
    // 粒子路径：y 轴自底向上，层间距 1.5；experts 处 x 向两侧偏移表示专家路由
    flow: [
      [0, 0, 0],     // 输入
      [0, 1.5, 0],   // kda
      [0, 3, 0],     // mla
      [0, 4.5, 0],   // router
      [2.4, 6, 0],   // 路由到专家（右）
      [-2.4, 6, 0],  // 扫过专家（左）
      [0, 6, 0],     // 聚合回主干
      [0, 7.5, 0],   // attnres
      [0, 9, 0],     // norm
      [0, 10.5, 0],  // output
      [0, 12, 0]     // softmax 输出
    ],
    meta: {
      notes: [
        '官方：total 2.8T、active 16/896 专家、context 1M、MoE 896 专家/激活 16/共享 1、MXFP4 权重 / MXFP8 激活',
        '估算：blockCount 60、d_model 7168（官方未公布，沿 K2 量级估）',
        'attnres（注意力残差）为 K3 特有结构，组件按官方架构描述占位',
        'kda + mla 混合注意力：KDA 线性注意力层与 MLA 层并存，场景中都应出现',
        'v2：posEncoding RoPE / sampler(1.0, 0.95) / training 三阶段（QAT 从 SFT 起、长程 agent RL）均为官方口径'
      ]
    }
  };
})();
