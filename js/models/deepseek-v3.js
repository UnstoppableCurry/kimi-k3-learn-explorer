// models/deepseek-v3.js — DeepSeek V3（671B/37B 激活，MLA + MoE，官方）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.deepseekv3 = {
    id: 'deepseekv3',
    name: 'DeepSeek V3',
    sourceTag: 'official',
    params: { total: '671B', active: '37B' },
    context: 131072, // 官方（YaRN 扩展）
    blockCount: 61,  // 官方：61 层（前 3 层 dense）
    d_model: 7168,   // 官方
    layers: [
      { type: 'mla' },
      { type: 'router' },
      { type: 'experts' },
      { type: 'norm' },
      { type: 'output' },
      { type: 'softmax' }
    ],
    moe: { experts: 256, active: 8, shared: 1 }, // 官方：256 路由专家选 8 + 1 共享
    quant: null, // 训练用 FP8 混合精度，非权重量化发布
    // ---- v2 字段 ----
    category: 'llm',
    posEncoding: { type: 'rope' }, // 官方：RoPE
    sampler: { temperature: 1.0, topP: 0.95 }, // 估算：官方未固定采样参数
    training: {
      stages: [
        { id: 'pretrain', desc: '14.8T token 预训练，FP8 混合精度（官方）' },
        { id: 'sft', desc: '监督微调（官方）' },
        { id: 'rl', desc: 'GRPO 强化学习（官方）' }
      ]
    },
    flow: [
      [0, 0, 0],     // 输入
      [0, 1.5, 0],   // mla
      [0, 3, 0],     // router
      [2.4, 4.5, 0], // 路由到专家（右）
      [-2.4, 4.5, 0],// 扫过专家（左）
      [0, 4.5, 0],   // 聚合回主干（含 1 共享专家）
      [0, 6, 0],     // norm
      [0, 7.5, 0],   // output
      [0, 9, 0]      // softmax 输出
    ],
    meta: {
      notes: [
        '官方（V3 技术报告）：671B 总参 / 37B 激活、256 路由专家选 8 + 1 共享、61 层、d_model 7168',
        'MLA（多头潜在注意力）：KV 压缩进低秩潜在向量，KV cache 远小于 GQA',
        '无辅助损失的负载均衡策略 + 多 token 预测（MTP）为训练创新点',
        '前 3 层为 dense 层无 router，此 spec 按 MoE 层简化展示'
      ]
    }
  };
})();
