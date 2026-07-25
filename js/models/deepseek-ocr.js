// models/deepseek-ocr.js — DeepSeek-OCR（视觉压缩 OCR，估算）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.deepseekocr = {
    id: 'deepseekocr',
    name: 'DeepSeek-OCR',
    sourceTag: 'estimated',
    params: { total: '3B', active: '3B' }, // 官方：DeepEncoder + DeepSeek3B-MoE 解码器
    context: 4096,   // 估算（视觉 token 压缩后序列）
    blockCount: 30,  // 估算
    d_model: 2048,   // 估算
    layers: [],      // CV 模型用 cv.components
    moe: null,
    quant: null,
    // ---- v2 字段 ----
    category: 'cv',
    cv: {
      components: [
        { type: 'vit-patch' },      // 文档图像分块
        { type: 'image-encoder' },  // DeepEncoder（SAM + CLIP 串联）视觉编码压缩
        { type: 'sam-decoder' }     // 视觉 token → 文本 token 的 MoE 解码器（占位）
      ]
    },
    posEncoding: { type: 'rope' }, // 估算：解码器为 DeepSeek 系，RoPE
    sampler: { temperature: 1.0, topP: 0.95 }, // 估算
    training: {
      stages: [
        { id: 'pretrain', desc: '文档视觉-文本对预训练（估算）' },
        { id: 'sft', desc: 'OCR 识别 + 版面解析微调（估算）' },
        { id: 'rl', desc: '无公开 RL 阶段（估算）' }
      ]
    },
    flow: [
      [0, 0, 0],     // 输入文档图像
      [0, 1.5, 0],   // vit-patch（分块）
      [0, 3, 0],     // image-encoder（视觉压缩）
      [0, 4.5, 0],   // sam-decoder（解码为文本）
      [0, 6, 0]      // 文本输出
    ],
    meta: {
      notes: [
        '官方：DeepEncoder（SAM-base + 16x 压缩卷积 + CLIP-large）+ DeepSeek3B-MoE-A570M 解码器',
        '核心思想：视觉 token 压缩——一张文档图压到几十~几百个视觉 token，验证"一图胜千言"',
        '估算：context / blockCount / d_model / 训练流程细节未逐项公开，sourceTag 标 estimated',
        'cv.components 中 sam-decoder 为占位类型，实际为 MoE 文本解码器，engine 按通用组件渲染即可'
      ]
    }
  };
})();
