// models/sam.js — SAM（Segment Anything Model，CV 通用分割，官方）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.sam = {
    id: 'sam',
    name: 'SAM',
    sourceTag: 'official',
    params: { total: '641M', active: '641M' }, // ViT-H 档；另有 ViT-L 312M / ViT-B 93M
    context: 1024,   // 输入图像边长（像素）
    blockCount: 32,  // ViT-H image encoder 层数
    d_model: 1280,   // ViT-H 嵌入维度
    layers: [],      // CV 模型用 cv.components
    moe: null,
    quant: null,
    // ---- v2 字段 ----
    category: 'cv',
    cv: {
      components: [
        { type: 'vit-patch' },      // 图像分块 + patch embedding
        { type: 'image-encoder' },  // ViT 图像编码器（重，一次性）
        { type: 'sam-decoder' }     // 轻量掩码解码器（点/框提示 → 掩码）
      ]
    },
    posEncoding: { type: 'rope' }, // 注：原作为 ViT 可学习位置嵌入，按契约统一 rope，见 meta.notes
    sampler: { temperature: 1.0, topP: 0.95 }, // 不适用：分割无采样，占位
    training: {
      stages: [
        { id: 'pretrain', desc: 'SA-1B 数据集 11M 图 / 1.1B 掩码训练（官方）' },
        { id: 'sft', desc: '提示式分割微调（点/框/掩码提示，官方）' },
        { id: 'rl', desc: '无 RL 阶段（官方）' }
      ]
    },
    flow: [
      [0, 0, 0],     // 输入图像
      [0, 1.5, 0],   // vit-patch（分块）
      [0, 3, 0],     // image-encoder（ViT 编码）
      [0, 4.5, 0],   // sam-decoder（提示融合 + 掩码解码）
      [0, 6, 0]      // 掩码输出
    ],
    meta: {
      notes: [
        '官方（Meta SAM, Kirillov et al. 2023）：image encoder + prompt encoder + mask decoder 三段式',
        'params 取 ViT-H 档 641M；图像编码器重但只跑一次，解码器轻量可实时交互',
        '位置嵌入为 ViT 可学习绝对位置嵌入；posEncoding 按契约统一填 rope',
        'zero-shot 分割能力：训练数据 SA-1B 为当时最大分割数据集'
      ]
    }
  };
})();
