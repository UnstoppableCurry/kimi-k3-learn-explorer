// models/yolov8seg.js — YOLOv8-Seg（CV 实例分割，官方）
(function () {
  'use strict';
  window.MODEL_SPECS = window.MODEL_SPECS || {};

  window.MODEL_SPECS.yolov8seg = {
    id: 'yolov8seg',
    name: 'YOLOv8-Seg',
    sourceTag: 'official',
    params: { total: '3.4M', active: '3.4M' }, // n 档；seg 各档 3.4M–71.8M
    context: 640,    // 输入图像边长（像素），非 token 上下文
    blockCount: 1,   // 非层叠 block 架构，按 cv.components 展示
    d_model: 0,      // 不适用（CNN 通道数随深度变化）
    layers: [],      // CV 模型用 cv.components，不用 layers
    moe: null,       // 非 MoE
    quant: null,
    // ---- v2 字段 ----
    category: 'cv',
    cv: {
      components: [
        { type: 'vit-patch' },  // 输入分块（占位：统一 CV 入口表示）
        { type: 'backbone' },   // CSPDarknet 主干特征提取
        { type: 'neck' },       // PAFPN 多尺度特征融合
        { type: 'yolo-head' }   // 检测 + 分割原型系数双分支头
      ]
    },
    posEncoding: { type: 'rope' }, // 注：CNN 无位置编码，按契约统一 rope，见 meta.notes
    sampler: { temperature: 1.0, topP: 0.95 }, // 不适用：CV 检测无采样，占位
    training: {
      stages: [
        { id: 'pretrain', desc: 'ImageNet 分类预训练主干（官方）' },
        { id: 'sft', desc: 'COCO 检测 + 实例分割端到端训练（官方）' },
        { id: 'rl', desc: '无 RL 阶段（官方）' }
      ]
    },
    flow: [
      [0, 0, 0],     // 输入图像
      [0, 1.5, 0],   // vit-patch（分块入口）
      [0, 3, 0],     // backbone（主干特征提取）
      [0, 4.5, 0],   // neck（多尺度融合）
      [0, 6, 0]      // yolo-head（框 + 掩码输出）
    ],
    meta: {
      notes: [
        '官方（Ultralytics YOLOv8）：anchor-free 检测头 + 分割原型分支，COCO 训练',
        'params 取最小 n 档 3.4M；s/m/l/x 档至 71.8M',
        '纯 CNN：无注意力、无位置编码；posEncoding 按契约统一填 rope',
        'cv.components 的 vit-patch 为统一 CV 入口占位类型，engine 按通用组件渲染即可'
      ]
    }
  };
})();
