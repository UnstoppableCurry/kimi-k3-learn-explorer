# 模块接口规范 v1

## 目标

每个模块是一个独立的 3D 可视化单元，负责神经网络中的一个真实组件（Token/Attention/FFN/MoE/Output/Training）。

## 核心原则（乔布斯式）

1. **真实结构** — 必须是真实的 3D 表示，不是抽象隐喻。能看到向量、矩阵、权重、数据流。
2. **沉浸式** — 3D 占满屏幕，UI 最少化。
3. **聚焦** — 一次只展示一个概念，避免信息过载。
4. **细节** — 每个像素都要精致，材质、光照、阴影都要真实。

## 模块接口

每个模块是一个独立的 JS 文件，暴露一个全局对象：

```javascript
window.MODULE_NAME = {
  // 初始化：传入 THREE.js 的 scene/camera/renderer
  init: function(scene, camera, renderer) {},

  // 显示模块：传入数据（token 向量、attention 权重等）
  show: function(data) {},

  // 隐藏模块
  hide: function() {},

  // 更新动画：每帧调用，delta 是时间增量（秒）
  update: function(delta) {},

  // 销毁：清理资源
  dispose: function() {},

  // 截图：返回当前模块的 canvas 截图（用于验证）
  screenshot: function() { return renderer.domElement.toDataURL(); }
};
```

## 数据流规范

模块之间通过 `data` 对象传递：

```javascript
// Token 模块输出
{
  tokens: [
    { id: 0, text: "你", embedding: [0.1, 0.2, ...], position: 0 },
    ...
  ]
}

// Attention 模块输出
{
  q: [...], k: [...], v: [...],
  weights: [[0.1, 0.2, ...], ...],  // attention 权重矩阵
  output: [...]  // attention 输出向量
}

// FFN 模块输出
{
  input: [...],
  hidden: [...],
  output: [...]
}

// MoE 模块输出
{
  routerScores: [0.1, 0.2, ...],
  selectedExperts: [3, 7, 15, ...],
  expertOutputs: [...],
  aggregated: [...]
}

// Output 模块输出
{
  logits: [...],
  probs: [...],  // softmax 概率分布
  sampledToken: { id: 42, text: "好" }
}
```

## 3D 视觉规范

### Token/Embedding 模块
- 每个 token 是一个彩色小球（球体半径 0.5）
- embedding 向量用彩色线条表示（从小球出发）
- 位置编码用旋转动画表示

### Attention 模块
- Q/K/V 向量用三个彩色小球表示
- attention 权重矩阵用热力图平面表示（红色=高权重，蓝色=低权重）
- 注意力输出用向量变形动画表示

### FFN 模块
- 输入向量 → 隐藏层向量 → 输出向量，用向量变形动画表示
- 隐藏层维度更大，用更多小球表示

### MoE 模块
- 路由分数用柱状图表示
- 选中的专家用高亮小球表示
- 专家输出用向量表示
- 聚合用加权求和动画表示

### Output 模块
- logits 用柱状图表示
- softmax 概率分布用饼图/柱状图表示
- 采样过程用动画表示

### Training 模块
- loss 曲线用 3D 折线图表示
- 梯度流用粒子流表示
- 参数更新用向量变化动画表示

## 文件结构

```
explorer/
  index.html          # 极简 UI（全屏 3D + 4 个按钮）
  js/
    main.js           # 主控制器（加载所有模块，协调显示）
    core/
      scene.js        # 场景管理（光照、阴影、背景）
      camera.js       # 相机控制（飞行、聚焦、重置）
      renderer.js     # 渲染器（WebGL、性能优化）
    modules/
      token.js        # Token/Embedding 模块
      attention.js    # Attention 模块
      ffn.js          # FFN 模块
      moe.js          # MoE 模块
      output.js       # 输出/采样模块
      training.js     # 训练可视化模块
    ui/
      panel.js        # 右侧面板（点击组件才显示）
      controls.js     # 控制按钮（播放/暂停/单步/重置）
```

## 验证要求

每个模块完成后必须：
1. 截图验证 3D 效果
2. 截图验证 data flow 效果
3. 确保接口正确（可以被 main.js 调用）

## 禁止事项

- 不要使用抽象的"塔"或"平板"表示
- 不要添加不必要的 UI 元素
- 不要牺牲真实性为了性能
