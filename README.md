# Kimi K3 架构 3D 探索器

一个交互式的 3D 可视化工具，用于学习大语言模型（LLM）的架构原理。

## 特性

- **真实结构**：不是抽象隐喻，而是真实的 3D 表示。能看到向量、矩阵、权重、数据流。
- **沉浸式体验**：全屏 3D，UI 最少化，聚焦核心概念。
- **模块化设计**：每个组件（Token/Attention/FFN/MoE/Output/Training）独立实现，可单独学习。
- **交互式探索**：点击组件查看详情，拖拽旋转，滚轮缩放。

## 模块

- **Token/Embedding**：文本 → token → embedding 向量 → 位置编码
- **Attention**：Q/K/V 向量、attention 权重矩阵热力图、注意力输出
- **FFN**：输入层 → 隐藏层 → 输出层的向量变换
- **MoE**：路由分数、专家选择、专家输出、加权聚合
- **Output**：logits、softmax 概率分布、采样过程
- **Training**：pretrain/SFT/RL 三阶段的 loss 曲线、梯度流、参数更新

## 技术栈

- Three.js（3D 渲染）
- 原生 JavaScript（无框架依赖）
- 模块化架构（每个组件独立实现）

## 本地运行

```bash
cd explorer
python3 -m http.server 8765
# 打开 http://localhost:8765
```

## 开源协议

MIT License

## 致谢

灵感来自 [bbycroft.net/llm](https://bbycroft.net/llm)，但重新设计为更真实、更沉浸的 3D 体验。
