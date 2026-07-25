// content/tour.js — 导览剧本（8 幕，第 3 幕"逐层穿越"拆为 4 小幕，共 11 个 act）
// 契约：window.CONTENT_TOUR = [act]
// act = {id, title, subtitle, cam:{pos:[x,y,z], target:[x,y,z]}, duration(s), focusComponent, effect, switchModel}
// 场景约定：塔高约 60 单位（y: 0→60，分词在塔底、softmax 在塔顶），塔宽约 40（x/z 以 0 为中心）
// effect 取值：引擎内置 'route' / 'softmax' / 'autoregress' / 'attnres' / 'kda-write'，
// 以及 pipeline.js 扩展的 'tokenize' / 'rope' / 'gate' / 'aggregate' / 'sample'，无动效用 null
window.CONTENT_TOUR = [

  // 第 0 幕：训练 vs 推理（开场全景，15s）
  {
    id: 'intro',
    title: '训练 vs 推理',
    subtitle: '训练像备考——把上万亿个参数调到最好；推理像上考场——接下来我们跟着一个 token，把考场路线完整走一遍。',
    cam: { pos: [37.44,27.36,44.64], target: [0, 28, 0] },
    duration: 15,
    focusComponent: null,
    effect: null
  },

  // 第 1 幕：分词
  {
    id: 'tokenize',
    title: '分词：字 → 编号',
    subtitle: '模型不认字、只认数字——分词器先把这句话切成一个个 token，每个 token 对应词表里的一个编号。',
    cam: { pos: [12.96,5.76,18.72], target: [0, 4, 0] },
    duration: 10,
    focusComponent: 'tokenizer',
    effect: 'tokenize'
  },

  // 第 2 幕：嵌入
  {
    id: 'embed',
    title: '嵌入：编号 → 坐标',
    subtitle: '嵌入层把编号翻译成一串几千维的坐标，意思相近的词坐标也近——从这里开始，模型就在"空间"里思考了。',
    cam: { pos: [11.52,10.08,17.28], target: [0, 10, 0] },
    duration: 10,
    focusComponent: 'embedding',
    effect: 'rope'
  },

  // 第 3 幕（小幕 1/4）：KDA 线性注意力
  {
    id: 'layer-kda',
    title: 'KDA：固定大小的记忆',
    subtitle: 'KDA 把读过的内容压缩进一块固定大小的记忆矩阵，边读边改写——所以一百万字的上下文也撑不爆显存。',
    cam: { pos: [10.08,15.84,15.84], target: [0, 20, 0] },
    duration: 14,
    focusComponent: 'kda',
    effect: 'kda-write'
  },

  // 第 3 幕（小幕 2/4）：MLA 潜在注意力
  {
    id: 'layer-mla',
    title: 'MLA：缩微胶卷式的缓存',
    subtitle: 'MLA 把每个词留下的线索压成缩微胶卷、用时再放大——K2 当年就是靠它把 128K 上下文的记忆成本砍下来的。',
    cam: { pos: [10.08,21.6,15.84], target: [0, 28, 0] },
    duration: 12,
    focusComponent: 'mla',
    effect: null
  },

  // 第 3 幕（小幕 3/4）：路由 → 专家
  {
    id: 'layer-router',
    title: '路由：896 选 16 的分诊台',
    subtitle: '路由器给这个 token 打分，从 896 位专科医生里挑 16 位最对口的来看它——其余 880 位这轮休息，算力就这么省下来的。',
    cam: { pos: [11.52,27.36,17.28], target: [0, 36, 0] },
    duration: 14,
    focusComponent: 'router',
    effect: 'route'
  },

  // 第 3 幕（小幕 3.5/4）：专家输出如何汇总
  {
    id: 'layer-aggregate',
    title: '聚合：16 个专家会诊出一份结论',
    subtitle: '16 位专家各算各的，最后按路由分数加权求和、合并成一股数据继续上行——容量万亿、每次只花零头的秘密就在这一步。',
    cam: { pos: [11.52,30.24,17.28], target: [0, 39, 0] },
    duration: 12,
    focusComponent: 'experts',
    effect: 'aggregate'
  },

  // 第 3 幕（小幕 4/4）：AttnRes 跨深度检索（机位移到侧面看纵向光束）
  {
    id: 'layer-attnres',
    title: 'AttnRes：回头点名某一层的结论',
    subtitle: 'AttnRes 让深层可以回头"点名"要前面某一层的结论，而不是把所有层的输出不分主次地糊在一起。',
    cam: { pos: [21.6,21.6,7.2], target: [0, 32, 0] },
    duration: 12,
    focusComponent: 'attnres',
    effect: 'attnres'
  },

  // 第 4 幕：softmax 输出概率
  {
    id: 'softmax',
    title: '塔顶决策：分数 → 概率',
    subtitle: '走到塔顶，softmax 把几万个候选词的打分变成概率——"的"30%、"是"18%……下一个词就按这个分布挑出来。',
    cam: { pos: [10.08,44.64,15.84], target: [0, 58, 0] },
    duration: 12,
    focusComponent: 'softmax',
    effect: 'softmax'
  },

  // 第 4.5 幕：采样参数怎么影响输出
  {
    id: 'sample',
    title: '采样：温度与 top-p 怎么"捏"分布',
    subtitle: '温度调高分布变平、调低变尖；top-p 把尾巴砍掉再归一化——K3 官方固定温度 1.0、top-p 0.95，现在轮盘会按这个分布真选一次。',
    cam: { pos: [10.08,44.64,15.84], target: [0, 58, 0] },
    duration: 12,
    focusComponent: 'softmax',
    effect: 'sample'
  },

  // 第 5 幕：自回归循环（视角拉高看 token 从塔顶滑回塔底）
  {
    id: 'autoregress',
    title: '自回归：一次只蹦一个字',
    subtitle: '刚生成的词会被接回句尾、重新从塔底再走一遍——你看到的"打字机效果"不是特效，模型物理上就在一个字一个字算。',
    cam: { pos: [24.48,28.8,28.8], target: [0, 32, 0] },
    duration: 14,
    focusComponent: 'output',
    effect: 'autoregress'
  },

  // 第 6 幕：1M 上下文 · 记忆矩阵 vs KV 缓存（拉远全景）
  {
    id: 'memory',
    title: '1M 上下文：白板 vs 便签墙',
    subtitle: '拉远看：传统模型给每个词留一张便签、越攒越多；KDA 只有一块固定白板——这就是 1M 上下文跑不跑得起来的分水岭。',
    cam: { pos: [56.16,39.6,61.2], target: [0, 30, 0] },
    duration: 14,
    focusComponent: 'kda',
    effect: 'kda-write'
  },

  // 第 7 幕：切换到 K2 重放，看架构差异（K3 独有的 KDA/AttnRes 缺失，MLA + 384 选 8+1）
  {
    id: 'k2-diff',
    title: '重放：换成 K2 走一遍',
    subtitle: '换成 K2 重放一遍：没有 KDA 和 AttnRes，靠 MLA 加 384 选 8 的专家——同一条流水线，两代人不同的取舍。',
    cam: { pos: [37.44,27.36,44.64], target: [0, 28, 0] },
    duration: 16,
    focusComponent: null,
    effect: null,
    switchModel: 'k2'
  }
];
