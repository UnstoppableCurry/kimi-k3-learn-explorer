// content/components.js — 组件中文内容库
// 契约：window.CONTENT_COMPONENTS = { type: {one, analogy, principle, formula, quote, quoteUrl, tag} }
// tag: 'official' 官方来源 | 'inferred' 由官方/公开资料推断 | 'simplified' 教学简化
// 官方来源：Kimi K3 博客（2026-07-16 发布）与 platform.kimi.com 文档
window.CONTENT_COMPONENTS = {

  tokenizer: {
    one: '分词器把一句话切成模型能处理的最小单位——token，每个 token 对应词表里的一个编号。',
    analogy: '像下厨前的切菜：整句话被切成大小合适的块，模型一口一口吃。比如"人工智能"可能被切成 2～3 块，英文单词常被切成词根碎片。',
    principle: '主流方案是 BPE（字节对编码）：从单个字节出发，反复合并出现频率最高的相邻片段，直到词表达标。文本和 token 序列之间可逆转换；词表大小决定了模型"认识"多少种片段，也直接影响中文等语言被切得多碎。',
    formula: 'BPE 训练：重复执行 合并 argmax_{(a,b)} freq(a,b)，直到词表达标（无显式损失函数，属工程算法）',
    quote: '分词是文字进模型前的第一道翻译：字 → 编号。此后模型看到的一切都是数字。',
    quoteUrl: 'https://arxiv.org/abs/1508.07909',
    tag: 'simplified'
  },

  embedding: {
    one: '嵌入层把每个 token 编号查表变成一串几千维的数字向量——意思相近的词，向量距离也相近。',
    analogy: '像给每个词发一张超大城市里的 GPS 坐标："猫"和"狗"住对门，"猫"和"民主"隔着半个城。模型不靠字面、靠坐标理解世界。',
    principle: '嵌入矩阵 E 的每一行是一个词的训练产物，查表即取出该词向量 x = E[id]。这些向量在训练中和全网络一起调整，语义结构（近义聚类、类比方向）自发涌现，是后续所有层加工的原料。',
    formula: 'x = E[token_id]，E ∈ ℝ^{V×d_model}（V=词表大小，d_model=模型宽度）',
    quote: '从嵌入层开始，语言变成了几何：词与词的关系就是空间里方向与距离的关系。',
    quoteUrl: 'https://platform.kimi.com/docs',
    tag: 'simplified'
  },

  kda: {
    one: 'KDA（Kimi Delta Attention）是 K3 自研的线性注意力：把"每步翻遍全部历史"改成"维护一块可增量改写的记忆矩阵"，长上下文又快又省。',
    analogy: '传统注意力像传话游戏：每多说一个词，都要把前面所有话重听一遍，记录纸越攒越长。KDA 像一块可倒带、可改写的录音白板——新信息写进去、旧信息按规则抹淡，板子大小永远不变，所以 1M 上下文也撑不爆。',
    principle: '全量注意力的计算量随长度平方增长、KV 缓存随长度线性膨胀，这是长上下文的两堵墙。KDA 用 Delta Rule 维护固定大小的状态矩阵 S：每一步先在旧 key 方向上"擦除"过期记忆，再把新 value 写入，单步开销与历史长度无关，把注意力从 O(T²) 拉到 O(T)。',
    formula: 'S_t = S_{t-1}·(I - β·k·kᵀ) + β·v·kᵀ　（括号项负责擦除旧记忆，末项写入新记忆）［推断：官方未公布实现细节，此处为 Delta Rule 标准形式］',
    quote: 'KDA provides an efficient foundation for scaling attention',
    quoteUrl: 'https://kimi.com/blog/kimi-k3',
    tag: 'official',
    numExample: '实算示例：token「暗」的 8 维向量前 4 维作 k、后 4 维作 v，4×4 状态矩阵 S 按 S←S·(I−0.35·kkᵀ)+0.35·vkᵀ 逐元素真更新，写入后 S 首行变为 [−0.17,0.03,0.50,−0.32]。'
  },

  mla: {
    one: 'MLA（多头潜在注意力）把每个 token 的 Key/Value 联合压缩成一个小"潜向量"存起来，用到时再展开——缓存大幅瘦身。',
    analogy: '像把整本书做成缩微胶卷：书架（显存）省下大半，要读哪页就放大还原哪页。K2 靠它在 128K 上下文下把记忆成本压到可负担。',
    principle: '标准多头注意力要为每个头、每个 token 存完整 K 和 V；MLA 把多头 KV 低秩压缩成一个共享 latent 向量，推理时只缓存它，计算注意力时再重构各头 K/V。在 K3 的层序列中，MLA 与 KDA 搭配，负责需要全量注意力的部分（按架构规格）。K2 则以 MLA 为主力（官方），支撑 128K 上下文。',
    formula: 'c_t = W^{DKV}·x_t（压缩）；k_t = W^{UK}·c_t，v_t = W^{UV}·c_t（用时重构）［公开发表的方法形式］',
    quote: 'K2 采用 MLA 并支持 128K 上下文（官方事实，转述）；K3 中 MLA 与 KDA 混合出现（按架构规格）。',
    quoteUrl: 'https://kimi.com/blog/kimi-k2',
    tag: 'official',
    numExample: '实算示例：第 3 个 token「暗」的 q 与前 3 个 token 的 k 点积后除以 √8，得原始分 [0.97,0.18,1.61]，真 softmax 出权重 [0.299,0.135,0.566]，和恒为 1。'
  },

  router: {
    one: '路由器是 MoE 的分诊台：给每个 token 打分，从 896 个专家里挑出最对口的 16 个（外加共享专家）来处理它。',
    analogy: '像一家 896 位专科医生坐诊的超级医院：每个病人（token）只挂 16 个最相关的号——既看得起病（算力省），又看得好（总容量大）。分诊依据是分数的分位数，而非人工调的均衡规则。',
    principle: 'router 对 token 表示做线性打分得到 router-score，取 top-16 激活对应专家。K3 用 Quantile Balancing：直接由分数分位数推导专家分配，消除了启发式负载均衡更新和那个敏感的均衡超参数——训练更稳，专家利用率更均。容量（总参数）与成本（激活参数）由此解耦。',
    formula: 'score = softmax(x·W_g)；激活 = Top-16(score) + 共享专家；分配按分位数确定（Quantile Balancing，官方）',
    quote: 'derives expert allocation directly from router-score quantiles, eliminating heuristic updates and a sensitive balancing hyperparameter',
    quoteUrl: 'https://kimi.com/blog/kimi-k3',
    tag: 'official',
    numExample: '实算示例：「暗」对 896 个专家真打分，#180 以 2.40 居首，第 16 名 1.87 入选、第 17 名 1.82 落选；16 个分数再过 softmax 归一化，#180 得权重 0.082，Σw=1。'
  },

  experts: {
    one: '896 个专家 = 896 套独立的前馈网络；每个 token 只激活 16 个 + 共享专家，所以万亿总参数每次只算很小一部分。',
    analogy: '专科医生各管一摊：有的专家擅长代码括号，有的擅长法律措辞，有的专管数字计算；共享专家像全科医生，每个病人都看，保底通用能力。',
    principle: '每个专家是一个标准 FFN（升维→激活→降维）。稀疏激活让"知识容量"（总参数）和"每 token 算力"（激活参数）解耦：K3 为 896 选 16 + 共享（官方）；K2 为 384 选 8+1（官方）。训练中专家自发分化出领域与语法分工，这不是人工指定的。',
    formula: 'MoE 输出 = Σ_{i∈Top-16} g_i·FFN_i(x) + FFN_shared(x)，其中 FFN(x) = W_2·σ(W_1·x)',
    quote: 'K3：896 个专家选 16 个，另有共享专家；K2：384 选 8+1（均为官方事实，转述）。',
    quoteUrl: 'https://kimi.com/blog/kimi-k3',
    tag: 'official',
    numExample: '实算示例：专家 #180 的激活 tanh(x·W₁₈₀)=0.98，16 个专家的激活按门控权重加权求和 y=Σwᵢ·Eᵢ(x)，真加出 8 维输出 [0.23,0.03,0.54,0.38,…] 送往输出头。'
  },

  attnres: {
    one: 'AttnRes（注意力残差）改进了经典残差连接：深层不再"所有层输出等权累加"，而是跨深度有选择地取回某一层的表示。',
    analogy: '普通残差像流水席传菜：每道菜都倒进同一口锅，越到后面味道越混。AttnRes 像美食评审——能回头点名"我要第 7 层那道菜"，按需取用，不照单全收。',
    principle: '深层网络里残差流逐层均匀累加，早期层的关键信号会被层层稀释（残差流同质化）。AttnRes 用注意力式的检索机制跨深度选择表示：每一层可以按权重取回前面任意层的输出，让信息通路和梯度通路都更干净，这也是 K3 能稳定堆得更深的原因之一。',
    formula: 'h_l = Σ_{j≤l} α_{l,j}·h_j，α 由跨深度检索产生；均匀残差是 α 全相等的特例［机制为官方，权重形式为教学简化］',
    quote: 'selectively retrieves representations across depth rather than accumulating them uniformly',
    quoteUrl: 'https://kimi.com/blog/kimi-k3',
    tag: 'official'
  },

  norm: {
    one: '归一化层在每一层前后把向量数值拉回稳定范围，防止信号越传越大或越传越小。',
    analogy: '像音响系统的自动音量控制：不管上一句喊得多大声，进下一级设备前先调回标准音量——不爆音、也不闷掉。',
    principle: '主流采用 RMSNorm：按向量的均方根做缩放，再乘可学习增益。它不显式去均值，比 LayerNorm 更省。Pre-Norm 结构（先归一化再进注意力/FFN）让数十层的深塔可以稳定训练，梯度不炸不消失。',
    formula: 'RMSNorm(x) = x / √(mean(x²) + ε) · γ',
    quote: '归一化是深塔能堆起来的隐形地基：每层信号都在同一量级上对话。',
    quoteUrl: 'https://arxiv.org/abs/1910.07467',
    tag: 'simplified'
  },

  output: {
    one: '输出头把塔顶的最终向量投影回词表大小的打分向量——词表里每个候选"下一个词"各得一分。',
    analogy: '像阅卷老师给整张志愿表打分：词表里几万个词都是候选，输出头一次性给所有志愿打分，分高者胜出。',
    principle: '输出投影矩阵把 d_model 维向量映射到 V 维 logits（常与嵌入矩阵共享权重以省参数、稳训练）。注意 logits 是原始分数、不是概率，必须经过 softmax 归一化；分数差 1 分，概率可能差好几倍。',
    formula: 'logits = h·W_out + b，W_out ∈ ℝ^{d_model×V}（常与嵌入矩阵 E 共享）',
    quote: '输出头是几何回到语言的出口：一个向量重新变成"下一个词该是谁"的候选名单。',
    quoteUrl: 'https://platform.kimi.com/docs',
    tag: 'simplified'
  },

  softmax: {
    one: 'softmax 把输出头的原始分数变成总和为 1 的概率分布，模型就按这个分布挑下一个词。',
    analogy: '像把评委原始分换算成得票率：高分候选拿走绝大多数概率，但低分候选仍留着小尾巴——正是这条尾巴，让采样时偶尔有"惊喜"而不是永远最保守。',
    principle: '指数化既保持分数大小顺序，又放大差距；除以总和归一化成概率。温度 T 调节"确定性"：T→0 时近似总选最高分（贪婪、稳定但死板），T 大时分布变平、输出更随机多样。推理时从分布采样或直接取最大。',
    formula: 'p_i = e^{z_i} / Σ_j e^{z_j}；带温度版本：p_i = e^{z_i/T} / Σ_j e^{z_j/T}',
    quote: 'softmax 是模型"做决定"的瞬间：从几万个可能里，按概率抽出一个字。',
    quoteUrl: 'https://arxiv.org/abs/1706.03762',
    tag: 'simplified',
    numExample: '实算示例：11 个候选字里「。」logit 最高（4.42），T=1.0 时概率 67.3%；T=0.3 分布变尖升到 99.5%，T=2.0 变平降到 35.6%；top-p=0.95 时 11 个候选只留 5 个再归一化抽签。'
  },

  autoregressive: {
    one: '自回归：模型一次只生成一个 token，把它接回输入末尾，再生成下一个——逐字写作，循环往复。',
    analogy: '像用输入法码字：每选定一个候选字，它就变成上下文的一部分，接着影响下一个候选。你看到的"打字机效果"不是界面特效，是模型物理上就在一个字一个字算。',
    principle: '整句的联合概率按链式法则分解为逐步条件概率之积：生成 N 个 token 就需要 N 次完整前向。这解释了长回复为什么慢、流式输出为什么天然存在；而 KDA 的固定大小记忆让第 100 万步和第 1 步一样快，长程生成不退化。',
    formula: 'P(x_1,…,x_T) = Π_t P(x_t | x_1,…,x_{t-1})',
    quote: '自回归是大模型的"心跳"：每跳一次，吐出一个 token。',
    quoteUrl: 'https://arxiv.org/abs/1706.03762',
    tag: 'simplified'
  },

  'memory-vs-kvcache': {
    one: '记历史有两种办法：传统注意力给每个 token 留一张便签（KV 缓存，越攒越多）；KDA 只用一块固定大小的记忆矩阵（写满就按规则改写）。',
    analogy: 'KV 缓存像传话游戏的手抄记录：队伍每多一人，记录就长一行，重述越来越慢、纸越用越多。KDA 记忆矩阵像一支可倒带的录音笔：容量固定，听第 100 万句和第 10 句一样快——代价是录的是"压缩版"，细节可能被新内容冲淡。',
    principle: 'KV 缓存的显存随上下文长度线性增长，是长文本部署的第一瓶颈：K3 官方推荐 64+ 卡 supernode 部署，整模型显存需求约 650GB–1TB［估算］。KDA 状态大小只取决于模型宽度、与长度无关；官方还承诺 vLLM 的 KDA prefix cache 支持随权重（7·27 前开源）一并放出，相同前缀不必重算。',
    formula: 'KV 缓存：O(T·d) 随长度增长　vs　KDA 状态：O(d²) 与 T 无关［复杂度对比为简化表述］',
    quote: '官方：权重于 7·27 前开源，vLLM KDA prefix cache 支持随权重发布（转述）；650GB–1TB 显存为估算值。',
    quoteUrl: 'https://kimi.com/blog/kimi-k3',
    tag: 'inferred'
  },

  quant: {
    one: 'K3 从 SFT 阶段起就做量化感知训练（QAT）：权重压到 MXFP4（4 bit）、激活 MXFP8，让万亿模型真的能部署落地。',
    analogy: '像把 RAW 原片转成高质量 JPEG：文件小几倍，画质几乎看不出差别。QAT 相当于"拍照时就按 JPEG 的特性调相机"，比拍完再硬压（训练后量化）保真得多。',
    principle: '4 bit 这种极限压缩下，训完再量化掉点明显；K3 把 MXFP4 权重、MXFP8 激活的舍入直接放进 SFT 起的训练循环里，让模型学着在量化世界里工作。配合 Per-Head Muon 优化器与 SiTU，实现了相对 K2 约 2.5× 的 scaling 效率；权重量化到 4 bit 意味着显存占用约为 16 bit 的 1/4。',
    formula: 'x_q = round(x / s)·s，s 为分组共享的缩放因子（MXFP4：每 32 元素一组）［格式细节为公开规范，应用细节以官方为准］',
    quote: '官方：QAT 自 SFT 起使用 MXFP4 权重 / MXFP8 激活，Per-Head Muon + SiTU，scaling 效率 2.5× vs K2（转述）。',
    quoteUrl: 'https://kimi.com/blog/kimi-k3',
    tag: 'official'
  },

  // ─── v2 新增组件 ───

  bpe: {
    one: 'BPE（字节对编码）是最主流的分词算法：从单个字节出发，反复把出现频率最高的相邻片段合并成新符号，直到词表达标。',
    analogy: '像乐高拼字：常用词是一整块积木，生僻字只能用小颗粒现拼。"人工智能"也许两块搞定，一个生僻汉字可能要 3 块碎片——因为 UTF-8 里一个汉字占 3 个字节。',
    principle: '训练是纯统计工程：统计语料中相邻符号对的频率，贪心合并直到词表达标，没有损失函数。编码时按合并优先级最长匹配。为什么中文一个字可能拆成多个 token：词表名额有限，英文单词、代码符号和高频中文词抢走了整块名额，低频汉字分不到，只能退回字节级碎片拼装——最坏 1 个汉字 = 3 个 token。同样一段话，切得越碎，消耗的 token 越多，上下文越不经用。',
    formula: '训练：repeat { (a,b) = argmax_{(a,b)} freq(a,b)；词表加入新符号 ab } 直到 |V| 达标；编码：按合并顺序最长匹配。最坏情况：1 汉字 = 3 token（UTF-8 3 字节）。Kimi 词表大小与切分细节未公开［推断：按主流实践估为 10 万～20 万级，多语混合词表］',
    quote: '分词是概率登场前的最后一次"确定性"：切法一旦定下，模型看到的世界形状就定了。',
    quoteUrl: 'https://arxiv.org/abs/1508.07909',
    tag: 'inferred'
  },

  rope: {
    one: 'RoPE（旋转位置编码）给每个位置一个专属旋转角度：把 Q/K 向量按所在位置旋转，两个向量的点积自然只反映它们的相对距离。',
    analogy: '像给每个词发一个时钟指针：位置越靠后指针转得越多。两个词隔多远 = 两根指针的夹角——模型看夹角就知道距离，不管它们出现在第几句话。',
    principle: '老办法（GPT-3 的学习式绝对位置嵌入）把位置当编号查表相加：训练没见过的更长位置直接抓瞎，无法外推。RoPE 把位置编码成旋转：q 和 k 点积的结果只依赖相对位移 m−n，天生表达相对距离；不同维度用不同转速（高频管近处、低频管远处），再配合 NTK/YaRN 等插值手段把转角"减速"，训练时只见过几万长度的模型也能外推到 1M 上下文。1M 上下文正是"RoPE 管位置外推 + KDA 线性注意力管长度开销"两条腿走路的结果（Llama、Qwen、K 系均为 RoPE 路线）。',
    formula: 'q_m = R(Θ,m)·W_q·x_m，k_n = R(Θ,n)·W_k·x_n；⟨q_m, k_n⟩ = f(x_m, x_n, m−n)，只含相对位移；第 i 维转角 m·θ_i，θ_i = base^(−2i/d)（base 常取 10000，长上下文模型调大）',
    quote: '加法式位置编码是"贴编号"，RoPE 是"给方向"——位置从此是几何关系，不是门牌号。',
    quoteUrl: 'https://arxiv.org/abs/2104.09864',
    tag: 'simplified'
  },

  gate: {
    one: '门控是 MoE 的打分器：一个小神经网络给每个专家打分 g(x)=softmax(W_g·x)，只留 top-k 高分的专家上场，其余本轮休息。',
    analogy: '像选秀评委举牌：896 位选手，评委（门控矩阵 W_g）瞬间打完分，只有前 16 名晋级——打分本身也要学，它和网络其它部分一起训练。',
    principle: '打分经 softmax 变成"选择意愿强度"，top-k 之外的专家直接不算，算力就这么省下来。但 softmax 打分有马太效应：热门专家越练越强、冷门专家饿死（负载失衡）。两派解法：DeepSeek 系的 aux-loss-free 给打分加一个无梯度偏置项，按负载动态升降（不污染主损失）；K3 的 Quantile Balancing 更激进——直接按分数分位数推导专家分配，从根上消掉均衡超参数（官方）。',
    formula: 'g(x) = softmax(W_g·x)，W_g ∈ ℝ^{d_model×N}；激活 = TopK(g(x))。aux-loss-free：s_i = x·w_i + b_i，b_i 按专家负载无梯度更新；Quantile Balancing：按 router-score 分位数切分分配（K3 官方）',
    quote: 'derives expert allocation directly from router-score quantiles, eliminating heuristic updates and a sensitive balancing hyperparameter',
    quoteUrl: 'https://kimi.com/blog/kimi-k3',
    tag: 'official'
  },

  aggregate: {
    one: '聚合把被选中的 k 个专家的输出按门控分数加权求和：y = Σ g_i(x)·E_i(x)，重新混成一个向量，继续往下一层走。',
    analogy: '像会诊汇总：16 位专科医生各自给出意见，按"这病例跟你多相关"的权重合并成一份诊断——权重就是门控刚才打的分，谁的领域谁声音大。',
    principle: '加权求和是 MoE 成立的关键：专家输出乘以自己的门控分再相加，相关性自然变成话语权。因为每个 token 只算 k 个专家，单次前向成本 ≈ k 份 FFN；但模型的知识容量 = 全部 N 个专家。"容量按全量计费、成本按零头计费"——这就是"大而不贵"：K3 背下 896 个专家的知识（2.8T 总参），每 token 只付 16 个专家的算力（官方）。',
    formula: 'y = Σ_{i∈TopK} g_i(x)·E_i(x) + E_shared(x)，E_i(x) = W_2·σ(W_1·x)；FLOPs ∝ k·|FFN|，容量 ∝ N·|FFN|，k≪N（K3：k=16，N=896，官方）',
    quote: '聚合让"万亿参数、百亿计算"同时成立：按相关性调配方子，而不是每次吃下整个药房。',
    quoteUrl: 'https://arxiv.org/abs/1701.06538',
    tag: 'simplified'
  },

  sampler: {
    one: '采样器决定"从概率分布里怎么挑下一个词"：temperature 调节分布胖瘦，top-p / top-k 先把长尾候选砍掉，再在剩下的里抽签。',
    analogy: '像点菜策略：temperature 低 = 永远点招牌菜（稳但无聊），temperature 高 = 敢试冷门菜（惊喜或翻车）。top-k 像"只看菜单前 40 名"，top-p 像"只看销量累计到 95% 的部分"——先把黑暗料理挡在门外，再凭运气抽。',
    principle: '三者串联作用：① temperature 先缩放 logits——T<1 分布变尖（更确定）、T>1 变平（更随机）、T=1 原样；② top-k 只保留分数最高的 k 个；③ top-p（nucleus）保留累计概率达 p 的最小候选集，能随分布形状自适应（尖分布候选少、平分布候选多）；截断后剩余概率重新归一化，再真正随机抽一个。top-k 是"固定名额"、top-p 是"固定份额"，实际部署常只用 top-p 或两者叠加。K3 官方推荐固定 t=1.0、top_p=0.95（官方）。',
    formula: 'p_i = e^{z_i/T} / Σ_j e^{z_j/T}；top-k：V′ = TopK(p, k)；top-p：V′ = 最小集合 s.t. Σ_{i∈V′} p_i ≥ p；截断后 p 重新归一化再采样。K3：T=1.0、top_p=0.95（官方）',
    quote: '采样是模型的"性格旋钮"：同一副脑子，参数一换，可以是会计也可以是诗人。',
    quoteUrl: 'https://arxiv.org/abs/1904.09751',
    tag: 'simplified'
  },

  training: {
    one: '训练分三幕：预训练学"世界知识和语言规律"，SFT（监督微调）学"听懂指令、按格式回答"，RL（强化学习）学"答得对、答得好"。',
    analogy: '像培养医生：预训练 = 医学院苦读十年（海量教材，全参数更新）；SFT = 跟师出门诊（看标准问答示范，学规矩）；RL = 独立行医后按疗效反馈自我改进（奖励信号驱动）。',
    principle: '三阶段练的是同一副参数，只是教材和评分标准换三轮：预训练用"预测下一个 token"在万亿级语料上更新全部参数，成本占大头——K3 预训练用 Per-Head Muon 优化器与 SiTU（官方）；SFT 换成高质量问答对，教指令遵循与对话格式，K3 从这一阶段起就做量化感知训练（QAT：MXFP4 权重 / MXFP8 激活，官方），模型从学"怎么回答"起就活在量化世界里，而不是训完再硬压；RL/后训练以奖励信号（RLHF 或可验证奖励 RLVR）用 PPO/GRPO 类算法微调，KL 散度拴住模型不许偏离参考策略太远——K3 在长程 agent 任务上训练，并保留完整 thinking history 供后续步骤学习（官方）。参数量直觉：预训练 = 全参数、万亿 token；SFT = 全参数、百万级样本；RL = 全参数但步数最少，只调"行为"不补"知识"。',
    formula: '预训练/SFT：L = −Σ_t log P(x_t | x_{<t})（同一目标，教材不同）；RL：max E[r(x,y)] − β·KL(π_θ ‖ π_ref)（KL 项防跑偏）；K3：QAT 自 SFT 起（官方）',
    quote: '同一个网络，三轮教案：先博学，再守规矩，后出成绩。',
    quoteUrl: 'https://arxiv.org/abs/2203.02155',
    tag: 'simplified'
  },

  'vit-patch': {
    one: 'ViT 把图片切成固定大小的小方块（patch），每块拉平、投影成一个向量——图像从此变成一串"视觉 token"，后面的事就和语言模型一样了。',
    analogy: '像把照片裁成棋盘格邮票：每张邮票展平、盖章（位置嵌入），排队进 transformer。模型不再"看整张图"，而是读这一叠邮票之间的关系。',
    principle: '标准 ViT 用 16×16 像素 patch：224×224 图 → 14×14=196 个 patch，各经同一个线性投影变成 d_model 维向量，加上可学习位置嵌入，序列最前面再塞一个 [CLS] 汇总 token 用于分类。这套"切块→嵌入"的输入侧设计是 CV transformer 的通用底座：SAM 的图像编码器（ViT-H，1024×1024 → 64×64 特征格）是它的直系后代；YOLO 虽走卷积路线，输入侧同样把图像离散成多尺度网格逐格处理——"图像格子化"是两家共用原理。',
    formula: 'x_p ∈ ℝ^{16×16×3} 拉平 → z_i = W·x_p^{(i)} + e_pos^{(i)}，W ∈ ℝ^{768×d_model}；224² 图 → 196 patch + 1 [CLS] = 197 token（ViT-B/16 官方配置）',
    quote: '一张图等于 196 个词——ViT 的洞见就是把 CV 翻译成 NLP。',
    quoteUrl: 'https://arxiv.org/abs/2010.11929',
    tag: 'simplified'
  },

  'yolo-head': {
    one: 'YOLO 检测头在多尺度特征网格上"每格直接报答案"：这格有没有目标、框在哪、是什么类——anchor-free，不再靠预设锚框去猜。',
    analogy: '像按答题卡格子阅卷：图片被划成 80×80、40×40、20×20 三种粗细的格子（管小/中/大目标），每格独立作答"我负责的区域：有猫，框在这，置信 92%"。分割版再给每个目标发 32 个"面具系数"。',
    principle: 'anchor-free（YOLOv8 起）：每个 grid cell 直接回归目标中心与框尺寸，省掉锚框聚类调参；三个尺度（P3/P4/P5，步长 8/16/32）分工看小、中、大目标。实例分割走 proto 路线：一个 Proto 分支产出全图共享的 32 张原型掩码，检测头给每个目标预测 32 个系数，线性组合即得该目标的掩码——无论图里多少个目标，成本几乎不增长。',
    formula: '框回归：中心 = cell 原点 + Δ，宽高按 stride 缩放；mask_i = Σ_{k=1..32} c_{i,k}·Proto_k（Proto 全图共享，c 为每目标系数）；grid 步长 8/16/32（P3/P4/P5，输入 640²）',
    quote: '一次前向、全图作答：YOLO（You Only Look Once）把检测变成一场网格问答。',
    quoteUrl: 'https://docs.ultralytics.com/',
    tag: 'simplified'
  },

  'sam-decoder': {
    one: 'SAM 的掩码解码器把你的提示（点/框）和图像特征放进两路交叉注意力里"对谈"两轮，毫秒级吐出目标掩码。',
    analogy: '像你指着照片问"这块是什么"：手指（提示 token）和照片（图像特征）互相提问——提示问图像"我落在哪类东西上"，图像问提示"你到底要哪一个"，两轮对谈后交出精确轮廓。',
    principle: 'decoder 是轻量两路 transformer（仅 2 层）：prompt token（点/框的位置编码 + 几个可学习输出 token）与图像嵌入互为 Q 和 KV 做双向交叉注意力——提示被图像特征更新，图像特征也被提示更新，每轮都让彼此更懂对方。最后输出 token 经 MLP 后与图像特征逐点相乘、上采样成掩码；同时预测 3~4 个候选掩码和各自的 IoU 置信分，消解"一个点可能对应多个目标"（整个苹果 vs 苹果梗）的歧义。',
    formula: '两路交叉注意力 ×2 层：P′ = P + Attn(Q=P, K=V=I)；I′ = I + Attn(Q=I, K=V=P′)；mask = Upsample(I″)·MLP(t_out)；同时输出 k 个候选 mask + IoU 分（默认 3~4 个）',
    quote: '提示进、掩码出，中间那两轮"对谈"就是 SAM 理解你意图的全部过程。',
    quoteUrl: 'https://arxiv.org/abs/2304.02643',
    tag: 'simplified'
  }
};
