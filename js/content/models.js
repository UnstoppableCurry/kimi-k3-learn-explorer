// content/models.js — 17 个模型的中文内容卡
// 契约：window.CONTENT_MODELS = { id: {overview, strengths[], vs_k3, facts[]} }
// facts 每行 ['指标', '值', 'official'|'estimated']；official=官方公开数据，estimated=社区估算/待确认
window.CONTENT_MODELS = {

  k3: {
    overview: 'Kimi K3 是月之暗面 2026 年 7 月发布的开源旗舰：2.8T 总参数、896 个专家每次只激活 16 个，混合 KDA 线性注意力与 MLA 全注意力，原生支持 1M token 上下文，MXFP4 量化训练。',
    strengths: [
      'KDA 线性注意力把长上下文开销从平方级降到线性，1M 上下文推理成本大幅低于纯全注意力模型',
      '896 选 16 的细粒度 MoE 让总参数 2.8T 而每 token 激活量可控，容量与推理成本解耦',
      'BrowseComp 智能体浏览基准 90.4 分，开源模型总排第四、前端任务排名第一',
      'MXFP4 低精度训练 + AttnRes 注意力残差结构，训练稳定性与推理效率兼顾'
    ],
    vs_k3: 'K3 是本站基准模型：KDA + MLA 混合注意力、896 选 16 MoE、AttnRes 残差、1M 上下文。其它模型的「对比 K3」栏目均以此为参照系。',
    facts: [
      ['总参数量', '2.8T', 'official'],
      ['MoE 专家', '896 选 16', 'official'],
      ['上下文长度', '1M token', 'official'],
      ['注意力结构', 'KDA 线性 + MLA 全注意力混合', 'official'],
      ['开源日期', '2026-07-27', 'official'],
      ['BrowseComp', '90.4（总排第四，前端 #1）', 'official'],
      ['训练量化', 'MXFP4', 'official'],
      ['注意力残差', 'AttnRes 结构', 'official']
    ]
  },

  k2: {
    overview: 'Kimi K2 是 K3 的前代开源旗舰：约 1T 总参数、激活 32B，384 个专家选 8 个加 1 个共享专家，全部层使用 MLA 全注意力，128K 上下文，以 MuonClip 优化器稳定训练。',
    strengths: [
      '384 选 8+1（含共享专家）的 MoE 设计在当时把开源模型规模推到 1T 量级',
      'MuonClip 优化器在 1T 规模上实现了无 loss 尖峰的稳定训练',
      'MLA 压缩 KV 缓存，128K 上下文下推理显存占用远低于标准 MHA',
      'Modified MIT 协议开放权重，商用门槛低'
    ],
    vs_k3: '对比 K3：K2 没有 KDA 线性注意力，全部层用 MLA 全注意力，上下文只有 128K（K3 为 1M）；专家 384 选 8+1 而非 896 选 16；也没有 AttnRes 结构。把两个模型并排可直观看到 K3 新增的三类组件。',
    facts: [
      ['总参数量', '约 1T', 'official'],
      ['激活参数', '32B', 'official'],
      ['MoE 专家', '384 选 8 + 1 共享', 'official'],
      ['注意力结构', 'MLA 全注意力（全部层）', 'official'],
      ['优化器', 'MuonClip', 'official'],
      ['上下文长度', '128K', 'official'],
      ['开源协议', 'Modified MIT', 'official']
    ]
  },

  k15: {
    overview: 'Kimi K1.5 是月之暗面 2025 年初发布的多模态推理模型，主打长链思维（long-CoT）强化学习，在数学与代码推理榜单上对标当时的 OpenAI o1 系列。',
    strengths: [
      '长上下文 RL：把 128K 窗口用于长链思维展开，长推理不截断',
      '多模态联合训练，文本与视觉推理共用一套思维链',
      '提出 long2short 方法：把长思维链模型的能力蒸馏进短链模型，省推理成本',
      '数学竞赛类基准（AIME、MATH-500）成绩与 o1 系列相当'
    ],
    vs_k3: '对比 K3：K1.5 是 dense 推理特化模型，参数规模远小于 K3 的 2.8T MoE，无 KDA 线性注意力，也无 1M 上下文；它的价值在长链 RL 训练方法，而 K3 把重心放在架构效率与智能体能力上。',
    facts: [
      ['模型类型', '多模态推理（long-CoT）', 'estimated'],
      ['参数规模', '未公开（估算远小于 K3）', 'estimated'],
      ['RL 上下文', '最长 128K 思维链', 'estimated'],
      ['训练方法', '长上下文强化学习 + long2short 蒸馏', 'estimated'],
      ['对标模型', 'OpenAI o1 系列', 'estimated'],
      ['发布时间', '2025 年初', 'estimated']
    ]
  },

  glm52: {
    overview: 'GLM-5.2 是智谱 AI 的旗舰开源模型，估算约 700B 总参数、激活约 40B 的 MoE 架构，200K 上下文，使用标准全注意力，公开结构细节有限，本站数值以社区估算为准。',
    strengths: [
      '估算约 700B 的 MoE 规模在开源阵营属第一梯队',
      '200K 长上下文支持，配合智谱的 Agent 工具链生态',
      '延续 GLM 系列的双语（中英）训练传统，中文任务表现扎实',
      '开放权重，可本地部署与二次微调'
    ],
    vs_k3: '对比 K3：GLM-5.2 无 KDA 线性注意力，使用标准全注意力；无 AttnRes 结构；上下文 200K 对 K3 的 1M。数值多为社区估算，橙色「估算」标记的内容请谨慎采信。',
    facts: [
      ['总参数量', '约 700B', 'estimated'],
      ['激活参数', '约 40B', 'estimated'],
      ['架构', 'MoE + 标准全注意力', 'estimated'],
      ['上下文长度', '200K', 'estimated'],
      ['开源状态', '开放权重', 'estimated'],
      ['发布方', '智谱 AI', 'estimated']
    ]
  },

  minimax3: {
    overview: 'MiniMax M3 是 MiniMax 的开源旗舰，估算约 2.7T 总参数、激活约 50B，以 Lightning Attention 类线性注意力支撑 1M 级超长上下文，结构细节多为社区估算。',
    strengths: [
      '线性注意力路线与 K3 相同，1M 级上下文下推理成本远低于全注意力模型',
      '估算约 2.7T 总参数，与 K3 同属「超大总参、小激活」设计流派',
      '超长上下文实测（大海捞针类测试）召回率高',
      '开放权重，附带 Agent 与代码场景的官方工具链'
    ],
    vs_k3: '对比 K3：两者都走线性注意力 + 超长上下文路线，但 M3 用 Lightning Attention 类机制替代 KDA；估算激活 50B 高于 K3 的每 token 激活量，推理成本更高；无公开证据显示其有 AttnRes 结构。',
    facts: [
      ['总参数量', '约 2.7T', 'estimated'],
      ['激活参数', '约 50B', 'estimated'],
      ['注意力结构', 'Lightning Attention 类线性注意力', 'estimated'],
      ['上下文长度', '约 1M 级', 'estimated'],
      ['开源状态', '开放权重', 'estimated'],
      ['发布方', 'MiniMax', 'estimated']
    ]
  },

  deepseekv4: {
    overview: 'DeepSeek V4 是深度求索的新一代旗舰，延续 MLA + MoE 技术路线并进一步扩大规模，公开细节有限，本站参数数值以社区估算标注，待官方技术报告确认。',
    strengths: [
      '延续 V3 验证过的 MLA + aux-loss-free MoE 路线，工程风险低',
      '估算进一步扩大专家数量与总参数，容量上限更高',
      'DeepSeek 系列一贯的训练成本控制，单位算力产出高',
      '开放权重预期延续，社区生态成熟'
    ],
    vs_k3: '对比 K3：V4 沿用 MLA 全注意力而非 K3 的 KDA 混合路线，超长上下文效率预计弱于 K3；两者 MoE 规模同为第一梯队，但 K3 已公布 1M 上下文与 BrowseComp 90.4 实测，V4 多数指标仍待官方确认。',
    facts: [
      ['总参数量', '未公开（估算超 V3 的 671B）', 'estimated'],
      ['架构路线', 'MLA + MoE（延续 V3）', 'estimated'],
      ['注意力结构', 'MLA 全注意力', 'estimated'],
      ['负载均衡', 'aux-loss-free（预计延续）', 'estimated'],
      ['开源状态', '待官方确认', 'estimated'],
      ['发布方', '深度求索 DeepSeek', 'estimated']
    ]
  },

  deepseekv3: {
    overview: 'DeepSeek V3 是 2024 年底发布的开源 MoE 模型：671B 总参数、激活 37B，256 个专家选 8 个加 1 个共享专家，MLA 注意力，首创无辅助损失的负载均衡策略。',
    strengths: [
      'aux-loss-free 负载均衡：不靠额外损失函数就能让 256 个专家均匀干活，训练更干净',
      'MLA 把 KV 缓存压成低秩向量，长文本推理显存大幅下降',
      '671B 规模仅用约 2.8M H800 GPU 小时完成训练，成本远低于同级别模型',
      '多 token 预测（MTP）训练目标提升数据效率，推理时可用于投机解码加速'
    ],
    vs_k3: '对比 K3：V3 是 K3 混合架构的「MLA 半边」——全层 MLA 无 KDA，上下文 128K 对 K3 的 1M；专家 256 选 8+1 比 K3 的 896 选 16 粗；总参数 671B 约为 K3 的四分之一，激活 37B 则更高。',
    facts: [
      ['总参数量', '671B', 'official'],
      ['激活参数', '37B', 'official'],
      ['MoE 专家', '256 选 8 + 1 共享', 'official'],
      ['注意力结构', 'MLA', 'official'],
      ['负载均衡', 'aux-loss-free', 'official'],
      ['上下文长度', '128K', 'official'],
      ['训练成本', '约 2.8M H800 GPU 小时', 'official'],
      ['训练目标', '多 token 预测（MTP）', 'official']
    ]
  },

  qwen3: {
    overview: 'Qwen3 是阿里巴巴通义千问第三代开源系列，提供从 0.6B 到 235B 的完整尺寸梯队，旗舰为 235B-A22B MoE，支持思维模式与非思维模式双切换，协议友好。',
    strengths: [
      '0.6B 到 235B 八个尺寸全覆盖，手机端到服务器端都有可用档',
      '单模型内切换「思考 / 非思考」两种模式，延迟与质量按需取舍',
      'Apache 2.0 协议，商用与二次分发几乎无限制',
      '119 种语言支持，多语覆盖在开源模型中最广'
    ],
    vs_k3: '对比 K3：Qwen3 旗舰 235B-A22B 总参数不到 K3 的十分之一，用标准 GQA 全注意力而非 KDA 混合，上下文 128K 对 1M；它的优势在尺寸梯队完整与 Apache 2.0 协议，K3 的优势在单模型能力上限与超长上下文。',
    facts: [
      ['旗舰总参数', '235B（激活 22B）', 'estimated'],
      ['尺寸梯队', '0.6B / 1.7B / 4B / 8B / 14B / 32B / 30B-A3B / 235B-A22B', 'estimated'],
      ['注意力结构', 'GQA 全注意力', 'estimated'],
      ['上下文长度', '128K', 'estimated'],
      ['推理模式', '思考 / 非思考双模式', 'estimated'],
      ['开源协议', 'Apache 2.0', 'estimated'],
      ['语言支持', '119 种', 'estimated']
    ]
  },

  llama3: {
    overview: 'Llama 3 是 Meta 2024 年发布的开源 dense 模型系列，旗舰 405B 参数，采用分组查询注意力（GQA）与 RoPE 旋转位置编码，128K 上下文，是现代 dense 开源模型的标杆。',
    strengths: [
      '405B dense 结构简单可预测：无 MoE 路由问题，微调与部署工具链最成熟',
      'GQA 在几乎不损质量的前提下把 KV 缓存缩小数倍，已成行业标准配置',
      'RoPE 位置编码外推性好，配合长上下文微调扩展到 128K',
      '15T token 高质量数据训练，数据清洗流程公开程度业内最高'
    ],
    vs_k3: '对比 K3：Llama 3 405B 是 dense 模型——每个 token 激活全部 405B 参数，推理成本远高于 K3 的 MoE 稀疏激活；用 GQA 全注意力而非 KDA 混合；128K 对 1M 上下文。它代表「暴力堆 dense」路线，K3 代表「稀疏 + 线性注意力」效率路线。',
    facts: [
      ['旗舰参数', '405B（dense）', 'official'],
      ['注意力结构', 'GQA 分组查询注意力', 'official'],
      ['位置编码', 'RoPE 旋转位置编码', 'official'],
      ['上下文长度', '128K', 'official'],
      ['训练数据', '15T token', 'official'],
      ['系列尺寸', '8B / 70B / 405B', 'official'],
      ['发布方', 'Meta（2024）', 'official']
    ]
  },

  gpt2: {
    overview: 'GPT-2 是 OpenAI 2019 年发布的自回归语言模型，最小档 117M 参数，纯 decoder-only Transformer，证明了「规模 + 无监督预训练」可以涌现零样本能力。',
    strengths: [
      'decoder-only 架构的奠基之作：后续 GPT 系列与绝大多数开源 LLM 都沿用此范式',
      '117M 小尺寸可在普通笔记本跑通，是学习 Transformer 内部结构的最佳教具',
      '零样本能力的首次系统展示：不微调也能做翻译、摘要、问答',
      '权重与代码完全公开，可复现性极佳'
    ],
    vs_k3: '对比 K3：GPT-2 的 117M 参数是 K3 的 2.4 万分之一，12 层 dense 全注意力对 K3 的混合架构，1024 token 上下文对 1M。两者放在一起，能直观看到 7 年间 LLM 在规模与架构上的跨度。',
    facts: [
      ['参数量', '117M（最小档）', 'official'],
      ['架构', 'decoder-only Transformer', 'official'],
      ['层数', '12 层', 'official'],
      ['注意力结构', '标准 MHA 全注意力', 'official'],
      ['上下文长度', '1024 token', 'official'],
      ['训练数据', 'WebText（约 40GB 网页）', 'official'],
      ['发布时间', '2019 年（OpenAI）', 'official']
    ]
  },

  gpt3: {
    overview: 'GPT-3 是 OpenAI 2020 年发布的 175B 参数 dense 自回归模型，以 in-context learning（上下文学习）惊艳业界：只给几个示例、不更新权重就能完成新任务。',
    strengths: [
      '175B dense 规模首次系统验证了 scaling law：越大越强的可预测性',
      'in-context learning：few-shot 提示即可迁移任务，开启了提示工程时代',
      '96 层、12288 维宽度的超大 dense 结构成为后续模型的参照系',
      '论文对 45TB 原始数据的清洗流程有详细披露'
    ],
    vs_k3: '对比 K3：GPT-3 的 175B 全部参数每 token 都激活，而 K3 只激活 16/896 个专家——同量级能力下推理成本差一个数量级；2048 token 上下文对 K3 的 1M。GPT-3 证明了 scale，K3 回答了 scale 之后怎么省。',
    facts: [
      ['参数量', '175B（dense）', 'official'],
      ['架构', 'decoder-only Transformer', 'official'],
      ['层数 / 宽度', '96 层 / d_model 12288', 'official'],
      ['注意力结构', '标准 MHA（含稀疏注意力变体）', 'official'],
      ['上下文长度', '2048 token', 'official'],
      ['核心能力', 'in-context few-shot learning', 'official'],
      ['发布时间', '2020 年（OpenAI）', 'official']
    ]
  },

  bert: {
    overview: 'BERT 是 Google 2018 年发布的 110M 参数 encoder-only 模型，用掩码语言建模（MLM）双向理解上下文，把 NLP 带入「预训练 + 微调」时代。',
    strengths: [
      '双向上下文：同时看左边和右边的词，理解类任务（分类、抽取）远超单向模型',
      'MLM 掩码训练目标简单有效，成为后续 RoBERTa 等无数变体的起点',
      '「预训练 + 微调」范式的普及者：一个底座适配所有下游任务',
      '110M 小尺寸至今仍是句向量、检索排序等工业场景的常客'
    ],
    vs_k3: '对比 K3：BERT 是 encoder-only——只能「理解」不能「生成」，与 K3 这类 decoder-only 生成模型是两条路线；110M 参数、512 token 上下文对 K3 的 2.8T 与 1M。看 BERT 的 3D 结构能明白：去掉因果掩码的注意力长什么样。',
    facts: [
      ['参数量', '110M（base 版）', 'official'],
      ['架构', 'encoder-only Transformer', 'official'],
      ['训练目标', '掩码语言建模（MLM）+ 下句预测', 'official'],
      ['层数 / 宽度', '12 层 / d_model 768', 'official'],
      ['上下文长度', '512 token', 'official'],
      ['训练数据', 'BooksCorpus + 英文维基', 'official'],
      ['发布时间', '2018 年（Google）', 'official']
    ]
  },

  transformer2017: {
    overview: 'Transformer 是 Google 2017 年论文《Attention Is All You Need》提出的原始架构：65M 参数、6 层编码器加 6 层解码器，第一次完全抛弃循环与卷积，只用注意力。',
    strengths: [
      '注意力机制的奠基之作：Q/K/V 三件套与多头注意力全部出自这篇论文',
      '并行训练：抛弃 RNN 的顺序依赖，训练速度数量级提升，大模型时代由此开启',
      'encoder-decoder 双塔结构完整展示了注意力在「理解」与「生成」两侧的用法',
      '65M 小尺寸，结构可在 3D 场景中完整呈现每一层，是理解一切后续模型的入口'
    ],
    vs_k3: '对比 K3：原始 Transformer 是 encoder-decoder 结构、sinusoidal 位置编码、post-norm 归一化，K3 是 decoder-only、混合 KDA/MLA、MoE 稀疏激活。从 65M 到 2.8T 的 4 万倍跨度里，注意力的核心公式却没变。',
    facts: [
      ['参数量', '65M（base 版）', 'official'],
      ['架构', 'encoder 6 层 + decoder 6 层', 'official'],
      ['注意力头数', '8 头', 'official'],
      ['位置编码', 'sinusoidal 正弦位置编码', 'official'],
      ['归一化', 'post-norm（残差后归一化）', 'official'],
      ['原始任务', '英德机器翻译（WMT14 BLEU 28.4）', 'official'],
      ['发布时间', '2017 年（Google）', 'official']
    ]
  },

  rwkv: {
    overview: 'RWKV 是完全不用注意力机制的开源架构：用 time-mix 与 channel-mix 两个模块模拟 RNN 的线性递推，同时保持 Transformer 式的并行训练，推理显存与长度无关。',
    strengths: [
      '无注意力：time-mix 用可学习的衰减系数递推状态，推理成本 O(1)/token，与历史长度无关',
      '训练时可并行（如 Transformer），推理时可递推（如 RNN），两头好处都占',
      'channel-mix 负责跨通道信息混合，替代 FFN 的角色，结构极简',
      '推理只需保存一个固定大小的状态向量，理论上上下文长度无限'
    ],
    vs_k3: '对比 K3：K3 用 KDA 线性注意力「折中」——保留注意力的表达能力、压低开销；RWKV 更激进，彻底没有注意力，纯 RNN 递推。两者都瞄准超长上下文成本问题，但 RWKV 省得更极致，长程精确检索能力则弱于带真注意力的 K3。',
    facts: [
      ['核心模块', 'time-mix + channel-mix', 'official'],
      ['注意力机制', '无（纯线性递推）', 'official'],
      ['推理复杂度', '每 token O(1)，与长度无关', 'official'],
      ['训练方式', '可并行（类似 Transformer）', 'official'],
      ['状态存储', '固定大小状态向量，无 KV 缓存', 'official'],
      ['开源状态', '开放权重（多代：RWKV-4/5/6/7）', 'official'],
      ['定位', 'RNN 复兴路线代表', 'official']
    ]
  },

  yolov8seg: {
    overview: 'YOLOv8-seg 是 Ultralytics 的单阶段实例分割模型：CSPDarknet 骨干提特征、PAN-FPN 颈部融合多尺度、解耦检测头加分支输出的原型掩码（proto mask）组合出逐像素分割结果。',
    strengths: [
      '单阶段设计：一次前向同时出检测框、类别与分割掩码，实时视频流可用',
      'proto mask 机制：共享一组掩码原型，每个实例只预测线性组合系数，分割几乎不加成本',
      'CSPDarknet 的跨阶段部分连接减半计算量同时保住梯度流',
      'PAN-FPN 双向特征金字塔，小目标与大目标兼顾'
    ],
    vs_k3: '对比 K3：这是 CV 模型——没有 token、没有注意力层堆叠，用卷积 + 特征金字塔做事；K3 从离散符号序列生成文本，YOLOv8-seg 从像素网格回归空间结构。放在对比模式里能看清「语言模型」与「视觉模型」数据形态的根本差异。',
    facts: [
      ['骨干网络', 'CSPDarknet', 'official'],
      ['特征融合', 'PAN-FPN 双向特征金字塔', 'official'],
      ['检测头', '解耦头（分类与回归分支分离）', 'official'],
      ['分割机制', 'proto mask 原型 + 逐实例系数', 'official'],
      ['任务类型', '实时实例分割', 'official'],
      ['输入形态', '像素网格（图像）', 'official'],
      ['发布方', 'Ultralytics（2023）', 'official']
    ]
  },

  sam: {
    overview: 'SAM（Segment Anything Model）是 Meta 2023 年发布的可提示分割基础模型：ViT-H 632M 参数的图像编码器把图编码成嵌入，再用轻量解码器按点/框提示即时生成任意物体掩码。',
    strengths: [
      '可提示分割：点一下或框一下就能切出任意物体，无需针对类别训练',
      'SA-1B 数据集：11 亿掩码，分割领域第一个「基础模型」规模的数据飞轮',
      '重编码器 + 轻解码器设计：图像嵌入算一次，多次提示实时响应',
      '零样本迁移到医学影像、遥感等未见领域仍可用'
    ],
    vs_k3: '对比 K3：SAM 与 K3 同为「基础模型」思路——大预训练 + 提示即用，但 SAM 的提示是点与框、输出是掩码，K3 的提示是文本、输出是文本。SAM 的 ViT-H 用标准全注意力处理图像 patch，恰好是 K3 里 MLA 部分的纯视觉版对照。',
    facts: [
      ['图像编码器', 'ViT-H（632M 参数）', 'official'],
      ['架构', '图像编码器 + 提示编码器 + 掩码解码器', 'official'],
      ['训练数据', 'SA-1B：1100 万图 / 11 亿掩码', 'official'],
      ['提示方式', '点 / 框 / 掩码 / 文本', 'official'],
      ['核心能力', '零样本可提示分割', 'official'],
      ['输入形态', '像素网格（图像 patch）', 'official'],
      ['发布时间', '2023 年（Meta）', 'official']
    ]
  },

  deepseekocr: {
    overview: 'DeepSeek-OCR 是深度求索的视觉-文本压缩模型：把长文档渲染成图像后用视觉编码器压缩成少量视觉 token，再让语言模型读图「还原文本，探索用光学压缩突破长上下文瓶颈。',
    strengths: [
      '光学压缩思路：一页文档压成几十到几百个视觉 token，压缩率约 10 倍仍保 97% 精度',
      '把 OCR 任务反过来用：不只为识字，更为验证「图像比文本 token 更省」的假设',
      '为 LLM 长上下文提供了全新路线：历史对话可渲染成图像存进上下文',
      '视觉编码器 + 语言模型解耦设计，两端可独立升级'
    ],
    vs_k3: '对比 K3：K3 用 KDA 架构在「文本 token」内解决 1M 上下文，DeepSeek-OCR 则尝试把文本转成图像来省 token——一个在架构层优化，一个在表示层绕行。两者都瞄准同一个物理约束：注意力成本随 token 数增长。',
    facts: [
      ['核心思路', '视觉-文本光学压缩', 'estimated'],
      ['压缩率', '约 10 倍压缩保 97% 精度', 'estimated'],
      ['架构', '视觉编码器 + 语言模型', 'estimated'],
      ['应用场景', '长文档 OCR / 长上下文压缩', 'estimated'],
      ['输入形态', '文档渲染图像', 'estimated'],
      ['发布方', '深度求索 DeepSeek', 'estimated']
    ]
  }
};
