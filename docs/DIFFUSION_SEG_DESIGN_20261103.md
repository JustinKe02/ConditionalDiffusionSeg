# 面向层厚有序结构的RGB条件高斯掩码扩散

更新与文献检索截止日期均为2026-11-03。本文记录研究依据、候选方法、实现约定和预先拟定的验证方案。候选方法的训练收益与材料泛化仍待实验检验。实现与工程检查见第8节。

## 1. 研究问题与证据范围

目标是在MoS₂光学图像的背景、单层、少层、多层四类分割中，检验层厚类别的有序关系能否改善RGB条件高斯掩码扩散。核心变量为被加噪掩码的状态表示。模型保留GLP的全局—局部图像条件与强特征提取能力。

当前历史R/E3R具有边界收益，但其结果对新方案仅提供问题背景。已有验证集受控比较中，R相对I的前景BF@2px提高7.5042个百分点，mIoU降低1.3629个百分点。去掉初态噪声的N与R接近。R的同一四步轨迹从首步到末步仅增加0.1591个百分点mIoU、2.0088个百分点BF。这些结果促使本方案直接检验掩码扩散的作用与层厚状态表示的作用。

历史数据来源为 `/root/autodl-tmp/ConditionalDiffusionSeg/outputs/analysis/chapter_controlled_ablation_20261031/comparison_summary.json` 的 `point_fraction`、`contrasts` 和 `trajectories`。其范围为训练seed42、历史开发验证集25张。I/N/F仅有验证结果，R另有历史test27结果。历史测试协议记录 `historical_test_exposure=true`，权重由验证集best固定。

本方案提出的可检验假设如下。

> 在匹配图像条件、训练曝光、更新数和采样成本后，将层厚类别表示为累计阈值掩码，有助于高斯去噪利用类别有序关系，改善薄层识别，并降低跨多个厚度类别的错误。

## 2. 已有研究与近期方法

### 2.1 条件分割扩散的方法参照

下表区分论文正式发表信息、预印本信息与代码可用性。论文中的结果和计算预算需在本数据上重新核验。

| 工作 | 截止日期内核验状态 | 与本方案有关的内容 | 原始来源与作者代码 |
|---|---|---|---|
| MedSegDiff | MIDL 2023，PMLR卷227于2024发布 | 动态条件与频域处理说明条件融合已有充分研究 | [论文](https://proceedings.mlr.press/v227/wu24a.html) · [代码](https://github.com/ImprintLab/MedSegDiff) |
| MedSegDiffV2 | AAAI 2024官方论文页 | Transformer条件融合、噪声预测及专用交互模块 | [论文](https://ojs.aaai.org/index.php/AAAI/article/view/28418) · [代码](https://github.com/ImprintLab/MedSegDiff) |
| HiDiff | IEEE TMI 2024，DOI 10.1109/TMI.2024.3424471 | 判别先验与Bernoulli扩散细化，终端分布和本方案的纯高斯初态不同 | [论文](https://arxiv.org/abs/2407.03548) · [代码](https://github.com/takimailto/HiDiff) |
| DG-DDM-Seg | IEEE TMI 2025，DOI 10.1109/TMI.2025.3564474 | 离散条件分布、鲁棒图像特征与掩码生成Transformer | [论文DOI](https://doi.org/10.1109/TMI.2025.3564474) · [代码](https://github.com/HeranYang/DG-DDM-Seg) |
| LDSeg | 2024预印本、2025修订，BSPC 114卷2026年109380 | 任务掩码自编码器与潜空间扩散，涉及额外的表示压缩误差 | [预印本](https://arxiv.org/abs/2407.12952) · [期刊论文](https://www.sciencedirect.com/science/article/pii/S1746809425018919) · [代码](https://github.com/FahimZaman/LDSeg) |
| LDMSeg | ECCV 2024官方论文集 | 浅层掩码自编码器与预训练潜扩散，原方案约800M参数且使用8张A100 | [论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2304_ECCV_2024_paper.php) · [代码](https://github.com/segments-ai/latent-diffusion-segmentation) |
| SegDT | ICIAP 2025论文，Springer章节于2026-01-02发布 | Rectified Flow与二值掩码Transformer，可用于区分生成路径选择。作者仓库当前仅见README | [论文](https://link.springer.com/chapter/10.1007/978-3-032-10185-3_5) · [代码](https://github.com/Bekhouche/SegDT) |
| DiGSeg | 2026-04-27预印本。作者项目标注ECCV 2026，独立论文集核验待补 | 预训练掩码潜空间、RGB/文本条件、多尺度噪声。原计算规模及集成采样成本较高 | [预印本](https://arxiv.org/abs/2604.24575) · [作者项目](https://wang-haoxiao.github.io/DiGSeg/) · [代码](https://github.com/huaxh6/DiGSeg) |

本轮候选优先沿用项目已有GLP图像网络，以减少骨干变化、预训练数据和算力预算对状态表示实验的干扰。外部方法比较需单独说明任务适配、初始化、数据和预算。

### 2.2 创新边界的直接先行工作

| 工作 | 已有内容 | 对候选的约束与差异 | 原始来源 |
|---|---|---|---|
| Leger等，Scientific Reports 2024 | 光学二维材料分割、颜色与厚度映射、图像预处理 | 光学对比度受材料、基底和成像条件影响，基底归一化已有研究基础 | [论文](https://www.nature.com/articles/s41598-024-79386-z) |
| Takezaki与Uchida，ISBI 2024，ODM | 在不同严重度条件下生成医疗图像，约束预测噪声的有序关系 | ordinal diffusion已有先例。候选研究像素层厚掩码的联合随机状态 | [原文](https://arxiv.org/abs/2403.00452) |
| Cruz等，IEEE Access 2025，预印本2024 | 累计标签编码、单调概率修正及空间有序正则 | §2.2.2已明确像素级累计编码，编码与单调解码本身属于已有思想 | [全文](https://arxiv.org/html/2407.20959v2) |
| Cho等，ICLR 2025，ConDOR | 以有序年龄、疾病因素生成纵向脑指标 | 有序条件与扩散结合已有研究，需限定任务、状态与机制贡献 | [论文](https://openreview.net/pdf?id=9UGfOJBuL8) · [代码](https://github.com/Hannah37/ConDOR-ICLR25) |
| Ao等，BMVC 2024，SAS | 逐遮挡层生成amodal掩码，以累计已分割对象作为条件 | §4.1中累计遮挡掩码保持清洁，仅当前对象掩码加噪。候选对三项层厚阈值联合加噪 | [会议原文](https://bmva-archive.org.uk/bmvc/2024/papers/Paper_15/paper.pdf) |
| Bogensperger等，GCPR 2023 | 图像条件SDF的score-based高斯扩散 | 距离场扩散已有直接先例，单独更换SDF表示的原创空间有限 | [会议原文](https://www.dagm-gcpr.de/fileadmin/dagm-gcpr/pictures/2023_Heidelberg/Paper_MainTrack/044.pdf) |
| Bogensperger等，IJCV 2025，FlowSDF | 图像条件SDF flow matching与不确定性估计 | 几何状态生成已形成方法路线，需明确与项目GLP历史工作的关系 | [原文](https://arxiv.org/abs/2405.18087) |
| Schmidt等，MICCAI 2026 | 有序疾病表示与结构条件下的眼底图像生成 | 近期仍有ordinal diffusion工作，贡献声明应定位到层厚掩码建模 | [官方论文页](https://papers.miccai.org/miccai-2026/0742-Paper3874.html) |

原始文献的检索范围覆盖2024—2026年相关方法，并回溯直接的SDF先行工作。Nature原文直开遇到访问跳转，官方PDF搜索记录核验了题名、作者与研究内容。ConDOR原始PDF搜索和作者代码可读，直接打开论文页遇到浏览器验证。有限检索尚未找到与本方案完全同构的材料分割实现，文献覆盖仍有继续更新的空间。

## 3. 已有仓库能力与候选贡献

`/root/autodl-tmp/ConditionalDiffusionSeg/src/conditional_diffusion/material_conditions.py` 已实现预测背景支持的稳健线性基底场、相对光学对比度、可靠度和直接分割先验。相邻 `material_categorical.py` 的N3已经使用局部光学类别关系调整离散类别转移。

本方案继承可复用的图像建模能力，候选主贡献集中于以下三项可核验设计。

1. 将四类层厚语义表示成三个累计阈值事件，使高斯掩码状态的类别距离反映层厚类别顺序。
2. 使用四类概率头统一产生干净状态估计，保证累计期望的一致性，并接入闭式高斯反向更新。
3. 用类别顺序置换、直接分割和相同条件的名义编码扩散，分别检验有序表示、辅助监督与扩散链的作用。

其中累计编码、概率一致性和高斯扩散均有已有研究基础。潜在贡献来自材料任务中的完整建模、机制解释及受控实证。候选目前具有可实现性与可证伪性，方法创新强度和性能价值仍待检验。

## 4. 数学定义与材料语义

### 4.1 层厚累计状态

令RGB图像为 \(I\)，像素标签 \(y\in\{0,1,2,3\}\) 分别表示背景、单层、少层和多层。定义三个事件

\[
b_k(y)=\mathbf1[y\geq k],\qquad z_0^{(k)}=2b_k(y)-1,\qquad k=1,2,3.
\]

四个码字为

\[
e(0)=(-1,-1,-1),\quad e(1)=(1,-1,-1),\quad e(2)=(1,1,-1),\quad e(3)=(1,1,1).
\]

第一个阈值表示该像素有材料。第二个阈值表示达到少层类别，第三个阈值表示达到多层类别。少层和多层属于区间标签，码字中的相邻级差只表示类别顺序。解释真实原子层数、厚度间隔及类别阈值，需要另外提供物理表征和标签定义。

码字满足

\[
\|e(a)-e(b)\|_2^2=4|a-b|.
\]

这一关系改变了标签之间的去噪距离。它提供可检验的归纳偏置，实际误差是否下降由训练与评价决定。

累计事件在单个像素内嵌套。任意二维位置均可具有任一类别，多层区域可以直接邻接背景。本方案的状态定义允许这些边界，空间邻接层级约束需要另行取得材料依据。RGB三个通道也无需随层厚单调变化。

### 4.2 标准高斯前向过程

取 \(T=1000\)，训练时从全部离散噪声时刻均匀采样。数学采用 \(t=1,\ldots,T\)，代码采用索引0—999。复用已有 `GaussianDiffusion` 的cosine噪声日程。

\[
\alpha_t=1-\beta_t,\qquad\bar\alpha_t=\prod_{s=1}^{t}\alpha_s,
\]
\[
q(z_t\mid z_0)=\mathcal N\!\left(\sqrt{\bar\alpha_t}z_0,(1-\bar\alpha_t)\mathbf I\right),
\qquad z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon,
\quad\epsilon\sim\mathcal N(0,\mathbf I).
\]

RGB条件始终作为干净观测。推理从 \(z_T\sim\mathcal N(0,\mathbf I)\) 开始，cosine日程终端近似纯噪声。有限日程的残余信号由实际保存的 `alpha_bar` 核验。

### 4.3 四类概率头与一致解码

GLP全局—局部网络接收RGB条件、当前噪声状态及噪声时间，输出四类logits。令

\[
p_c=\operatorname{softmax}(\ell)_c,\qquad
\hat z_0^{(k)}=2\sum_{c=k}^{3}p_c-1.
\]

于是 \(\hat z_0^{(1)}\geq\hat z_0^{(2)}\geq\hat z_0^{(3)}\)，且各分量位于[-1,1]。该结果由概率映射保证。最终语义预测统一使用 \(\arg\max_c p_c\)。带噪的中间状态具有高斯扰动，其分量顺序可以变化。

通用码本写法为 \(\hat z_0=\sum_{c=0}^{3}p_c e(c)\)。onehot、ordinal与permuted三种扩散版本共享四类概率头，通过各自码本得到干净状态估计。

### 4.4 DDIM反向更新

由干净状态估计计算隐含噪声

\[
\hat\epsilon_t=\frac{z_t-\sqrt{\bar\alpha_t}\hat z_0}{\sqrt{1-\bar\alpha_t}}.
\]

默认采用20次网络调用的DDIM，\(\eta=0\)。对相邻选定时刻 \(t>s\)，更新为

\[
z_s=\sqrt{\bar\alpha_s}\hat z_0+\sqrt{1-\bar\alpha_s}\hat\epsilon_t.
\]

最后一次调用的类别概率用于分割输出。精确时刻网格、最终时刻处理和实际网络调用数必须保存，以区分训练噪声步数与推理采样步数。固定初始噪声时，\(\eta=0\) 的DDIM路径确定。改变初始高斯噪声仍可能改变结果。

### 4.5 损失与训练约定

当前候选默认损失为

\[
\mathcal L=0.5\mathcal L_{\mathrm{CE}}+0.5\mathcal L_{\mathrm{Dice,fg}}
+\lambda_s\operatorname{mean}_{B,C,H,W}\!\left[(\hat z_0-z_0)^2\right]+0.5\mathcal L_{\mathrm{prior}},
\qquad\lambda_s=1.
\]

CE与Dice保留四类语义监督，stateMSE监督相应码本的干净状态。全局先验损失包含各占0.5的软标签CE与前景Dice，沿用已有GLP辅助监督。MSE对状态通道取均值，通道数仍需在结果中披露。此目标属于干净状态预测参数化。与噪声MSE的相对时间权重不同，目标名称和实现应保持一致。

direct版本也使用ordinal映射产生stateMSE辅助目标，通过RGB直接预测四类概率。该设置使direct与ordinal比较能部分控制有序辅助监督收益。四组共享的数据、优化器、初始化政策、损失权重及选模规则应一并冻结。

资源实测后固定150轮、batch12。train120每图每轮2个裁块，每轮240次裁块曝光、20次更新，全程36000次曝光、3000次更新。四组新实验保持相同预算。历史GLP的batch8、100轮为3000次更新及24000次曝光，跨历史版本比较分别报告两种预算。

## 5. 四组主对照与归因限制

| 配置 | 状态与监督 | 推理 | 主要回答的问题 |
|---|---|---|---|
| onehot | 四通道有符号onehot码本，四类CE/Dice与对应stateMSE | 纯高斯起点，20步DDIM | 常规名义类别掩码扩散参照 |
| ordinal | 三通道累计阈值码本，四类CE/Dice与对应stateMSE | 纯高斯起点，20步DDIM | 层厚顺序是否帮助掩码去噪 |
| permuted | 同一三通道累计码本，预先固定打乱类别至码字的映射 | 纯高斯起点，20步DDIM | 物理类别顺序是否具有作用 |
| direct | RGB直接四类预测，附加ordinal stateMSE | 一次预测 | 有序辅助监督与多步扩散的相对贡献 |

permuted映射在读取新结果前固定，并保存到配置。全序反转保留所有成对层级距离，属于不合适的破坏性对照。当前permuted配置固定顺序为[0,2,1,3]，交换单层与少层的码字位置，保留材料有无阈值。各组的最终输出均映射回原始四类语义。

onehot实际为4个状态通道，ordinal/permuted实际为3个状态通道。网络的状态输入投影参数量随之变化，需报告参数总数及差值。direct的数据路径与前向调用数也不同，需实测单窗和整图成本。

幅度均为±1时，onehot每像素状态范数平方为4，ordinal为3，噪声总能量也随维数变化。两者按总原始能量计算的期望信噪比均为 \(\bar\alpha_t/(1-\bar\alpha_t)\)。类别间距离、按训练分布中心化后的状态方差、通道相关性和类别判别的有效信噪比存在差异，归因应逐项报告。

特别地，onehot不同类码字的平方距离恒为8，ordinal为 \(4|a-b|\)。三通道permuted保持码字集合与维数，但类别频率重新分配到码字会改变各通道统计量。因此该对照提供顺序机制证据，其解释仍需结合类别频率与状态统计。首轮方案保持固定幅度与统一日程，阈值特异权重应作为后续独立假设登记。

## 6. 预登记评价与成功判据提案

### 6.1 统一协议

- 保留现有train120/val25划分与图像ID，固定训练seed42和推理采样seed。使用训练集生成的统计量与数据版本哈希。
- 四组匹配裁块规则、增强、优化更新、初始化来源及训练曝光。模型权重按验证四类mIoU选择，同时保留固定最终轮结果。
- 原尺寸推理采用相同窗口、步幅与重叠融合规则，记录实际网络调用数、显存、参数量、单窗和整图时延。
- 保存每图混淆矩阵与边界计数。主指标同时包含四类mIoU、单层IoU、少层IoU、薄层平均IoU和前景BF@2px。
- 同时报告单层↔少层双向混淆、跨两级及以上的错误比例、预测类别频率和不同噪声时刻的干净状态误差。阈值码字一致性由结构保证时，应作为实现属性报告。
- 边界评价辅以1/4像素容差及Boundary IoU，检查区域收益是否以边界退化为代价。
- 图像级配对重采样用于描述固定权重的结果不确定性。单训练seed的结论范围与样品独立性问题单独披露。

现有test27曾参与历史开发。若后续执行测试，需先冻结所有待比较权重及配方，并将结果明确标为既有测试划分上的固定模型比较。新采集样品或新批次可提供额外泛化证据。

### 6.2 提议的实际阈值

以下阈值属于尚待训练前锁定的提案，单位pp表示百分点。它们用于决定研究是否值得继续，与统计显著性属于不同概念。

| 判断 | 提议门槛 | 解释 |
|---|---|---|
| 材料任务收益 | ordinal相对onehot的薄层平均IoU提高至少0.50pp，同时四类mIoU下降不超过0.30pp、BF@2下降不超过0.50pp | 优先检验薄层识别，保留总体区域和边界约束 |
| 类别顺序作用 | ordinal相对固定permuted的薄层平均IoU提高至少0.30pp，单层↔少层混淆方向与改善一致 | 检验正确顺序是否提供额外信息 |
| 扩散实用价值 | ordinal相对direct至少满足上述薄层门槛，或BF@2提高至少1.00pp且mIoU与薄层均未下降超过0.30pp | 同时报告达到该收益的时延与调用成本 |
| 跨级错误机制 | 相对onehot的跨两级及以上错误率下降至少10%，同时报告绝对变化和原始计数 | 基线错误接近零时以计数及绝对差解释，避免放大相对比例 |

主要结论按固定best规则与全部验证图像给出。固定最终轮结果若改变方向，应在结论中保留该敏感性。门槛通过也只支持本数据、预算和种子范围内的实用收益。

### 6.3 失败与收缩判据

若ordinal与permuted接近，正确类别顺序的解释受到削弱。若direct获得同等收益，证据更支持有序辅助监督。若薄层收益伴随超过门槛的总体区域或边界代价，应按取舍报告。若收益只出现在best而最终轮反转，应限制训练稳定性的表述。

若打乱状态、改变初始噪声或缩短反向链几乎不影响输出，应进一步量化模型对状态的依赖。可在固定权重诊断后决定是否需要新的机制训练，临时输入干预的因果解释保持有限范围。

一次未通过门槛的实验仍可用于更新问题判断。后续变更需单独登记目标、预算与成功条件，并保留失败结果，避免根据既有test排名反向筛选方案。

## 7. 当前可用的贡献表述与待验证事项

可用于研究方案的表述为

> 本研究探索将MoS₂层厚类别的有序关系编码为联合累计阈值掩码，在RGB条件高斯扩散中建立与类别级差相关的状态表示，并通过类别顺序干预、统一条件对照及薄层错误分解检验其作用。

已有研究部分包括累计编码、概率一致性、高斯扩散、条件融合、光学归一化与几何状态生成。候选原创部分为本任务中的联合状态建模、统一概率端点及机制实验组合。待验证事项包括薄层与边界收益、状态利用程度、成本合理性和跨样品泛化。

工程检查覆盖码本映射、四类语义保持、时间索引、DDIM端点、纯高斯初态、有限梯度、输入尺寸和正常入口。工程验证与训练效果记录分别报告。

## 8. 实现入口与工程记录

当前分支为 `feat/diffusion-seg`。实现包含独立的状态编码与采样器、全局—局部条件去噪网络、数据与训练循环、恢复入口和四份配置。

- `/root/autodl-tmp/ConditionalDiffusionSeg/src/conditional_diffusion/mask_diffusion.py` 实现onehot/ordinal码本、cosine前向过程、状态重建目标与DDIM。
- `/root/autodl-tmp/ConditionalDiffusionSeg/src/conditional_diffusion/diffusion_seg_model.py` 从既有GLP提取纯网络模块，去掉固定步数level和自条件输入，使用真实扩散时间。
- `/root/autodl-tmp/ConditionalDiffusionSeg/src/conditional_diffusion/diffusion_seg_experiment.py` 管理整图条件、裁块、共享噪声滑窗、评价及可恢复训练。
- `/root/autodl-tmp/ConditionalDiffusionSeg/scripts/diffusion_seg.py` 提供train、evaluate、smoke三个显式入口。

ordinal、permuted和direct均为18,191,860个参数，onehot为18,192,148个参数，相差288。默认局部宽度为32/64/128/192，全局宽度为32/64/128/256/256，token维度256。各组保留相同图像主干。direct以零状态和零时间直接输出语义概率，并使用ordinal重建辅助目标。

新数据路径将整图边缘复制补齐到64倍数后构造缩略图、全局软标签和裁块坐标，评价裁回原始尺寸。训练使用整图gamma/gain、旋转翻转以及薄层偏置裁块。该预处理与历史GLP裁掉末端不足64倍数部分的流程有差异，历史分数仅供背景参考。四组新实验使用同一新数据路径。

当前训练保存原始模型权重，采用FP32、关闭TF32及严格确定性算子设置。新增入口采用epoch边界恢复，保存模型、AdamW、调度器、随机状态、源文件和划分哈希。last检查点携带所选best权重及历史记录，使中途导出失败可在恢复时修复。正式配置已固定150轮、batch12，具体队列与启动规则见训练计划。

### 8.1 可执行命令

以下命令在AutoDL项目内执行。每条train命令只启动显式指定的一组。

```bash
cd /root/autodl-tmp/ConditionalDiffusionSeg
/root/miniconda3/envs/diffusion/bin/python -B scripts/diffusion_seg.py smoke --config configs/diffusion_seg_ordinal.json

/root/miniconda3/envs/diffusion/bin/python -B scripts/diffusion_seg.py train \
  --config configs/diffusion_seg_ordinal.json \
  --run outputs/diffusion_seg/ordinal_seed42

/root/miniconda3/envs/diffusion/bin/python -B scripts/diffusion_seg.py train \
  --config configs/diffusion_seg_ordinal.json \
  --run outputs/diffusion_seg/ordinal_seed42 --resume

/root/miniconda3/envs/diffusion/bin/python -B scripts/diffusion_seg.py evaluate \
  --config configs/diffusion_seg_ordinal.json \
  --run outputs/diffusion_seg/ordinal_seed42
```

另外三份配置的完整路径为 `/root/autodl-tmp/ConditionalDiffusionSeg/configs/diffusion_seg_onehot.json`、`/root/autodl-tmp/ConditionalDiffusionSeg/configs/diffusion_seg_permuted.json` 和 `/root/autodl-tmp/ConditionalDiffusionSeg/configs/diffusion_seg_direct.json`。各组使用独立run目录。

### 8.2 已完成的工程检查

AutoDL环境为PyTorch2.5.1+cu124、RTX3090。全套88项单元测试通过，含20项新增测试。四份配置均通过真实train图m1的CLI smoke。512×512、batch2的一次更新及DDIM2检查通过，峰值预留3764MiB。原尺寸1922×2560输出及20步DDIM、35窗、700次窗口调用通过。正式batch12经2次预热和16次计时更新通过，峰值分配21448.54MiB、预留23582MiB。工程检查只验证执行和数值，分割收益通过完整训练评价。
