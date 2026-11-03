# 第四章（扩散分割）图表规划与轻量化对比训练计划（2026-10-24）

状态：**计划，待执行。** 本文件不启动任何训练、推理或绘图。

## 0. 用户已确认的决定（2026-10-24）

1. 论文目录调整为：第一章 绪论；第二章 理论基础；第三章 数据集清洗与制备；**第四章 扩散/条件流匹配分割（GLP-Flow E3R，即原"第3章"）**；第五章 小论文分割模型 RepELA-Net + 部署（原第4、5章合并）；第六章 总结。
2. 授权对 val25 做一次**仅前向**的过程导出（E3R best 权重），用于过程图 P1/P8。
3. 授权绘图时**只读** val25 的原始 RGB/GT（用于标正误、边界带等）。
4. 本文件的图表只针对第四章；训练计划包含与轻量化模型 RepELA-Net 的对比。

第二轮确认（2026-10-24）：

5. 正文图不放附录：原附图A-1/A-2 合并为正文图4-16（两子图），A-3 进正文为图4-17，A-4 并入图4-9 作为子图。两张附表保留在附录。
6. RepELA 采用 **Small** 变体，未用预训练；代码从 PhysicalNet **只读复制**到本项目，源项目不删除、不修改。
7. RepELA 统一协议训练一组，损失用 **0.5 CE + 0.5 前景 Dice**（与 9 个对比方法一致）；不另跑小论文原配方。
8. 定性对比图允许只读 test27 原始 RGB/GT。
9. 阶段1–4 由一个协调器自动串行执行，任一校验失败即停止并保留日志。测速只在本机 RTX 3090 上做。

小论文原配方（`PhysicalNet/tools/train_oldcfg.py`，仅作背景记录）：类别加权 Focal（α=[0.15,3.6,4.56,0.57]，γ=2）+ Dice，200 轮×每轮 15 次更新=3000 次更新，AdamW 6e-4、wd 0.01、warmup 10，无 EMA、无梯度裁剪，512/384 滑窗验证。PhysicalNet 的 `output/` 目录在本机不存在（被 `.gitignore` 排除，10-17 起即不在），小论文原始权重与日志无法在本机复核。

沿用的约定：训练 seed 42、无交叉验证；val25 保留 m109/m31；不纳入 DINO 路线；正式稿不放失败/局限性案例图，不利数值在表格和正文如实报告；test27 已暴露，不称盲测，不按 test 重选权重。

本文件取代 `THESIS_FIGURE_TABLE_PLAN_20261024.md` 中"第3章"部分的编号（原图3-x/表3-x 改为图4-x/表4-x）。

## 1. 第四章叙事主线

全局先验给出粗略答案 → 标准流匹配训练存在训练/推理状态不一致，模型不会沿自身轨迹修正（E3）→ 采样对齐训练让模型学会沿自己的轨迹修正（E3R）→ 每一步只改少量边界像素，边界不确定性逐步收敛 → 边界质量显著提升，区域精度与最强基线相当 → 代价是多步推理，引出第五章轻量化。

章节结构：4.1 引言 → 4.2 任务定义与总体框架 → 4.3 全局—局部条件流匹配网络 → 4.4 采样对齐训练与推理 → 4.5 实验（设置、对比、消融、过程分析、边界分析、步数与成本、定性）→ 4.6 小结。

## 2. 第四章图（全部放正文，共 17 张）

> 2026-10-24 修订：下表"附图"行按第0节第5条并入正文——A-1 与 A-2 合为图4-16（两子图），A-3 改为图4-17，A-4 并入图4-9（新增子图）。图4-15 为定性对比。

状态：**现成**＝已有图或数值；**轨迹**＝用已存 val 轨迹（`glpflow_e3r_mechanism_20261022/trajectories/`）；**GT**＝需只读 val 原始 RGB/GT；**导出**＝需阶段 1 的过程导出；**测速**＝需阶段 4 的统一测速。

| 编号 | 图 | 讲什么 | 数据来源 | 状态 |
|---|---|---|---|---|
| 图4-1 | 总体框架（四分区，中文版） | 方法全貌 | `docs/figures/glpflow_e3r_cvpr_20261023` | 现成，需中文化 |
| 图4-2 | 端点预测网络细节 | 三流联合注意力、MP 跳跃融合 | `docs/figures/glpflow_e3r_20261022` | 现成 |
| 图4-3 | 训练状态 vs 推理状态示意：标准流匹配用 GT 插值状态，采样对齐用模型自身无梯度轨迹 | 为什么要采样对齐 | 示意图 | 新绘 |
| 图4-4 | **推理过程全景**（每张图一行）：RGB → 全局先验 P → 熵 U → 初态 x₀ → 第1–4步端点 → 最终 → GT | 从"先验+噪声"到分割结果的完整流动 | P/U/端点来自轨迹；x₀、x_t 来自导出 | 导出 + GT |
| 图4-5 | 状态差异诊断：横轴 t，纵轴"GT 插值状态 − 自生成状态"的端点 mIoU 差，E3 vs E3R | t=0.25 时 12.27pp → 0.24pp | `diagnosis.json`、`report.json` | 现成（图注标"oracle 分析"） |
| 图4-6 | 四步链逐步指标：mIoU、前景边界 F1、薄层均值三个子图，E3 vs E3R | E3 平坦；E3R 边界 F1 72.1→73.1→73.4→74.1 | `report.json` | 现成 |
| 图4-7 | 逐步变化像素图 + 边界局部放大：第 k→k+1 步改变的像素，按"改对/改错"着色；右侧子图为变化像素到 GT 边界的距离直方图 | 后续步只改 ‰ 级像素，且集中在边界附近 | 轨迹 + GT | GT |
| 图4-8 | 逐步距离带准确率增量：第 k 步减第 1 步，按 0–2/2–4/4–8/8–16/>16 px | 细化集中在 0–4 px | `report.json`（逐步 distance_accuracy） | 现成 |
| 图4-9 | 置信度演化：每步置信度图，加"边界带 vs 内部"置信度分布 | 边界不确定性逐步下降 | 轨迹 + GT | GT |
| 图4-10 | 像素级概率轨迹：若干边界像素与内部像素的四类概率随步数变化 | 直观表现"流"：边界像素逐步移动，内部像素一步稳定 | 导出 | 导出 + GT |
| 图4-11 | 逐步类别转移（桑基图或转移矩阵）：第1步→第4步各类像素去向，按正误拆分 | 迭代修正了哪些混淆 | 轨迹 + GT | GT |
| 图4-12 | 分类别边界 F1（背景/单层/少层/多层，test27），E3R 与主要对比方法 | 少层边界 F1 68.8 vs SegFormer 58.0 | 各 `*_confusion.npz` 的 `boundary_counts` | 现成数据 |
| 图4-13 | 边界距离分层准确率：E3R、E0、E1、SegFormer、Mask2Former、DDP，val 与 test 各一子图 | 差异集中在边界 4 px 内 | `strata_correct/strata_total` | 现成数据 |
| 图4-14 | 质量—成本：左为采样步数 K=1/2/4/8 的质量与耗时；右为各方法边界 F1 vs 纯网络延迟（点大小=参数量） | 四步是训练匹配的工作点；多步的代价 | `report.json` + 阶段 4 测速 | 测速 |
| 附图A-1 | 训练曲线：E1/E3/E3R 的 val mIoU 与边界 F1 随轮次 | 最佳在第 50–60 轮，后期无需加轮 | 各 run `metrics_val.csv` | 现成 |
| 附图A-2 | 全部对比方法的 val mIoU–轮次曲线（最终版） | 支撑"统一有限预算"的说明 | 各 run `metrics_val.csv` | 现成数据（需重画） |
| 附图A-3 | 逐图配对差值分布：E3R 减 SegFormer/E0/E1/Mask2Former/DDP，每图一点，标中位数和胜场 | 整图胜场多、全局差距集中在少数图 | 各 `per_image` 混淆矩阵 | 现成数据 |
| 附图A-4 | 熵与误差：按熵分箱的误差率 | 熵作为先验不确定性的意义（不称已校准） | 轨迹 + GT | GT |

另有**定性对比图**（图4-15，正文）：RGB / GT / SegFormer / Mask2Former / DDP / E0 / E3R，加边界局部放大；使用 test27 已保存掩码（`outputs/inference/*_test_20261022/`），只读 test 原始 RGB/GT（已获确认）。

### 2.1 选图规则（预先登记，绘图前固定）

- 过程图（图4-4、4-7、4-9、4-10）和定性图的候选：单层、少层像素各 ≥1.5% 的图。val25 中为 m108、m109、m127、m132、m22、m23、m86、m92。
- 在候选中按 E3R 逐图 mIoU 排序，取最接近候选中位数的 2 张进正文，其余进附录。不按"E3R 效果最好"挑，也不选极端值。
- 规则、候选名单和最终选中的图写入绘图目录的 `sources.json`。
- E3 只出现在曲线图（图4-5、4-6）中，不做 E3 的误差图像面板。

## 3. 第四章表（共 9 项：正文 7，附录 2）

| 编号 | 表 | 数据来源 | 状态 |
|---|---|---|---|
| 表4-1 | 网络配置与参数分组（局部编码器、全图分支、状态编码器、联合注意力、解码器等） | `docs/figures/glpflow_e3r_20261022/parameter_groups.csv` | 现成 |
| 表4-2 | 统一实验协议（100 轮/3000 次更新、batch 8、AdamW、EMA、512/384 滑窗、选模规则） | `plan.json` | 现成 |
| 表4-3 | **test27 主对比**：12 个方法 × mIoU、四类 IoU、薄层均值、前景边界 F1、参数量、NFE | `comparison_summary.json` + GLP test | 现成 |
| 表4-4 | val25 对比（选模口径）；表注给出去掉 m109 后的描述值 | `best_val_confusion.npz` | 现成数据 |
| 表4-5 | **消融**：E0/E1/E2/E3/E3R，val+test，四类 IoU、边界 F1、最佳轮 | 两批 `comparison_summary.json` | 现成 |
| 表4-6 | 四步链逐步端点数值（mIoU、单层/少层 IoU、少层→单层率、边界 F1、相邻步变化像素比例） | `report.json` | 现成 |
| 表4-7 | **推理成本**：参数量、每窗/每图 MACs、NFE、纯网络延迟（中位/P95）、整图预测时间（网络/融合分列）、推理峰值显存 | 阶段 4 测速 | 测速 |
| 附表A-1 | 对比方法实现说明及与原文的差异 | `sources.json` | 现成 |
| 附表A-2 | 扩散方法 3407/3408 复核；逐图 test mIoU 全表 | `final_analysis.md`、`per_image_miou.json` | 现成 |

算法4-1：采样对齐训练伪代码（含推理流程），依据 `work/glpflow_rollout_20261022.py`。

## 4. 执行计划

总体顺序：阶段1 → 阶段2/3 → 阶段4 → 阶段5。阶段1–4 均已获授权；**代码已在本对话写完并通过冒烟检查（第7.0节），正式运行由用户在另一个对话中启动**，执行规格见第7节；由协调器 `work/launch_ch4_extension_20261024.py` 串行执行；批次目录 `outputs/launches/p1_ch4_extension_20261024/`。

实现文件（只新增，不改已有文件）：

- `work/repela_20261024/physicalnet_models/`：从 PhysicalNet 逐字节复制的 `repela_net.py`、`rep_conv.py`、`ela_block.py`、`decoder.py`（源与副本 SHA256 记入 `sources.json`），另加空的 `__init__.py`。
- `work/repela_20261024/model.py`：RepELA-Small 包装（统一损失、确定性算子替换、重参数化）；`core.py`：在对比方法框架中注册 `repela_small`；`verify.py`：工程验证。
- `work/export_e3r_process_20261024.py`：阶段1；`work/benchmark_ch4_20261024.py`：阶段4。

### 阶段1：val25 过程导出（已授权，仅前向）

- 模型：E3R val-best EMA 权重（ep60），采样种子 3407，K=4，与选模设置完全相同。
- 只新增脚本 `work/export_e3r_process_20261024.py`，复用 `analyze_glpflow_e3r_20261022.py` 的冻结函数，不修改已有文件。
- 每张 val 图额外保存：初态 x₀、每步状态 x_t（四通道）、每步四类概率 q_k，均为 1/4 分辨率 float16；输出 `outputs/analysis/e3r_process_export_20261024/`。估计约 0.3–0.5 GiB，现有磁盘余量 23 GiB。
- 校验：最终端点与已存轨迹逐像素一致；val25 mIoU 复现 88.56%。不一致即停止并报告。
- 不跑 test，不导出 E3（E3 只用现有曲线数据）。预计 10–15 分钟。

### 阶段2：RepELA-Net 移植与工程验证（已授权）

- 从 PhysicalNet 只读复制 RepELA 模型代码（`models/repela_net.py`、`rep_conv.py`、`ela_block.py`、`decoder.py`）到 `work/repela_20261024/`，记录源文件 SHA256；不改动 PhysicalNet。
- 核对：参数量与小论文所用变体一致；RepConv 重参数化前后输出等价（最大绝对误差 <1e-4，val 窗口 argmax 一致率记录）。
- 工程验证同对比方法：P0 固定裁块过拟合、断点恢复逐位一致、batch 8 吞吐与峰值显存。

### 阶段3：RepELA-Net 统一协议训练（已授权）

| 项目 | 设置 |
|---|---|
| 组别 | RL-U：统一协议，**一组**，seed 42 |
| 数据与增强 | 与 GLP-Flow/对比方法完全相同（train120/val25/test27，512 裁块，50% 薄层中心） |
| 预算 | 100 轮，batch 8，3000 次更新，从头训练（不用预训练） |
| 优化器 | AdamW 2.8e-4，wd 1e-4，warmup 5 轮，余弦到 1e-6，梯度裁剪 1.0，power-EMA σ_rel=0.10 |
| 损失 | 0.5 CE + 0.5 前景 Dice；若小论文规定了辅助/边界损失，按"原论文辅助损失照用"规则保留并在 `sources.json` 注明 |
| 选模与测试 | 每 10 轮 val25（512/384 滑窗）选 best；全部完成后 test27 评估一次；训练形态与部署形态（重参数化后）各评估一次 |
| 门槛 | 不设门槛，作为参照记录；不因结果追加训练、改超参或换结构 |
| 预计耗时 | 训练+验证约 2–2.5 小时，test 约 10 分钟 |

RL-U 结果用于第五章跨章对比；第四章不放 RepELA 行。

### 阶段4：统一推理成本测速（已授权，不训练）

对象：9 个对比方法、E0/E1/E3R、RepELA（训练形态与部署形态），各用 val-best EMA 权重。

- 环境：RTX 3090，FP32，TF32 关闭（与训练/评估一致），每个方法单独进程、清空缓存后测。
- **窗口级纯网络延迟**：输入 1×3×512×512 已在 GPU，预热 50 次、计时 200 次，CUDA 同步，报中位数与 P95。扩散方法包含全部采样步；E1/E3R 的全图分支按每图一次单独计时，再折算到每图。
- **整图预测时间**：2560×1922，沿用各方法现有的 35 窗口滑窗与融合，网络时间与 CPU 融合时间分列，不含读图与指标计算；用 val25 计时（只计时，不选模）。
- **MACs**：自写 hook 计数器（Conv/ConvTranspose/Linear/matmul/bmm/注意力），环境中无 thop/fvcore，不新装依赖；先用小网络与解析计算值对拍验证。可变形注意力的采样操作单独说明口径。
- **推理峰值显存**：整图预测过程的 max_memory_allocated 与 reserved。
- 预计约 1.5–2 小时。

### 阶段5：数据汇总与绘图（CPU）

- 只读已保存输出和阶段1/4结果，以及 val25 原始 RGB/GT；脚本放 `docs/figures/chapter4_20261024/`，每张图配 `sources.json`。
- 版式沿用既定规范：完整闭合边框、NotoSerifSC + Tinos、统一类别配色。

## 5. 跨章对比的报告口径（预先登记）

- 精度：mIoU、单层/少层 IoU、前景边界 F1，用百分点差值逐项列出，不用百分比相除。
- 资源：参数量、MACs、纯网络延迟、整图预测时间、推理峰值显存，给出比值。
- 不预设加速倍数或精度损失上限；RepELA 精度更高或更低都如实写。
- 小论文中的历史结果（不同预算/协议）只作背景，不与 RL-U 或第四章数值混排。

## 6. 用户确认记录

第 0 节第 5–9 条已回答原先的全部问题（2026-10-24）。阶段5（绘图）在阶段1–4 完成并汇报后进行。

## 7. 执行规格（供另一个对话执行）

### 7.0 代码状态（2026-10-24，本对话完成）

所有代码已写完；正式运行（工程验证、导出、训练、测速）尚未开始，项目中没有任何本批次的正式输出目录。

| 文件 | 行数 | SHA256 前16位 | 作用 |
|---|---:|---|---|
| `work/repela_20261024/physicalnet_models/{repela_net,rep_conv,ela_block,decoder}.py` | — | 见 7.2 节表 | PhysicalNet 逐字节副本 |
| `work/repela_20261024/physicalnet_models/__init__.py` | 1 | 8c98d0c2d2f1dcde | 包初始化 |
| `work/repela_20261024/__init__.py` | 0 | e3b0c44298fc1c14 | 包初始化 |
| `work/repela_20261024/model.py` | 93 | e98a03fdd6c35d4a | RepELA-Small 包装 |
| `work/repela_20261024/core.py` | 45 | 4fc7b45801e9de52 | 注册到对比方法框架、路径、配置 |
| `work/repela_20261024/verify.py` | 222 | 97d59cc005b5931c | 阶段2 工程验证（7 项） |
| `work/export_e3r_process_20261024.py` | 151 | 7f6968a0805f35d3 | 阶段1 过程导出 |
| `work/benchmark_ch4_20261024.py` | 306 | c04924ee661b596a | 阶段4 统一测速 |
| `work/launch_ch4_extension_20261024.py` | 307 | 90621ef78c9f069c | 协调器（阶段1–4 + 汇总） |

协调器启动时会把以上文件和依赖的旧文件的哈希冻结进 `plan.json`；执行对话若需修改代码，必须在启动协调器之前完成，并在报告中说明改了什么。

本对话已做的冒烟检查（均不写正式输出）：

| 检查 | 结果 |
|---|---|
| 阶段1：在 val 图 m108 上运行 `chain_states` | 4 步 argmax 与已存轨迹逐像素一致；终步混淆矩阵与 best_val 一致；终态 x₄ 取值 [-1, 1] |
| 阶段2：verify 第 1/2/3/7 项（直接调用函数） | 全部通过：参数 2,124,322（部署形态 2,122,658）；确定性替换误差 ≤1.1e-8；重参数化误差 1.7e-6、argmax 一致率 100% |
| 阶段2：verify 第 4/5/6 项 | **未运行**（留给执行对话；第5项会在 `outputs/integration/` 建临时目录并删除） |
| 阶段4：`CH4_BENCH_SMOKE=1 CH4_BENCH_OUT=/tmp/...` 跑 hrnet_w48、ddp、E0、E3R_K1 | 全部完成；原生算子与原评估代码像素一致率 100%；临时输出已删除 |
| 阶段4：RepELA 两种形态 | 用随机权重核对过部署转换与 FLOPs 统计；正式测速需等阶段3 的 best.pt |
| 协调器 | 可导入；PhysicalNet 4 个源文件哈希一致；12 个已有 best.pt 均存在 |

需要用户留意：

- **RepELA 计算量**：统一计数器（`FlopCounterMode`）得到 RepELA-Small 单个 512×512 窗口约 **5.25 GMACs**（卷积 5.17 + 矩阵乘 0.09），而源代码注释写的是设计目标"<2G FLOPs"。小论文若报告了 FLOPs/MACs，统计工具与口径可能不同（例如 thop 不计 einsum），请核对；论文中跨方法比较应统一用本次计数。
- 冒烟测速显示整图 `predict` 中 CPU 融合与数据搬运约占 1.1–2.1 s/图（如 HRNet 网络 0.50 s、整图 1.61 s），因此表4-7 必须分列"网络时间"和"整图时间"。


### 7.0.1 通用约束

- 只新增文件；不修改 `src/`、`scripts/`、`configs/`、`work/glpflow_*.py`、`work/baselines_20261022/`、`work/launch_baselines_20261022.py`、历史 `outputs/` 与 `docs/` 中的其他文件。
- PhysicalNet 只读：不删除、不移动、不写入。开始和结束时各校验一次 `PhysicalNet/models/{repela_net,rep_conv,ela_block,decoder}.py` 的 SHA256（见 7.2）。
- 训练 seed 42；FP32、TF32 关闭、确定性算法开启（训练/评估）；单次训练，不做多种子、交叉验证或追加训练。
- val25 选模；test27 只评估一次，不按 test 重选；不纳入 DINO。
- 使用 `outputs/.experiments.lock` 独占锁；GPU 空闲时才启动（启动前 `nvidia-smi` 确认无其他进程）。
- 磁盘：开工前空闲 ≥15 GiB（当前约 23 GiB；项目约 20 GiB，StorageGuard 项目上限 26 GiB）。
- 任一校验失败：写 `status.json`（stage=failed_*）、保留日志、停止，不自动修复重跑。

### 7.1 阶段1：E3R 过程导出

脚本：`work/export_e3r_process_20261024.py`（已写好，见7.0）。

1. 加载 `outputs/P1/glpflow/seed_42/p1_glpflow_rollout_20261022_e3r/checkpoints/best.pt`（ep60，EMA），`G.configure(42)`，模型 `G.GLPFlowNet(c)`。
2. 对 val25 每张图，按 `analyze_glpflow_e3r_20261022.chain_probs` 的同一算术跑 K=4 链（同一噪声场 `sha256(f'{id}:3407')`、同一窗口坐标、同一自条件与 Euler 更新），额外记录：
   - 初态 x₀ 与每步输入状态 x_k（k=0..3）以及终态 x₄，四通道；
   - 每步端点概率 q_k（k=0..3），四通道；
   - 均按重叠窗口平均融合到整图（与概率融合方式相同；状态的融合仅用于可视化，在 README 注明）。
3. 保存：每图一个 `outputs/analysis/e3r_process_export_20261024/<id>.npz`，键 `states`（5,4,481,640）、`probs`（4,4,481,640），float16，按 `[::4, ::4]` 取样（与已存 `confidence_div4` 同一网格）；另存 `report.json`（脚本/权重哈希、耗时、校验结果）和 `README.md`。
4. 校验（全部通过才算完成）：
   - 每步全分辨率 argmax 与 `outputs/analysis/glpflow_e3r_mechanism_20261022/trajectories/E3R/<id>.npz` 的 `endpoints` 逐像素一致；
   - 第4步逐图混淆矩阵与 E3R `best_val_confusion.npz` 的 `per_image` 完全一致（val mIoU 88.56%）。
5. 不跑 test，不导出 E3。预计 10–15 分钟，约 0.3–0.6 GiB。

### 7.2 阶段2：RepELA 移植与工程验证

代码文件见 7.0 节表格（RepELA 部分：`work/repela_20261024/`）。要点：

- `model.py`：`RepELASmall` 包装 RepELA-Net-Small（无 CSE、无深监督），统一损失 `standard_sem_loss`；把复制模块内的双线性插值/自适应池化路由到 `baselines_20261022.models` 的确定性实现，`AdaptiveAvgPool2d(1)`（共 5 处）换成空间均值；`native_ops()` 关闭路由；`switch_to_deploy()` 融合 RepConv 分支。
- `core.py`：在 `baselines_20261022.core` 的 `METHODS` 中登记 `repela_small`，并在运行时包装 `build_model`（不改原文件），因此训练/评估与 9 个对比方法走同一代码路径。
- `verify.py`：下列 7 项工程验证，输出 `outputs/launches/p1_ch4_extension_20261024/engineering_verification_repela.json`。

源文件 SHA256（复制时记录，源与副本相同）：

| 文件 | SHA256 |
|---|---|
| repela_net.py | 9f864363825775c1537dd4a5f01620125f1d14fc10cecdfbe3563d6c79b96f3d |
| rep_conv.py | dd6b6739c82bb6c2b0cda6dff92b5b65957d7610f1f3f1eacd7c6b270c217d34 |
| ela_block.py | 90bb164dd2d9895722ae86afe99ae58a7b767bc6ff5c1a2b66795d31bb7d7870 |
| decoder.py | e17ad4c639222514e39325751222c57d18940ba8f15a693ae1492a99c77c3a30 |


工程验证项（全部通过才进入阶段3）：

1. 副本与源逐字节一致；参数量在 2.0–2.3M；1×3×512×512 → 1×4×512×512 且有限。
2. 确定性替换 vs 原生算子：随机输入与真实 val 窗口最大绝对误差 <1e-4。
3. 重参数化：真实数据训练 20 步后，eval 模式训练形态与部署形态输出最大误差 <1e-4×max(1,|logit|max)，argmax 一致率 >99.99%；记录两种形态参数量。
4. batch 8 真实训练 10 步（3 预热 + 7 计时）：样本/秒、峰值显存（须 <22.5 GiB）、损失有限。
5. 断点恢复：连续 2 轮 vs 1+1 轮（每轮 3 个 batch），模型/优化器/调度器/EMA/RNG 逐位一致；临时目录用后删除。
6. P0 固定 4 裁块记忆门槛：300 次更新、lr 2e-4，训练 mIoU ≥0.95。
7. `predict` 签名不含 GT。

### 7.3 阶段3：RepELA 统一协议训练与评估

- 配置：`repela_20261024.core.config()`，即 `baselines_20261022.core.config('repela_small')` 加 RepELA 字段；与 9 个对比方法同协议（第4节阶段3表）。
- 运行目录：`outputs/P1/repela/seed_42/p1_repela_20261024_repela_small/`（结构同对比方法：checkpoints/best.pt、last.pt、metrics_*.csv、validation_epoch*.json、best_val_metrics.json、best_val_confusion.npz、tensorboard/）。
- 训练：`baselines_20261022.core.train(c, RUN, plan_sha, guard_fn=...)`，子进程执行，日志 `train_repela_small.log`。
- 评估（训练完成后）：
  1. 训练形态 val-best EMA 权重，test27 评估一次：保存 `outputs/inference/repela_test_20261024/train_form/{masks,color}/`、`train_form_confusion.npz`，并按对比方法的做法逐图用保存掩码重算混淆矩阵核对。
  2. 部署形态（同一权重 `switch_to_deploy()`）：val25 与 test27 各评估一次，保存 `deploy_form_{val,test}_confusion.npz`；报告与训练形态的逐图混淆矩阵差异和像素一致率。论文主表用训练形态数值，部署形态作等价性证明。
  3. 汇总 `test_metrics.json`、`per_image_miou.json`。
- 预计：训练+验证约 2–2.5 小时，评估约 20 分钟。

### 7.4 阶段4：统一推理成本测速

脚本：`work/benchmark_ch4_20261024.py`（已写好，见7.0），结果 `outputs/analysis/ch4_benchmark_20261024/{benchmark.json,benchmark.csv,README.md}`。

对象（各用 val-best EMA 权重，每个对象一个独立子进程）：

| 组 | 对象 |
|---|---|
| 对比方法 | unet、unetpp、deeplabv3p、pspnet、segformer_b1、hrnet_w48、mask2former、ddp（3 步）、ccdm（10 步） |
| GLP-Flow | E0、E1、E3R（K=4）；E3R 另测 K=1/2/8 的窗口级指标（用于图4-14 左） |
| RepELA | 训练形态、部署形态 |

测速环境：RTX 3090；FP32、TF32 关闭；关闭确定性算法、`cudnn.benchmark=True`（推理只需前向）；对比方法中的确定性算子（`det_interpolate`、`det_adaptive_avg_pool2d`、`det_grid_sample`）在测速进程中替换为 PyTorch 原生实现，RepELA 用 `native_ops()`；每个对象先在第一张 val 图上核对替换前后预测像素一致率（≥99.99%，记录实际值）。记录 GPU 型号、驱动、torch/cuDNN 版本、CPU 线程数，测前确认 GPU 无其他进程。

测量项：

1. **参数量与权重大小**：可训练参数总数；FP32 权重 MiB；RepELA 两种形态分别列。
2. **计算量**：`torch.utils.flop_counter.FlopCounterMode`（PyTorch 自带，不装新依赖）统计 1×3×512×512 单窗口 FLOPs，报 GMACs=FLOPs/2；扩散方法含全部采样步；GLP 的全图分支按每图一次单独统计（缩略图输入），每图总量 = 35 窗口 + 全图分支。未计入的算子（逐元素、grid_sample 采样等）在 README 列明。
3. **窗口级纯网络延迟**：batch 1，输入已在 GPU；预热 50 次、计时 200 次，每次 CUDA 同步；报中位数与 P95。另测 batch 8 吞吐（窗口/秒）。GLP 的全图分支单独计时。
4. **整图预测时间**：val25，原尺寸 2560×1922，用各方法现有 `predict`（35 窗口、384 步长、batch 8 窗口、CPU 概率融合），不含读图与指标；第一张图作预热不计。另用相同 35 窗口只跑网络（不融合）得到网络时间，二者之差记为"融合与数据搬运"时间；报每图中位数。
5. **推理峰值显存**：整图预测期间 `max_memory_allocated` 与 `max_memory_reserved`（每个对象独立进程、测前清空）。

预计 1.5–2 小时。只计时，不选模，不产生精度结论。

### 7.5 协调器与汇总

- 协调器：`work/launch_ch4_extension_20261024.py`；批次目录 `outputs/launches/p1_ch4_extension_20261024/`。
- 启动时：获取锁 → 校验 PhysicalNet 4 个源文件哈希 → 冻结 `plan.json`（各新文件、数据清单、划分文件、E3R 与 12 个已训练模型的 best.pt 的 SHA256）→ 依次执行 7.1 → 7.2 → 7.3 → 7.4，每阶段前后 `guard()` 校验冻结哈希。
- `status.json` 记录阶段、PID、时间；重跑同一命令可从中断处恢复（已完成阶段跳过，训练从 last.pt 恢复）。
- 结束时：再次校验 PhysicalNet 源文件哈希；输出 `comparison_summary.json` 与 `final_analysis.md`（RepELA val/test 结果、部署等价性、测速总表），然后停止，不进入阶段5。
- 合计约 4.5–5.5 小时。

### 7.6 禁止事项

- 不因 RepELA 或测速结果好坏调整超参、追加轮数或换配方；不跑小论文原配方。
- 不用 test 结果重选任何权重；不重训任何已有模型。
- 不删除或改写已有输出；不写 PhysicalNet。
- 不进入绘图（阶段5 另行开始）。

## 8. 新对话启动提示（可直接复制）

> 请接手 `/root/autodl-tmp/ConditionalDiffusionSeg`，执行第四章扩展批次（E3R 过程导出、RepELA-Net-Small 统一协议训练、统一推理测速）。
>
> 1. 先完整阅读 `docs/CHAPTER4_FIGURES_TABLES_AND_TRAINING_PLAN_20261024.md`，重点是第0节（用户决定）和第7节（执行规格、代码状态、冒烟检查结果）；再读 `HANDOFF.md` 第59–64节和 `docs/TRAINING_PLAN_BASELINES_20261022.md`。
> 2. 代码已写好，不要重写：`work/repela_20261024/`、`work/export_e3r_process_20261024.py`、`work/benchmark_ch4_20261024.py`、`work/launch_ch4_extension_20261024.py`。先复核这些代码与第7节是否一致；如必须修改，在启动协调器前改完，并在汇报中说明。不得修改 `work/repela_20261024/physicalnet_models/` 下的复制文件。
> 3. 确认 GPU 空闲、磁盘空闲 ≥15 GiB、没有活跃的实验进程后，在项目根目录后台启动：`nohup python -B -W ignore work/launch_ch4_extension_20261024.py > outputs/launches/p1_ch4_extension_20261024_coordinator.log 2>&1 &`，记录 PID。
> 4. 协调器会依次执行：过程导出 → RepELA 7 项工程验证 → 训练（100 轮）→ test27 评估（训练形态 + 部署形态）→ 统一测速 → 汇总，然后停止。用 `outputs/launches/p1_ch4_extension_20261024/status.json` 跟踪进度，预计 4.5–5.5 小时。任一校验失败会自动停止：保留日志并报告原因，不要自行修复后重跑，先问我。
> 5. 约束：PhysicalNet 只读，不删除、不写入；训练 seed 42，不做多种子或交叉验证；不纳入 DINO；test27 只评估一次，不按 test 重选；不因结果好坏调整超参或追加训练；不重训已有模型；不删除已有输出。
> 6. 完成后汇报 `final_analysis.md` 的内容：导出校验、RepELA 的 val/test 结果、部署等价性、测速总表。不要进入绘图（阶段5 另行安排）。
