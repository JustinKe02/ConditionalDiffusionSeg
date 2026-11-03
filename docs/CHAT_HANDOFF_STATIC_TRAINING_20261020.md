> 最新续接：用户再次授权后，有界状态P0/P1批次已于2026-10-20北京时间15:45启动。请先读[新批次完整交接](CHAT_HANDOFF_STATE_OBJECTIVE_20261020.md)和实时状态；以下是此前静态平台批次的历史交接。

# 静态RGB语义条件平台训练交接

最终状态与写作新要求：本批已于北京时间2026-10-20 14:01:32全部完成并停止；S2主采样82.3998%，同best3408为82.4174%，所有筛选未通过，GPU空闲。正式论文不放探索失败案例，原记录留项目；后续图统一完整边框。继续工作前补读HANDOFF第47节及`docs/CHAPTER3_CHAPTER4_POSITIONING_AND_NEXT_PLAN_20261020.md`，其中规定第3章机制/过程与第4章轻量结构/资源的图表分工。新诊断/训练仅为计划，没有新队列。下方“3408待复核”等属于此前时间快照。

最新更新（2026-10-20 13:59北京时间）：S1/S2均已完成40轮、各2400更新，当前仅剩同S2 best的3408复核。主采样S0/S1/S2 mIoU为83.1345/82.6806/82.3998%，主要筛选未通过；后续只完成已登记复核和报告后停止。详细分析见HANDOFF第46节及`outputs/analysis/static_semantic_progress_20261020T055756Z/README.md`。下文启动时进程/轮次为历史快照；是否全批完成必须核对实时`status.json`。

登记时间：2026-10-20。用户明确要求“开始启动训练”，本文件取代此前“新结构尚未实施、没有排队”的实时状态描述。历史计划与论文文档仍作为对应日期的快照保留。

## 1. 已启动的范围

批次：`outputs/launches/p1_static_semantic_20261020`。正式冻结时间见`plan.json`；进入S1训练时间为2026-10-20 04:32:15 UTC / 12:32:15北京时间。规范协调器PID启动时为15998，后续以`launcher.pid`、`status.json`、实际`ps`为准。

顺序：复用S0→新训练S1→新训练S2→同S2 best的3408完整val复核→生成报告→停止。S0是历史`outputs/P1/direct/seed_42/p1_cosine_20261017`，已完成40轮，best第30轮，val mIoU83.1344865%。**不重跑S0，本批只有两次新效果训练。**S1开始时还没有新验证结果，不能写成改进已成立或实验已完成。

新run：

- S1：`outputs/P1/static_semantic/seed_42/p1_static_semantic_20261020_s1`
- S2：`outputs/P1/static_semantic/seed_42/p1_static_semantic_20261020_s2`

都为从零40轮、seed42、512、batch2、2400次更新、4800次样本曝光，原120条记录/119唯一RGB。m169与m56重复RGB且有929像素标注差异，保留原始清单与标注，不静默去重。每10轮完整val25选best。没有新test、其他材料、200轮、门控、解冻或参照挽救队列。旧11次P1已完成（9次scratch＋2次适配），本批两次新训练未完成前不可计入“已完成清单”。

## 2. 实际结构与公平性

新平台代码`src/conditional_diffusion/static_semantic.py`；独立训练/滑窗路径`static_experiment.py`。旧源文件、模型工厂、训练器及配置未修改，旧221项冻结条目仍匹配；新增批次独立冻结，不能绕过旧launcher哈希守卫。

RGB编码32/64/128/256，每级2个GN/SiLU残差块；状态16/32/64/128，每级2个带时间残差块，128维时间嵌入，1×1无偏置投影相加，共享高分辨率解码器和4类单头。无预训练、参照、注意力、新类别采样或新损失。

**实现发现：S0的RGB编码器/解码器与历史Direct实际完全等价，不能声称相对旧Direct新增了更强主干。**136个旧张量映射、真实32/64裁块连续更新的输出/梯度/参数/Adam、旧best固定小val裁块输出均逐位一致。共同batch2、滑窗1、数据/优化/初始化/选模等协议相同，所以复用S0。新平台主要改变的是相对旧F0较弱图像条件路径的语义表达，以及独立状态融合方式。

S0参数4,929,156；S1/S2均5,939,156，状态/时间新增1,010,000。两个独立CPU初始化域都使用seed42；共享RGB/解码/头完全相同，S1/S2状态也完全相同，不占位伪造S0容量。

S1从编码均匀概率−0.5开始，[999,749,500,250,0]五次确定性修正、2softmax−1更新，梯度贯穿五步。S2为T1000/cosine/x0，高噪声50%、回灌50%，教师当前模型终端高斯输入、无梯度、独立噪声。S0/S1损失0.5CE+0.5前景Dice；S2额外1.0MSE。总loss不可直接比较。

每个推理图块仅编码一次RGB；状态和解码每步执行。S2缓存DDIM与旧逐步重算公式在相同噪声下逐位一致，最终单头clamp(x0)再softmax融合。全图噪声坐标、窗口顺序和概率融合不变。滑窗512/384、batch1；35窗/图，S0 NFE35，S1/S2 NFE175。S2五步3407选best，3408只复核同一best，不重选、不代表另一次训练。

共同AdamW 2e−4→1e−6，warmup5/cosine，weight_decay1e−4、clip1。FP32张量，确定性算法；显式沿用历史后端设置：matmul TF32关闭、cuDNN TF32允许。

## 3. 已完成的前置工作

B训练集零优化步诊断：`outputs/analysis/b_zero_step_diagnostic_20261020/`。8个预定接受参照训练裁块，真实/置零/置换C共用q/noisy/t；没有optimizer构造或更新，冻结权重与源输入不变，硬回退精确通过。置零/置换仅改变约0.001%的预测像素；修正较小，CE/Dice与MSE梯度有方向对抗，未发现断接/冻结/回退错误，不据此改变原A/B失败结论或延长训练。

60项CPU测试通过；真实512/batch2 CUDA模型/Adam/RNG恢复逐位一致、五步缓存/未缓存输出逐位一致。额外5个恢复边缘测试覆盖epoch0检查点、epoch40恢复、best旁文件损坏/缺失、初始化目录归档及拒绝丢弃未 checkpoint 训练证据。

工程预检发现并修复中断恢复边缘，发生在效果冻结之前。第一版预检文件、原临时哈希守卫停止日志保留在`preflight_attempt1`、`launcher.log`等。`recovery_hardening_registration.json`记录明确版本变更，并以AST证明训练测速调用的更新/预测等函数及其他依赖保持不变，保留适用测速结果，没有重跑已完成效果实验。规范日志是`launcher_canonical.log`。

## 4. 吞吐选择与成本

各模式训练batch2/4/8/16均已独立进程实测10步预热＋30步计时，包含真实I/O、增强、适用教师、反向与AdamW。测速权重隔离，不用于正式初始化或best。

| 模式 | batch2张/秒 | batch2峰值reserved GiB | 其他档位 |
|---|---:|---:|---|
| S0 | 9.568 | 4.23 | b4/b8可用，b16 OOM |
| S1 | 2.058 | 16.43 | b4/b8/b16 OOM |
| S2 | 7.161 | 5.10 | b4/b8可用，b16 OOM |

共同batch2由S1五步完整反传的内存需求约束，保留超过10%显存余量；不是仅验证batch2能运行。所有OOM记录保留。滑窗1/4/8/16也已实测，保留全图噪声并审计概率/标签差异，拆分读取、传输、采样、融合、指标。为精确复用S0既有选模协议而统一采用滑窗batch1，不宣称它吞吐最快；完整表与选择理由在`throughput_summary.json`。

`complexity.json`：512单窗仅Conv/Linear MAC，S0 94.8835 GMAC，S1/S2缓存五步360.6854 GMAC。不含归一化/插值/softmax/DDIM算术、融合、I/O或指标，不能标成FLOP或实测时延。总ETA等第一轮完整验证后再估，不能套旧A/B时长。

## 5. 冻结门槛与停止

S2−S0四类mIoU≥0.50pp，S2−S1≥0.30pp，相对S0单层/少层平均IoU≥+0.50pp，各薄层类及前景边界F1下降≤0.50pp，3408两项mIoU差保持正向。这是探索筛选，不是显著性或原创性证明。三组固定40轮，不按中途排名修改预算；报告后停止，即使未通过也不自动挽救。

WS₂跨划分时间戳来源疑点、公开数据出处、石墨烯语义/无独立test、外部MoS₂原始类别依据仍待核查。见`docs/DATASET_REVIEW_AND_TRAINING_PLAN_20261020.md`。跨材料不在本批队列。

## 6. 继续协作的操作

解释器`/root/miniconda3/envs/diffusion/bin/python`。数据只读`/root/autodl-tmp/PhysicalNet/Mos2_data`，不改PhysicalNet。保留所有权重、日志、失败、临时归档，不绕过冻结检查。

先查`status.json`、`launcher.pid`、`ps`、`nvidia-smi`及run的`status.json`/CSV/日志；活动协调器或训练存在时，不再启动重复进程。TensorBoard原`outputs/P1`可自动发现新run。

需要恢复时，只用同一冻结入口`work/launch_static_semantic_20261020.py`，它持有`outputs/.experiments.lock`。先确认没有存活协调器/子进程，核对plan/workflow/source哈希，再从last完整恢复。不要用旧`scripts/train.py`处理新schema2配置；不要强行运行旧launcher。

新完整last从epoch0保存；best主检查点原子包含指标及混淆矩阵，可重建旁文件；epoch40 last已提交时直接恢复completed；初始化未完成目录和未提交临时文件归档保留。磁盘8GiB保留、项目26GiB/run1GiB上限不变。

论文第3章仍是`docs/thesis/chapter3_20261019/`的历史结果初稿，尚非最终核心算法结论。9组中文图采用用户允许的临时NotoSerifSC/Tinos字体替代，模型图是旧HR结构而不是新平台；新结果完成后另行更新，不能提前改写失败历史。

## 7. 可复制到下一对话的提示

> 请接手/root/autodl-tmp/ConditionalDiffusionSeg。先完整阅读docs/CHAT_HANDOFF_STATIC_TRAINING_20261020.md、HANDOFF.md第45节、本批outputs/launches/p1_static_semantic_20261020/{README.md,plan.json,status.json}，并核查真实PID、GPU、子进程、run状态与CSV。用户已授权启动本批MoS₂最小平台研究；S0经全协议等价审计复用历史Direct，只有S1/S2两次新scratch40，seed42、batch2、512、各2400更新，S1→S2→同best3408复核→报告停止。不要重跑S0或另起重复队列，不修改冻结源/配置/工作流，不绕过哈希，不改PhysicalNet，不启动test、跨材料、200轮或参照挽救。3407/3408仅同权重推理采样检查。新结构、60测试、GPU恢复/缓存、B零步诊断、全batch吞吐已完成；以实时状态判断训练完成情况。S0实际上与旧Direct同结构，不声称新强RGB主干；新best结论等真实val。其他材料有来源/分组问题，先保持未入队。所有数据、权重、日志与失败必须保留。若训练已结束，审计混淆矩阵/预算/预设筛选后解释结果并更新论文；没有授权扩展矩阵。
