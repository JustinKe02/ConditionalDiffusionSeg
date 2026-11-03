> DINO路线已于2026-10-21迁移至同级独立项目 [DINOOpticalSeg](/root/autodl-tmp/DINOOpticalSeg/README.md)，包含训练脚本、四组权重、完整训练过程及验证/测试分割结果。后续DINO工作使用新项目的`scripts/`入口；本目录相关旧位置保留兼容链接。测试报告见[这里](/root/autodl-tmp/DINOOpticalSeg/docs/TEST_EVALUATION_20261021.md)。

# ConditionalDiffusionSeg

## feat/diffusion-seg 开发入口（2026-11-03）

本分支实现RGB条件下的掩码高斯扩散，并研究层数累计阈值表示。正式对照为ordinal、onehot、permuted和direct，共用全局—局部图像骨干。

- [方法设计与文献依据](docs/DIFFUSION_SEG_DESIGN_20261103.md)
- [固定训练计划与资源实测](docs/DIFFUSION_SEG_TRAINING_PLAN_20261103.md)
- 单组入口为 `scripts/diffusion_seg.py`，串行后台队列为 `scripts/run_diffusion_seg.sh`。
- 本批训练参数和进度以固定计划、队列plan/status及各run的history为准。下方保留各日期的历史研究记录。


<!-- RESEARCH_RESET_20261021 -->
**GLP-Flow四组已完成（2026-10-22 06:15:56，北京时间）：** E0/E1/E2/E3 val25 mIoU 85.26/88.32/86.70/86.96%，前景边界F1 67.86/71.06/55.95/54.07%。主门槛未通过：收益来自全局上下文（E1−E0 +3.06pp），流匹配采样（E3）在25/25张图上低于E1、边界F1低17pp，E3−E2仅+0.26pp。前向诊断已确认训练/推理状态不一致（GT插值状态下t=0.25端点mIoU高12.3pp，多步采样无增益）；采样对齐的E3R一组于10-22 08:53开训，预计约11:55完成，见[状态](outputs/launches/p1_glpflow_rollout_20261022/status.json)。见[结果](outputs/launches/p1_glpflow_20261021/final_analysis.md)、HANDOFF第59–60节。

**当前研究与运行状态（2026-10-21，北京时间）：基底相对外观 × 区域对比监督四组已于16:44:34全部完成并停止。** A/B/C/D均完成100轮、6000更新；最佳val mIoU分别为90.6884%/89.5378%/88.5413%/88.4448%。基础分割A最好，光学输入、区域对比及组合均未超过A，D未通过预登记门槛。四组best/完整last及结果已保存，当前GPU空闲、无后续训练队列。见[最终报告](outputs/launches/p1_optical_region_20261021/final_analysis.md)、[机器结果](outputs/launches/p1_optical_region_20261021/comparison_summary.json)、[完整交接](docs/CHAT_HANDOFF_OPTICAL_REGION_20261021.md)，HANDOFF第56节。单seed旧train/val开发筛选，数据及历史结果保留。

**分割结果已导出（2026-10-21 17:09，北京时间）：** 四组best在原25张验证集上完成100张全分辨率预测，保存0–3类别mask、彩色分割、前景叠加、25张原尺寸六面板对比及预览。逐图混淆矩阵与best记录完全一致；原数据、权重和冻结文件不变。见[结果目录与逐图索引](outputs/inference/optical_region_val_20261021/README.md)、[完整压缩包](outputs/inference/optical_region_val_20261021.zip)。本次未进行27张test推理。

**下方条目均为历史记录。** “已开训”“尚未训练”和旧路线建议只描述各自日期；不构成当前启动或恢复授权。当前决策以本段、新方案及对应批次最终结果为准，历史冻结文件不改写。

**最新实际状态：材料条件类别扩散四组已开训（2026-10-20 18:31北京时间）：** 新实现、145张共同RGB先验缓存、CPU8项与四组512/B2 CUDA恢复检查、完整训练/滑窗吞吐选型及冻结已完成。N0正在训练（18:32已完成4/100轮、240更新），随后N1→N2→N2同best3408→N3→N3同best3408→报告停止。共同seed42、trainbatch2、windowbatch16，各新增100轮/6000更新/12000裁块曝光；共享历史Direct40预算另列。旧实验不重跑。见[实时状态](outputs/launches/p1_material_categorical_20261020/status.json)、[冻结计划](outputs/launches/p1_material_categorical_20261020/plan.json)、[最新完整交接](docs/CHAT_HANDOFF_MATERIAL_CATEGORICAL_20261020.md)。HANDOFF第53节。

**第三章下一批训练计划已整理（2026-10-20）：** [具体四组计划](docs/TRAINING_PLAN_CHAPTER3_MATERIAL_DIFFUSION_20261020.md)：N0直接、N1确定性五步、N2全局关系类别扩散、N3材料条件类别扩散；建议各新增100轮/seed42，复用共同Direct先验并披露历史40轮。质量与整图成本共同形成第三章→第四章的衔接证据。当前仅proposal，未实施/测速/排队/开训，旧批次不重跑。HANDOFF第52节。

**最新研究方向（2026-10-20）：** 用户要求停止既有方案的小幅涨点试错，先检索近期diffusion并制定面向二维材料的新计划。已完成2025—2026文献筛选，推荐“基底相对光学对比度条件下的离散类别扩散修正”，见[文献与完整改进计划](docs/LITERATURE_DRIVEN_DIFFUSION_PLAN_20261020.md)。提案包含四组有明确问题分工的比较、建议共同100轮新增预算与停止条件，尚未实施或排队；当前不执行此前P0/P1额外诊断。HANDOFF第50节。

**最新完成状态（2026-10-20）：** P0/P1批次已于北京时间16:51:37全部结束并停止，GPU空闲，无后续队列。各从零40轮/2400更新/seed42。P0主/3408为81.9788/82.0357%，P1为82.8448/82.8353%；P1比P0提高0.8660/0.7996个百分点，但未达到相对S0/S1及薄层的完整筛选要求。[最终结果分析](outputs/analysis/state_objective_final_20261020T092155Z/README.md)。此前启动条目为历史快照。HANDOFF第49节。

**最新实际状态：有界状态对照批次已启动（2026-10-20 15:45北京时间）：** 用户授权后，已完成训练集零更新诊断、68项测试、真实batch吞吐与恢复检查。当前P0通用有界状态从零训练中，P1类别概率一致状态随后执行；两组各40轮、seed42、batch2、2400更新。历史S0/S1/S2复用，不重跑；不涉及test或其他材料。顺序为P0→同best3408→P1→同best3408→报告停止。此前“无队列”均为历史状态，以[实时状态](outputs/launches/p1_state_objective_20261020/status.json)为准。[训练计划](docs/TRAINING_PLAN_STATE_OBJECTIVE_20261020.md)、[最新完整交接](docs/CHAT_HANDOFF_STATE_OBJECTIVE_20261020.md)。HANDOFF第48节。

**章节定稿规范与本批完成（2026-10-20）：** 静态语义平台批次已于北京时间14:01全部完成并停止，S2主采样82.3998%、同best3408为82.4174%，筛选未通过，无后续队列。用户要求正式论文不放探索失败案例，相关记录留项目；后续新图统一完整四边框。第3章侧重有效学习/状态修正机制及薄层质量，第4章侧重RepELA结构与精度—资源取舍，第5章侧重实际部署。[两章定位、图表分工与下一步计划](docs/CHAPTER3_CHAPTER4_POSITIONING_AND_NEXT_PLAN_20261020.md)。当前论文文件是历史材料初稿，下一版按新提纲重组。HANDOFF第47节。

**主采样训练结果（2026-10-20 13:59北京时间）：** S1/S2均完成40轮、各2400更新。S0/S1/S2最佳val mIoU为83.1345/82.6806/82.3998%，S2主筛选未通过；当前仅剩同best的3408推理复核，完整批次是否结束请查实时状态。训练无非有限日志，预算、冻结哈希、混淆矩阵/边界指标复算通过。[当前分析与中文曲线](outputs/analysis/static_semantic_progress_20261020T055756Z/README.md)。不自动延长或追加实验。HANDOFF第46节。

**正式训练已启动（2026-10-20 12:32北京时间）：** 用户明确授权启动。新平台、B零步诊断、60项测试、真实batch/滑窗测速及冻结已完成；S0经完整等价审计复用历史Direct，不重跑。S1已完成第1/40轮，之后依次执行S2 40轮和同best的3408验证，然后报告停止。共同seed42、batch2、512、每组2400更新，仅MoS₂ train/val。实时状态优先于下方历史条目，见[批次状态](outputs/launches/p1_static_semantic_20261020/status.json)、[冻结计划](outputs/launches/p1_static_semantic_20261020/plan.json)、[最新完整交接](docs/CHAT_HANDOFF_STATIC_TRAINING_20261020.md)。HANDOFF第45节。

**材料数据复查与训练计划（2026-10-20）：** 已只读审计PhysicalNet中WS₂、石墨烯、补充集和外部MoS₂，见[数据复查与分阶段计划](docs/DATASET_REVIEW_AND_TRAINING_PLAN_20261020.md)、[审计记录](outputs/analysis/material_dataset_audit_20261020/README.md)。WS₂发现跨划分同时间戳前缀，需先核查母图/样品来源；公开与补充石墨烯标签不同。当前仍无新平台/测速/S0S1S2训练启动；首批最多三次MoS₂40轮，跨材料验证条件登记。HANDOFF第44节。

**论文初稿与中文配图（2026-10-19）：** 已基于既有结果完成[第3章Word初稿](docs/thesis/chapter3_20261019/chapter3_draft.docx)、[PDF阅读版](docs/thesis/chapter3_20261019/chapter3_draft.pdf)、9组中文图及来源审计。[最新续接交接](docs/CHAT_HANDOFF_THESIS_DRAFT_20261019.md)记录已完成/待补、字体替代和用户参考图风格。仅离线写作绘图，没有新增训练/推理；新架构、测速和S0/S1/S2仍是提案。HANDOFF第43节。

**New-chat handoff (2026-10-19):** [Standalone summary and next-work prompt](docs/CHAT_HANDOFF_20261019.md). Latest live check: GPU idle, no training/evaluation/queue;11P1 complete, current reference screens failed, v5newarchitecture/profile/S0S1S2 remain unimplemented proposals. HANDOFF section42.

**Chapter structure and gaps (2026-10-18):** [Target chapter outline and evidence mapping](docs/CHAPTER3_OUTLINE_AND_EVIDENCE_GAPS_20261018.md) distinguishes current11P1 (9scratch＋2adaptation) from unvalidated core-method/public-baseline/formal evidence. Single trainingseed42 per user; no new experiment started. Fresh inventory: outputs/analysis/chapter3_chapter_design_20261018/; HANDOFF section40.

**Next plan after A/B failure (2026-10-18):** Stop current reference-residual route. [Proposed next study](docs/NEXT_PLAN_STATIC_SEMANTIC_CONDITION_20261018.md): train-only failure diagnosis, realbatch/window-throughput selection, then shared strongRGB direct/deterministic5-step/diffusion5-step controls, trainingseed42 only. Planning only; no new run queued. HANDOFF section39.

**A/B completed (2026-10-18 18:27 Asia/Shanghai):** Both40adapterepochs finished. F0valmIoU81.4125/81.3352%, A81.4661/81.3762%, B81.4696/81.3843%; B-A only+0.0035/+0.0081pp, predeclared screen failed. Frozen-base/rejected-image fallback preserved; no further queue, GPU idle. [Result interpretation](outputs/launches/p1_reference_adapter_20261018/final_interpretation.md); HANDOFF section38. Earlier startup notes below are historical.

**A training started (2026-10-18 16:02 Asia/Shanghai):** All20native rejection-image/seed fallback checks passed; plan frozen and A trainer launched automatically, with B next. Read [live status](outputs/launches/p1_reference_adapter_20261018/status.json). Earlier queued/preflight notes below are historical. No additional variants/seeds/test queued.

**A/B batch authorized and queued (2026-10-18):** FrozenF0 reference residual implemented;46tests, old-weight compatibility,512FP32 and exactA/Brecovery passed. Native rejected-image fallback audit is running before automatic A→B training (each40adapterepochs,seed42). Read [live status](outputs/launches/p1_reference_adapter_20261018/status.json) and [batch scope](outputs/launches/p1_reference_adapter_20261018/README.md); HANDOFF section37. This startup note is not a completed-training claim. No extra variants/seeds/test queued.

**Latest planning preference (2026-10-18):** User cancels multi-training-seed experiments; use trainingseed42. [Concrete optimization directions](docs/CONCRETE_OPTIMIZATION_DIRECTIONS_20261018.md): independent reference residual first, conditional noise/state gating second, class-aware crop controls separately. Earlier multi-seed proposals are historical. Planning only; no new training started. HANDOFF section36.

**F0/F1 completed (2026-10-18 14:53:34 Asia/Shanghai):** F0valmIoU81.41/81.34%, F1withreference80.58/80.52%; preregistered screen failed, no further queue. Both40epochs/2400updates; no test. Posthoc reference-accepted15val images improved but rejected10 declined; retain overall failure and distinguish this hypothesis from established benefit. [Result interpretation](outputs/launches/p1_reference_fusion_20261018/final_interpretation.md); HANDOFF section34. Earlier launch notes are historical.

**Reference-fusion pilots launched (2026-10-18 12:23 Asia/Shanghai):** User authorized implementation. Train-only RGB-reference gate passed (71/120accepted);40tests, old-weight compatibility,512FP32 and exact F0/F1 recovery passed. Detached batch `p1_reference_fusion_20261018` runs only F0→F1, each40epochs/2400updates, then samebest3408fullval and report/stop. No test/extra variants/seeds/formal queue. Live [batch scope](outputs/launches/p1_reference_fusion_20261018/README.md), status.json and console.log; HANDOFF section33. Earlier planning notes are historical.

**Chapter3 positioning confirmed:** Algorithm innovation. Current candidate hypotheses, direct recycling/high-noise prior art and revised research order are in [algorithm innovation brief](docs/CHAPTER3_ALGORITHM_INNOVATION_BRIEF_20261018.md). Prioritize literature differences and a bounded controlled candidate study before broad repetition/200-epoch matrices. The candidate is unimplemented and novelty is unestablished; no new experiments are queued.

**Chapter3 evidence review (2026-10-18):** Seven P1 trainings and three fixed test evaluations inventoried. Current gaps, independent-test exposure, staged6+6 P1 repetition proposal, conditional formal matrix, public baselines and chapter outline: [evidence and supplement plan](docs/CHAPTER3_EVIDENCE_AND_SUPPLEMENT_PLAN_20261018.md). Inventory/CSV/proposed plan: `outputs/analysis/chapter3_evidence_audit_20261018/`. This is planning only; no new training, inference or test queue started.

**Test evaluation completed (2026-10-18 11:12:50 Asia/Shanghai):** Frozen comparison on all27 test images: Direct mIoU88.69%, Iterative5 89.81%, best single-head high-noise+recycling Diffusion20 86.86%; foreground boundaryF1 47.17%/50.43%/42.68%; full evaluation158.1/279.7/748.6s. All checkpoints were selected on validation before test; no test-driven tuning/reselection. One training seed42,40-epochP1; source/sample independence still pending. Results and audit: `outputs/launches/p1_test_20261018/{comparison_summary.json,comparison.csv,final_analysis.md}`; HANDOFF section29. No queued training/evaluation remains. Earlier validation rankings and status notes below are historical.

**Latest completion (2026-10-18 10:06:54 Asia/Shanghai):** Dual-head batch p1_dual_head_20261018 completed40epochs/2400updates and both full25-val sampling-seed evaluations. Best epoch40 mIoU79.65%/79.56% versus matched single-head high-noise+recycling81.23%/81.14% (3407/3408); monolayer and boundary scores also decreased. Keep single-head as the current best diffusion recipe. Analysis: outputs/launches/p1_dual_head_20261018/{comparison_summary.json,final_analysis.md,final_analysis.json}; HANDOFF section27. No further training/evaluation started. Earlier launch/status notes below are historical.

**Latest completion (2026-10-17 21:32:31 Asia/Shanghai):** All authorized queues finished. Uniform+recycling mIoU79.45%/79.38% versus best high-noise+recycling81.23%/81.14% (sampling seeds3407/3408). Complete single-training-seed2×2 ablation and1/5-step measurements: `outputs/launches/p1_uniform_recycling_20261017/{comparison_summary.json,final_analysis.md}`. Five-step high-noise+recycling retains81.11%/81.02% at lower cost. No new training/test started. See HANDOFF section24; earlier queue notes below are historical.

**Current queue (2026-10-17 20:08 Asia/Shanghai):** Uniform-recycling completion batch `p1_uniform_recycling_20261017`, PID101373, is running: existing recycling best1/5-step full-val measurements, then exactly one40-epoch uniform-timestep+recycling P1 and candidate3408 evaluation.26 tests and full model/optimizer/scheduler/RNG resume equality passed. Live state and plan: `outputs/launches/p1_uniform_recycling_20261017/`; see HANDOFF section23 and RUNBOOK section14. Earlier completion notes below describe prior batches.

**Latest completion (2026-10-17 19:33:14 Asia/Shanghai):** Recycling P1 batch completed; no active training queue. High-noise+recycling mIoU81.23%/81.14% at sampling seeds3407/3408 versus high-noise75.90%/76.13%. Best remains epoch40 selected by3407; directional screen passed. Follow-up on the fixed12 train crops did not show net20-step correction, so the mechanism remains under investigation. Results and interpretation: `outputs/launches/p1_recycling_20261017/{comparison_summary.json,analysis/}`; see HANDOFF section22. Earlier queue/status notes below are historical.

**Current authorized queue (2026-10-17 18:09 Asia/Shanghai):** `p1_recycling_20261017`, detached PID83800, runs baseline1/5-step full-val measurements, exactly one40-epoch high-noise+predicted-state-recycling P1, then candidate seed3408 evaluation.22 tests and model/optimizer/scheduler/RNG resume equality passed. Live state: `outputs/launches/p1_recycling_20261017/status.json`; plan and schedule are in that batch directory. Actual512 training follows the baseline measurements. See HANDOFF section21 and RUNBOOK section13.

**Latest completed batch (2026-10-17 17:26:55 Asia/Shanghai):** High-noise P1 and both seed3408 evaluations completed. New diffusion mIoU75.90%/76.13% versus original68.59%/68.76% (sampling seeds3407/3408); the pilot screen passed. No further experiment has started. Comparison and forward-only follow-up: `outputs/launches/p1_high_noise_20261017/{comparison_summary.json,analysis/}`; see HANDOFF section20. Launch notes below are historical.

**High-noise P1 comparison launched (2026-10-17):** Batch `p1_high_noise_20261017` runs one independently initialized40-epoch diffusion pilot with50% high-noise timesteps, then baseline/new seed3408 full-val evaluations. Live state and logs: `outputs/launches/p1_high_noise_20261017/`. See HANDOFF section19 and RUNBOOK section12. Prior completed P1 results below remain preserved.

**P1 completed (2026-10-17 15:31):** All three 40-epoch runs completed. Best val mIoU: direct83.13%, iterative81.95%, diffusion68.59%. Analysis and proposed next-stage checks are in HANDOFF section17 and `outputs/launches/p1_cosine_20261017/analysis/`. No new training was started. Earlier launch/status notes below are historical.

**P1 launch (2026-10-17):** Batch `p1_cosine_20261017` is running: fresh three-model P0 recheck passed; full-val timing precedes automatic direct/iterative/diffusion 40-epoch P1 training. Live stage and console: `outputs/launches/p1_cosine_20261017/`. All methods start from scratch under the existing uniform-t cosine pilot recipe. See HANDOFF section16; older status notes below are historical snapshots.

**Latest P0 ablations (2026-10-17 12:57):** Both independent 600-update diffusion P0 trials passed the unchanged gate. High-noise strata: mIoU 98.47%/98.44%; uniform timesteps with cosine LR: 96.49%/96.93% (sampling seeds3407/3408). These are fixed training-crop diagnostics. Batch `outputs/launches/p0_ablations_20261017/` is completed; P1 has not started. See HANDOFF section15; preceding continuation notes describe older snapshots.

**Live continuation (2026-10-17):** diffusion environment and all preflight checks passed; all P0 training finished, but the diffusion memorization gate failed and batch `diffusion_resume_20261017` stopped before P1. Live state: `outputs/launches/diffusion_resume_20261017/status.json`; console: `outputs/launches/diffusion_resume_20261017/console.log`. The earlier status below is a historical baseline; see HANDOFF sections 11-12.

二维材料光学图像的四类别条件扩散语义分割研究，对应硕士论文第3章。第4章是相邻PhysicalNet中的RepELA-Net轻量模型，第5章是其本地与边缘部署。

**当前状态：首版代码可运行，已完成核心测试、真实数据两步调试、512尺寸显存检查和断点恢复一致性验证。未启动40轮P1或200轮正式实验，也未评估测试集性能。** 小样本过拟合与完整验证集耗时仍待检查。

## 从这里开始

2026-10-17新增系统盘`diffusion`环境及Bash入口：`scripts/setup_diffusion.sh`安装固定版本依赖，`scripts/run_experiments.sh --stage all --batch-id <唯一ID>`按顺序执行环境/恢复检查、P0固定训练裁块过拟合、完整验证集测速、P1三组40轮。仅当P0达到预设门槛后进入P1；不启动正式200轮矩阵或测试集评估。运行与恢复说明见RUNBOOK第10节。

- [新对话交接文档](HANDOFF.md)：背景、已定方案、当前进度与下一步。
- [运行手册](docs/RUNBOOK.md)：配置、训练、恢复、TensorBoard、评估及绘图命令。
- [实验设计](docs/EXPERIMENT_PLAN.md)：研究问题、对照、消融与论文结构。
- [日志规范](docs/TENSORBOARD_LOGGING.md)与[磁盘预算](docs/STORAGE_BUDGET.md)。
- [数据全量审计](data_manifests/full_audit.json)、[512/batch2显存检查](docs/resource_smoke.json)、[流水线验证](docs/pipeline_verification.json)。

## 数据与配置

数据直接读取`/root/autodl-tmp/PhysicalNet/Mos2_data`，不复制原图。训练120、验证25、测试27张；类别0背景、1单层、2少层、3多层。图像和mask尺寸2560×1922，标签0—3已全量核验；样品来源独立性仍待核查。

`configs/debug.json`、`pilot.json`、`formal.json`是可执行配置。`experiment_plan.json`是研究计划登记，不能作为训练配置。

**正式候选统一512×512、batch2、FP32。** batch4下五步迭代模型OOM，所以三组一起改为batch2；原记录保存在`docs/resource_smoke_batch4.json`。

## 代码入口

```text
src/conditional_diffusion/
    data.py             数据、增强、split哈希与审计
    model.py            直接/迭代/条件扩散U-Net
    diffusion.py        cosine扰动、x0表示与DDIM采样
    objectives.py       CE、Dice、去噪MSE与边界监督
    metrics_impl.py     混淆矩阵、IoU/F1、边界F1
    inference.py        固定滑窗、整图噪声、概率融合
    runtime.py          TensorBoard/CSV、磁盘检查、原子checkpoint
    trainer.py          训练、验证、选模与完整恢复
scripts/
    train.py            单次训练/恢复
    evaluate.py         验证、采样设置比较、受控测试入口
    plot_curves.py      CSV到PDF/SVG/PNG
    audit_data.py       全量数据审计
    check_storage.py    只读整套实验磁盘预算
    smoke_resources.py  计划尺寸一次合成训练步的显存检查
    verify_pipeline.py  两步真实数据训练及恢复/日志/评估/绘图检查
```

训练默认按run_id隔离输出；只保留best权重与滚动last完整状态。TensorBoard记录训练损失、验证指标、学习率、梯度范数、资源与固定样本图，并同步保存CSV/JSON。数据盘保留8 GiB空间，不自动删除其他实验。

首版参考SegDiff/DDIM的思路进行独立适配，不是原论文严格复现。已有通用扩散方法、类别权重及边界损失不自动构成论文创新；贡献要由后续实验支撑。
