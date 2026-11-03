# 类别采样状态训练 A/B/C：交接

用户先要求下一步训练计划，审阅后明确授权“开始启动训练”。
2026-10-21北京时间08:12:52启动后台协调器，初始PID4192；当前状态必须核对进程与status，不把此PID当作存活证明。
用户跳过固定诊断；必要工程检查已全部通过，工程权重与效果训练隔离。

项目根：/root/autodl-tmp/ConditionalDiffusionSeg。
解释器：/root/miniconda3/envs/diffusion/bin/python。

## 权威入口与文件

- work/launch_categorical_rollout_20261021.py：协调器与worker入口。
- work/categorical_rollout_core.py：训练状态生成、目标、检查点、验证。
- work/verify_categorical_rollout_20261021.py：工程检查。
- outputs/launches/p1_categorical_rollout_20261021/README.md：冻结协议。
- 同目录plan.json：配置、门槛、文件哈希、run路径。
- 同目录status.json、launcher.log、launch_record.json：整批状态。
- 同目录train_A_40.log、train_B_40.log、train_C_40.log：首阶段日志。
- outputs/P1/categorical_rollout/seed_42/：三组run，名称为p1_categorical_rollout_20261021_a/b/c。

冻结后不可改写本批三份工作流、README或工程验证JSON；不修改旧src/scripts/configs、模型或日志。

## 确定方案

A使用原N3局部类别关系，全部GT前向加噪。
B使用原N2全局关系，C使用原N3局部关系；每batch2随机1张GT前向、1张模型实际类别采样状态。
时间点每图独立均匀100/75/50/25/1；模型状态从固定先验开始，沿原五步Bayes类别链运行到目标时间，eval/no_grad，随后只对选定时间的预测反向。
教师无GT输入、无伪标签监督；t100模型状态直接从先验采样，0个教师预测。
三组统一0.5CE+0.5前景Dice，kl_weight=0。A/C隔离状态来源，B/C隔离局部关系；与旧批比较包含KL目标变化。
训练在512裁块上生成状态，不包含整图重叠窗口融合，不能声称完全消除了训练/推理差异。

三组从共同历史Direct兼容权重重新初始化，新增分支一致，优化器全新。
train120记录/119独立RGB、val25，batch2、workers4、FP32、seed42。
AdamW2e-4→1e-6、warmup5、cosine100、wd1e-4、clip1。每20轮完整验证，3407 mIoU选best。
每轮记录模型状态数、各时间点数、教师样本前向数与耗时；相同更新不等于相同算力。

## 队列与停止

A40 → B40 → C40 → 精确epoch40筛选，四项全部满足：

- C mIoU ≥83.93%。
- C前景边界F1 ≥54.56%。
- C−A mIoU ≥0.30pp。
- C−B mIoU ≥0.20pp。

未通过：报告并停止。
通过：三组均恢复完整Adam/scheduler/RNG续至100 → 各同best3408 → 报告并停止。
100轮研究标准仍对比原N0/N1及B、薄层质量和边界；C/A状态贡献另保留≥0.30pp与3408正方向。
不自动添加新批次、test、其他材料、训练seed、步数矩阵或固定诊断。

## 验证与恢复

工程检查通过：两种链所有五个状态逐位匹配原采样器、t100先验、GT隔离、教师无梯度、主分支梯度、三组匹配初始化/随机数消耗、四项门槛独立生效。
三组CPU和真实CUDA512/B2完整模型/Adam/scheduler/RNG恢复后的下一步逐位一致；GPU工程峰值reserved约5.2GiB。
报告见engineering_verification.json和.log，工程权重未用于正式训练。
新JSON/checkpoints原子写入，last.pt决定恢复位置，保留原CSV/TB恢复逻辑。
协调器持outputs/.experiments.lock；阶段边界完整核验前两批结果、条件缓存及历史哈希。

如果进程中断，先确认协调器和worker均未存活，再运行同一入口；不可启动重复队列或绕过哈希检查。
不要重新运行旧批协调器；旧区间对齐批已因40轮门槛未过正常停止。
本次预算为三组40轮约2–3小时，若通过追加2.5–4小时；启动后的实际速度优先于估计。
