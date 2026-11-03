> **研究方向更新：** 用户已要求停止P0/P1式小幅试错，先做近期文献与二维材料研究计划。最新提案见docs/LITERATURE_DRIVEN_DIFFUSION_PLAN_20261020.md、HANDOFF第50节；没有新训练或诊断队列。

> **完成更新：2026-10-20北京时间16:51:37已全部结束，筛选未通过，GPU空闲且无队列。P0/P1主mIoU为81.9788/82.8448%，同best3408为82.0357/82.8353%。请先读HANDOFF第49节及outputs/analysis/state_objective_final_20261020T092155Z/README.md；下方运行描述是启动时历史快照。**

# 有界状态与类别概率一致性训练交接

最新用户授权：**“开始罗列训练计划并训练”**。本批已执行前置诊断、预检、冻结并启动，不能再按上一版“下一步仅为计划”处理。

## 当前批次与实际状态

- 项目`/root/autodl-tmp/ConditionalDiffusionSeg`，解释器`/root/miniconda3/envs/diffusion/bin/python`。
- 新批次`outputs/launches/p1_state_objective_20261020`；冻结时间2026-10-20 07:45:00 UTC / 15:45:00北京时间。
- 协调器启动PID27506，CUDA训练PID27532；以真实ps/GPU及`launcher.pid`、`status.json`为准。
- 文档核实时P0已完成第9/40轮、540更新，继续训练；P1未启动，已排队。DataLoader子进程继承命令行，不是多次训练。
- 顺序：P0从零40轮→P0同best3408→P1从零40轮→P1同best3408→比较报告→停止。不要启动重复队列。
- 运行入口`work/launch_state_objective_20261020.py`；规范日志`launcher.log`，训练日志`train_P0_result.log`、`train_P1_result.log`。
- run位于`outputs/P1/state_objective/seed_42/p1_state_objective_20261020_p0`及`_p1`。

详细训练计划见`docs/TRAINING_PLAN_STATE_OBJECTIVE_20261020.md`及新批次README/plan。旧S0/S1/S2全部复用，不重新训练；没有其他材料、test、200轮、其他训练seed、门控或参照挽救队列。

## 研究假设与控制

新`state_objective.py`为原静态语义平台添加固定、无参数的x0映射。P0：x0=tanh(z)，通用有界对照；P1：x0=2softmax(z)−1，类别概率一致候选。均为同一个四通道raw logits头，不是双头；参数均5,939,156，与原S2相同，初始state_dict逐位一致，从零seed42初始化。

共同损失0.5CE(z)+0.5前景Dice(softmax z)+1.0MSE(映射后的x0,2onehot−1)。回灌教师与DDIM都使用映射后的x0；最终两组都返回raw logits再softmax融合。保留连续高斯cosineT1000、高噪声50%、回灌50%、五步DDIM，不是离散分类扩散。没有新增参照、注意力、新类别采样、边界损失或预训练。

旧S2最终读出为clipped x0，因此与旧S2的比较同时包含状态映射和读出改变，不能将全部增量归为仅修改MSE。P1对P0在读出、容量和预算一致时比较两种映射。概率/模拟比特参数化已有文献，本批只做平台修正筛选，不直接声称原创或保证改进。文献与因果边界见`research_rationale.json`。

两个新训练都原train120条/119唯一RGB、512裁块、每图每轮一个裁块、batch2、40轮、2400更新、4800曝光；同增强/样本顺序、AdamW lr2e−4→1e−6、warmup5/cosine、weight_decay1e−4、clip1。FP32张量，确定性，matmulTF32false/cuDNN TF32true匹配历史。

每10轮完整val25以全局4类mIoU选best；固定512/384滑窗、batch1、五步DDIM、单采样3407。3408只复核各自同best，不重选、不算独立训练。35窗/图，NFE175/图、4375/25图。RGB图块特征缓存一次，状态/解码每步执行。

主候选固定P1；筛选：P1−S0 mIoU≥0.50pp、P1−S1≥0.30pp、P1−P0≥0.20pp；相对S0单层/少层平均IoU≥+0.50pp，各薄层类与前景边界F1下降≤0.50pp；3408相对S0/S1及P0同采样mIoU均正向。完整跑完两组，不按中途排名改预算或事后将P0重命名为原主候选。报告后停止，无自动扩展。

## 已完成的诊断与预检

1. `outputs/analysis/static_state_diagnostic_20261020/`：现有S1/S2训练集8个预定含薄层裁块，512状态/时间敏感性及五步轨迹、64全参数梯度。零optimizer、零更新、输入与权重哈希不变。状态分支起作用；S2所选轨迹新增错误多于纠正。64上低/中噪声CE与raw-MSE常负向，CE与prob-MSE更一致。仅局部训练诊断，GT扰动状态/缩小梯度检查不证明唯一原因或验证收益，旧筛选结论不改。
2. 68项CPU测试（原60+新增8）通过：同初始权重/容量、值域/类别和、raw最终读出、独立无梯度教师、cached/旧dict-DDIM、各损失有效梯度、两映射完整model/Adam/scheduler/RNG恢复。测试后进程内旧绑定恢复。
3. 两种映射真实512/batch2 CUDA恢复与缓存DDIM逐位一致，工程checkpoints隔离，不用于效果训练。
4. P0/P1分别真实batch2 10预热+30计时，含真实I/O/增强/教师/反向/clip/AdamW，约7.19/7.16裁块每秒，reserved均约5.1GiB。此前S0/S1/S2已测2/4/8/16且S1限制共同batch为2，继续共同2以复用对照。本批只补新映射受影响路径，不重复原完整选型。
5. 两种映射新原尺度整图滑窗batch1成本检查通过，读取/传输/采样/融合/指标分开记录。独立代码审查无阻断项。

## 冻结与恢复

原47源/配置、25工作流及旧批次/权重/日志等合并412项历史文件SHA保持一致。新源只增加`src/conditional_diffusion/state_objective.py`和两组新配置；新plan为50项源码/配置及专用工作流另行冻结。旧源备份`work/backups/pre_state_objective_20261020`。

专用worker在**自己的进程内明确绑定**shared.build=build_state_model、shared.sample_cached=sample_scores，复用旧循环/损失/回灌/滑窗/原子检查点/恢复。这是新批次记录的执行依赖，不修改旧文件、不关闭或绕过任何哈希检查，不影响其他worker。旧launcher面对增加后的源码集合会按原规则拒绝，不使用force。

恢复先查真实PID/子进程/GPU/锁，确认没有活动协调器或CUDA训练，再用同一新入口。相同plan重新校验冻结并复用已完成阶段，从last完整恢复；epoch0、epoch40、best旁文件、未提交临时文件和失败3408归档沿用已验证路径。磁盘保留8GiB、项目26GiB/run1GiB，不删除旧数据或失败记录。

## 论文及其他材料约束

正式论文不设置失败路线/失败案例展示，探索记录留项目；必要对照按真实数字报告。旧Word/PDF是历史材料初稿，最终核心方法尚待有效证据。第3章侧重状态学习/修正机制和薄层质量，第4章RepELA侧重紧凑结构与精度—资源，第5章实际部署。后续图完整四边框，中文/西文数字分别沿用用户允许的NotoSerifSC/Tinos替代，旧图未重绘。

PhysicalNet完全只读。新开发仅MoS₂ train/val，旧test已暴露。WS₂来源/同时间戳分组、石墨烯3类/无独立test、外部MoS₂类别依据仍待确认，不进入本批。

此前静态语义批次已完成并停止：S0 val83.1345%，S1 82.6806%，S2 82.3998/82.4174%（同best两采样），未过筛选。此前13次完成训练是11次scratch+2次F0适配，本批两次未完成前不能写成15次已完成，也不能把适配当相同预算scratch。

## 可复制给下一对话

> 接手/root/autodl-tmp/ConditionalDiffusionSeg，先读docs/CHAT_HANDOFF_STATE_OBJECTIVE_20261020.md、HANDOFF最新48节及outputs/launches/p1_state_objective_20261020/{README.md,plan.json,status.json}，核查真实GPU/PID/子进程/日志/CSV/last，不重复启动。用户已授权P0 tanh有界控制和P1类别概率一致候选两次新scratch40seed42，batch2/512/各2400更新4800曝光，P0→同best3408→P1→同best3408→报告停止。原S0/S1/S2复用；无test、其他材料、200轮或模块队列。新50源/配置和专用workflow、412历史文件冻结，不能改写或绕过。专用worker显式进程内绑定新工厂/采样器复用旧训练loop；只用新launcher恢复。教师/训练MSE/DDIM均用对应bounded x0，最终两组raw logits一致；新旧S2比较包含读出改变，P1vsP0才固定读出。68CPU测试、512CUDA恢复/缓存、真实吞吐及整图检查通过；零步诊断已完成，不重做。按实际状态判断训练是否完成，结束后审计全部指标与预设门槛，保留全部结果并停止，不自动扩展。正式论文不展示探索失败案例，后续图完整边框，章节3机制/过程与章节4结构/资源明确分工。
