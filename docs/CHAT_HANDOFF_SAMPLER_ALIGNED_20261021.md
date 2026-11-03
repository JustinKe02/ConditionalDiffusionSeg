# 五步区间对齐训练续接记录

用户在审阅两组、40轮筛选、条件续训至100轮及时间预算后，明确授权：“按照这个方案开始训练”。
用户明确跳过固定诊断。本次只做必要工程验证，然后直接启动。

启动时间：北京时间2026-10-21 00:40；协调器初始PID65424。
该PID仅是启动记录，当前状态需核对进程及状态文件。

## 权威文件与入口

- 冻结协议：`outputs/launches/p1_sampler_aligned_20261021/README.md`
- 最终冻结配置与哈希：`outputs/launches/p1_sampler_aligned_20261021/plan.json`
- 整批状态：`outputs/launches/p1_sampler_aligned_20261021/status.json`
- 协调器日志：`outputs/launches/p1_sampler_aligned_20261021/launcher.log`
- 训练日志：同目录 `train_N2_40.log`、`train_N3_40.log`，条件续训则有 `_100.log`
- 新run根目录：`outputs/P1/sampler_aligned/seed_42/`
- 执行入口：`work/launch_sampler_aligned_20261021.py`
- 新训练目标与训练器：`work/sampler_aligned_core.py`
- 必要工程检查：`work/verify_sampler_aligned_20261021.py`

上述路径均相对项目根 `/root/autodl-tmp/ConditionalDiffusionSeg`。
旧 `scripts/train.py` 不适用于该新工作流。

## 已确定的执行顺序

N2′40 → N3′40 → 第40轮全量验证筛选 →
若全部通过：N2′续至100 → N3′续至100 → N2′同best3408 → N3′同best3408 →
比较报告 → 停止。

若40轮筛选未通过，生成报告后停止，两组都不继续。

40轮门槛使用准确第40轮指标：

- N3′ mIoU >= 0.8393。
- N3′ 前景边界F1 >= 0.5356。
- N3′−N2′ mIoU >= 0.002。

各组从共同历史Direct兼容权重与seed42新增分支重新初始化，Adam全新；
不是旧N2/N3完成100轮后的微调。数据/增强/训练顺序与旧批匹配。
batch2、FP32、100轮学习率日程、warmup5轮、每20轮完整25图验证。
40轮暂停后恢复完整优化器、scheduler和RNG，不重启或压缩日程。

## 新目标

t在100/75/50/25/1等概率抽样，s对应75/50/25/1/0。
使用原100步前向核的精确有序乘积构造q(x_s|x_t,GT,I)。
目标为0.5CE+0.5前景Dice+1.0跳步后验KL，t=1的KL置零，
按完整batch均值归约。一次模型预测/反向，原五步采样保持不变。

## 验证及保护

工程检查已通过：独立double精度区间乘积参考、终端P、Bayes混合、
t1零KL、匹配初始化、主要分支梯度，以及CPU和真实CUDA512/B2的
模型/Adam/scheduler/全部RNG精确恢复。
结果在 `engineering_verification.json` 与同名log，工程权重单独留档且不用于效果训练。

旧src/scripts/configs、原N0—N3模型与日志保持原文件哈希。
新工作流、先验、条件元数据、25张val标签均冻结；阶段边界核验完整缓存与父批结果。
新JSON原子写入，恢复位置以last.pt决定，不依赖status.json能否解析。
既有CSV/TensorBoard在恢复前备份，未提交检查点临时文件留档。

## 恢复方式

先检查现有协调器/worker是否仍在运行，避免启动第二个队列。
确认没有存活协调器后，可在同一个diffusion环境重新启动原入口：

```bash
/root/miniconda3/envs/diffusion/bin/python -u work/launch_sampler_aligned_20261021.py
```

协调器持 `outputs/.experiments.lock`，冻结配置恢复，已经提交的epoch不重训。
如果因冻结哈希或数据变化失败，先核对变更原因，不能绕过断言。
不得在冻结后修改本批三个工作流文件、README或工程验证JSON。

## 范围

不追加固定诊断、采样步数矩阵、其他材料、test、额外训练seed、
新模块、边界损失或精度切换。未经新的任务指令，不在本队列结束后自动加实验。
本批100轮完成后的最终筛选继续采用原完整研究价值门槛，不能因40轮门槛通过就宣称研究目标已达成。
