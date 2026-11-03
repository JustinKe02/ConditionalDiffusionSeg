# 条件掩码扩散四组串行训练计划

计划日期为2026-11-03。用户已要求完整执行ordinal、onehot、permuted、direct四组训练，在24 GB GPU上合理使用显存，并在开训前将本次代码与配置推送到现有GitHub origin。本计划已按真实GPU训练与原尺寸DDIM20实测锁定参数，四组在推送核验完成后依次执行。

## 1. 批次范围与完成条件

- 项目根目录为 `/root/autodl-tmp/ConditionalDiffusionSeg`。
- 顺序固定为ordinal → onehot → permuted → direct，每次仅一个训练进程占用GPU。
- 四组均从当前新实现的初始化规则开始，训练seed42，验证采样seed3407。
- 每组目标3000次优化更新，整批目标12000次。该默认沿用历史GLP的更新规模，四组内部使用相同真实batch。
- 使用train120与val25，主计划在既有训练和验证划分内完成。
- 每组达到完整更新预算并保存best、最终权重、训练历史及验证结果后进入下一组。
- 整批完成条件为四组均成功结束，结果表与执行记录齐全。中途指标排名仅用于观察。

## 2. 四组方法与共同设置

| 组别 | 状态表示 | 训练与验证路径 | 主要比较 |
|---|---|---|---|
| ordinal | 三通道层厚累计阈值 | RGB条件高斯掩码扩散，四类头回映累计期望 | 候选方法 |
| onehot | 四通道有符号onehot | 相同高斯扩散日程、损失政策与采样步数 | 名义类别表示 |
| permuted | 同一三通道码本、固定类别顺序置换 | 映射写入配置，最终仍输出原始四类语义 | 正确层厚顺序的作用 |
| direct | 三通道ordinal辅助目标，状态与时间输入置零 | RGB直接预测，保留同样的有序stateMSE及全局先验监督 | 有序辅助监督与扩散链的作用 |

共同网络采用强GLP全局—局部RGB条件。扩散三组的训练时间在1000个离散噪声时刻上均匀抽样，采用cosine噪声日程。验证从纯高斯状态开始，默认DDIM20、eta0。direct每窗口执行一次预测。

四组统一512×512裁块、整图一致的gain/gamma增强及旋转翻转，每图每轮2次裁块曝光，薄层中心采样比例0.5。全分辨率验证使用512窗口、384步幅和相同概率融合规则。图像条件及滑窗坐标遵循当前对齐实现。

## 3. 优化器、损失与预算

| 参数 | 冻结值或确定规则 |
|---|---|
| 优化器 | AdamW，weight_decay=1e-4 |
| 峰值学习率 | 2e-4，实际每轮学习率写入history |
| 最小学习率 | 1e-6 |
| 学习率调度 | 前8轮线性warmup，共160更新，随后cosine衰减 |
| 梯度裁剪 | max_norm=1.0，非有限梯度触发失败 |
| 计算精度 | FP32，TF32关闭，EMA关闭，严格确定性设置 |
| CE权重 | 0.5 |
| 前景Dice权重 | 0.5 |
| stateMSE权重 | 1.0，对样本、状态通道与空间取均值 |
| 全局prior_loss权重 | 0.5，内部为0.5软标签CE与0.5前景Dice |
| 每轮训练曝光 | 120张×2裁块=240次 |
| 每组优化更新 | batch12，每轮20更新，150轮共3000更新 |
| 验证频率 | 每15轮即300更新一次，共10次，包含最终轮 |
| 选模 | val四类mIoU最高，固定相同时保持先前best |
| 运行存储预算 | 每组1 GiB，数据盘保留至少8 GiB |

实际训练batch固定为12，每轮240次裁块曝光形成20次优化更新。150轮共3000更新、36000次裁块曝光，整批四组共12000更新和144000次曝光。历史GLP使用batch8、100轮、3000更新和24000次曝光，新实验每组曝光增加50%。四组内部预算一致，跨历史版本同时披露更新数和曝光差。

## 4. 24 GB GPU实测与最终参数

训练profiling采用onehot真实前向、反向、优化器及全局条件路径。B12长测经过2个预热batch和16个计时batch，完成192裁块，用时34.5867秒，吞吐5.55127裁块/秒。B16在反向阶段OOM。

| 实测项目 | 结果 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090，物理显存24576 MiB，CUDA可用约24126 MiB |
| B4初测 | 4.961裁块/秒，peak reserved 8156 MiB |
| B8初测 | 5.232裁块/秒，peak reserved 17070 MiB |
| B12长测 | 5.55127裁块/秒，peak allocated 21448.54 MiB，peak reserved 23582 MiB |
| 验证window batch8 | 单张原尺寸图DDIM20实测21.53秒，peak reserved 6512 MiB |
| 验证window batch16 | 单张原尺寸图DDIM20实测20.34秒，peak reserved 12888 MiB |
| 验证window batch35 | OOM，正式配置采用16 |

allocated统计已分配张量，reserved包含分配器持有的张量与缓存。B12记录中的两项峰值相差约2133 MiB，末端CUDA free约245 MiB，缓存与系统空闲分别记录。此长测成功结果用于锁定当前共同训练batch，正式运行保持单训练进程并持续记录显存。

| 最终字段 | 冻结值 |
|---|---|
| train batch | 12 |
| epochs / 每轮更新 | 150 / 20 |
| warmup_epochs / val_every | 8 / 15 |
| window_batch_size | 16 |
| workers / cpu_threads | 8 / 4 |
| lr / min_lr | 2e-4 / 1e-6 |
| sampling_steps / sampling_seed | 20 / 3407 |

训练实测记录位于 `/root/autodl-tmp/ConditionalDiffusionSeg/outputs/analysis/diffusion_seg_20261103/profile_initial.json` 与 `profile_batch12_stress.json`。原尺寸推理耗时为profiling样本实测，完整验证时延由正式history记录。
## 5. Git推送与开训契约

1. 将按实测锁定的四份配置、训练计划及必要代码写入现有仓库。
2. 执行现有测试、真实GPU前向反向和原尺寸验证检查，保存明确的通过与失败记录。
3. 检查现有Git remote origin、分支及待提交差异，将本轮代码和配置提交到当前工作分支。
4. 推送到该origin，并核对远端分支commit与本地HEAD一致。
5. 冻结批次计划，记录Git commit、源文件哈希、配置哈希、数据划分与数据版本、profile结果及启动时间。
6. 上述契约完成后才启动后台串行协调器。正式run保存相同配置与provenance。

后台串行入口为 `/root/autodl-tmp/ConditionalDiffusionSeg/scripts/run_diffusion_seg.sh`，批次目录为 `/root/autodl-tmp/ConditionalDiffusionSeg/outputs/diffusion_seg_20261103`。协调器依次调用 `scripts/diffusion_seg.py` 的train入口，按ordinal、onehot、permuted、direct顺序执行并写入独立日志。每组目录、命令、PID、起止时间、退出码与最后已提交epoch写入批次执行记录。

## 6. 评价与记录

每组10次原尺寸val25评价。history保存epoch、global_step、实际lr、训练用时、各损失、梯度范数及验证明细。训练吞吐按实际完成裁块与训练时间计算，验证耗时单独统计。

汇总表同时列best与最终epoch的四类mIoU、单层IoU、少层IoU、薄层平均IoU、BF@2px、单层↔少层错误率和跨级错误率。现有实现中两项错误率的分母为全部原尺寸像素，跨级定义为原语义顺序下类别差绝对值≥2。保留逐图混淆矩阵和边界计数，以便复算。

同权重DDIM10/5步属于可选成本分析。先完成四组既定训练及20步主验证，再根据实际耗时单独决定其安排。主批次的完成记录以固定预算与主评价为准。

代码与配置均披露onehot4通道、ordinal/permuted3通道及对应参数差异。报告编码幅度、MSE归一化、训练曝光、实际更新、噪声采样和真实推理调用数。新实验使用无EMA权重，历史GLP的EMA协议在跨版本比较中单独列出。

## 7. 失败、恢复与停止条件

- OOM、非有限损失或梯度、数据/代码哈希改变、磁盘余量低于8 GiB、检查点写入失败时，当前进程失败退出，协调器暂停后续组。
- 协调任务检查日志、磁盘、GPU占用与最新有效检查点，修复允许恢复的运行问题后继续同一预算。
- 每个已提交epoch原子保存last.pt，内含model、optimizer、scheduler、RNG、history和best信息。恢复入口使用同一冻结配置及 `--resume`。
- 未提交epoch在恢复时重跑，日志同时保留失败尝试和已提交global_step，完成预算以检查点中的有效更新计数为准。
- 配置或来源与检查点不一致时恢复检查会拒绝。涉及训练语义的修改需形成新的可追溯版本与执行记录。
- 各组均运行到3000有效更新。中途低分、组间排名或暂未达到研究收益门槛均继续按完整计划执行。
- 四组成功完成后核验文件、指标和预算，生成最终比较。仍有失败组时保持明确的未完成状态并继续处理可恢复问题。

## 8. 数据事实与最终交付

train120内部的m169与m56存在解码RGB重复、mask相差929像素的历史记录，本批固定当前数据版本。val25已经参与开发，本批结论限定为这一固定划分和seed42下的受控比较。样品来源独立性和物理层厚标签依据由后续数据工作继续补齐。

最终交付包括已推送的Git commit、冻结四组配置、真实GPU profiling、后台执行记录、每组best/last及history、best与最终epoch比较表，以及完成状态和任何恢复记录。尚未获得的性能收益在最终报告中按实际结果判断。