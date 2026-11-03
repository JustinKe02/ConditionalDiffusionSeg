# TensorBoard与论文绘图记录规范

状态：已接入首版训练器并生成调试事件日志；正式实验尚未开始。实现与验收范围见RUNBOOK.md和HANDOFF.md。

## 1. 每次运行独立保存

所有普通、迭代、扩散与消融实验统一启用TensorBoard。输出布局为：

```text
outputs/<phase>/<method_or_experiment>/seed_<seed>/<run_id>/
    tensorboard/
    metrics_steps.csv
    metrics_epochs.csv
    metrics_val.csv
    config.json
    run_metadata.json
    checkpoints/
    predictions/
```

run_id区分相同配置的独立重跑；断点恢复才沿用原run。有效配置而非仅默认值写入config.json；记录数据划分哈希、类别映射与颜色、训练seed、采样seed、模型参数量、软件与GPU信息、代码版本或源码哈希，以及恢复checkpoint标识。超参数可在TensorBoard文本/超参数视图展示，并始终保留JSON原件。

## 2. 标量标签、频率与含义

| 标签前缀/内容 | 频率与横轴 | 说明 |
|---|---|---|
| train_step/loss_total、loss_ce、loss_dice_fg | 每10个optimizer step，横轴global_step | 记录本步原始值；各分项记录未乘权重值，总损失按配置加权 |
| train_step/loss_x0_mse、loss_boundary | 同上，适用时 | 非扩散/无边界实验不填虚构的0；CSV不适用项留空 |
| train_step/lr_group_*、grad_norm_pre_clip | 同上 | 学习率记录各参数组；梯度范数记录裁剪前值 |
| train_epoch/loss_* | 每轮，横轴完成的epoch数，从1开始 | 对该轮全部batch按样本数加权汇总，不只平均每10步抽样日志 |
| val/miou_4class、miou_foreground | 每次实际验证，横轴epoch | 基于全验证集累计混淆矩阵；四类mIoU包含背景 |
| val/iou_<class>、f1_<class> | 同上 | 固定background/monolayer/fewlayer/multilayer命名 |
| val/boundary_* | 同上，边界指标定义冻结后 | 保存指标定义、容差与缺失类别处理规则 |
| optim/epoch_lr_group_* | 每轮，横轴epoch | 补充学习率随轮数变化曲线 |
| resources/train_seconds、val/seconds | 对应阶段结束，横轴epoch | 区分训练与完整滑窗验证耗时 |
| resources/train_samples_per_second | 每轮，横轴epoch | 定义为处理训练样本数/训练阶段耗时 |
| resources/train_peak_*、val/peak_* | 对应阶段结束，横轴epoch | GPU allocated/reserved峰值，统一MiB；阶段开始重置峰值计数 |
| sampling/steps、samples_per_image | 扩散每次验证，横轴epoch | 记录真实采样设置；采样seed另存元数据 |

CSV标量字段包括epoch、global_step、wall_time_unix、tag、value；run_id由独立运行目录标识，跨run汇总时应添加该列；TensorBoard与CSV使用同一个已计算的指标来源。IoU/F1原始数值统一0—1，绘图若显示百分比明确乘100。不同方法总损失定义不同，不直接根据总损失大小判断方法优劣。

验证指标必须来自完整的实际推理：扩散从独立高斯噪声采样，不用GT带噪输入的单步去噪分割准确率替代。默认每10轮及最后一轮验证，因此验证曲线只写这些真实观测点，不补零或伪造中间轮数。test不进入训练过程监控，只在最终协议冻结后单独保存结果。

## 3. 固定样本可视化

正式训练前从val固定选择4张图并保存ID，覆盖主要类别与典型边界；所有方法与轮次保持一致。每次验证展示原图、GT、预测与错误图，使用固定类别颜色，并保持图像ID和采样seed一致。GT仅用于展示和算分，不进入采样器。

TensorBoard预览最长边不超过512像素，类别/错误图缩放采用最近邻，避免显示伪类别。完整验证仍按原分辨率计算指标。默认只保留最佳checkpoint及最终评估所需的原分辨率整数PNG；固定预览不能代替全量数值评估。默认关闭全参数直方图、网络大图与逐步采样轨迹，控制日志体积。需要研究采样轨迹时仅对预先固定少量验证样本单独导出。

## 4. 中断恢复与可靠性

checkpoint保存global_step、epoch以及训练恢复状态；日志轴从checkpoint恢复，不重置到0。处理日志领先于checkpoint的情况：对后续记录进行清理/屏蔽后再继续，同时保存原日志备份，CSV采取对应处理，避免重复点混入论文数据。独立重跑创建新run_id。

每30秒定期flush，并在验证结束、保存checkpoint和退出时flush/close；异常退出路径尽可能关闭writer。日志依赖缺失应在启动预检查时明确报错，不静默关闭用户要求的记录功能。

## 5. 绘图产物与验收

后续绘图脚本优先读取CSV/JSON，输出PDF/SVG及高分辨率PNG：

1. 各训练损失分项与验证mIoU随epoch变化曲线。
2. 各类别IoU/F1、边界指标比较图。
3. 三个训练seed的均值与样本标准差，采用一致的epoch观测点。
4. 采样步数—精度—实际耗时图；推理研究使用独立eval记录，不混入训练曲线。
5. 固定样本分割对比与局部边界放大图。

原始数据不做平滑覆盖；展示平滑曲线时标注方法与窗口。训练seed方差和采样seed方差分别统计。最好checkpoint最终得分与逐轮曲线分开保存，不能将各轮测试最优结果拼成曲线。

训练器验收包含：短程训练能生成可读events文件；至少一个loss与验证指标在TensorBoard/CSV数值一致；不同run不会覆盖；断点恢复后的step连续且无重复污染；固定样本配色一致；关闭可选图像记录不改变数值评估。此时才可将状态标为“TensorBoard已接入”。

## 6. 50 GiB数据盘的空间约束

每run的TensorBoard预算256 MiB，CSV/JSON预算32 MiB；固定预览缩到最长边512。增加磁盘剩余GiB、当前run占用GiB、项目总占用GiB的监控。断点恢复日志备份同样计入预算，不能无限积累。达到预算需报告并停止继续扩张，保留原始标量与现有结果，不静默删除或覆盖历史run。详细估算与检查策略见[存储预算](STORAGE_BUDGET.md)。
