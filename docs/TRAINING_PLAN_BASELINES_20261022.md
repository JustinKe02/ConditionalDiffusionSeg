# 第3章对比方法训练计划（2026-10-22）

状态：**计划，待执行。** 本文件只记录用户已确认的范围和协议，本身不启动任何训练。执行由用户在另一个对话中发起。

## 0. 用户已确认的决定

1. 对比方法共 9 个：U-Net、UNet++、PSPNet、DeepLabv3+、HRNet、SegFormer-B1、Mask2Former、DDP、CCDM。
2. 全部**从头训练**，不使用 ImageNet 或任何预训练权重，与 GLP-Flow 一致。
3. 依赖方案 a：**纯 PyTorch 实现**，不安装 torchvision、timm、mmseg、mmcv，不改动 `diffusion` 环境。
4. 训练只用种子 42，不做多种子、不做交叉验证。val25 保留 m109/m31（与第4章同一划分）。
5. 分析和对比不纳入 DINO 路线（包括光学/区域 A–D）。
6. 正式稿不放失败或局限性案例图；数值上的不利结果在表格中如实报告。

## 1. 目的与定位

- 为第3章补齐"与现有方法对比"（表3-5、图3-6）。GLP-Flow 自身的消融（E0–E3R）已完成，见 HANDOFF §59–§62。
- 对比方法是**参照**，不设筛选门槛。每个方法训练完、评估完即记录，不因结果好坏追加训练、改超参或换结构。
- 所有方法都在同一框架、同一协议下按原论文结构**复现**，不是官方代码。正文统一写"在同一训练框架内按原论文结构复现，未使用预训练"。

## 2. 统一协议（与 GLP-Flow E0–E3R 相同，除非方法本身要求不同）

| 项目 | 设置 | 来源 |
|---|---|---|
| 数据 | `data_manifests/source_dataset.json`，train120/val25/test27，原图 2560×1922 | 同 GLP-Flow |
| 训练裁块 | 512×512，每图每轮 2 个裁块，50% 裁块以薄层像素为中心 | `glpflow_core.GLPDataset`（train=True） |
| 数据增强 | 光度（gain 0.10、gamma 0.10、vignette 0.15）、90° 旋转、水平翻转 | 同上，按 `[seed, epoch, index]` 确定性生成 |
| 轮数与更新 | 100 轮，batch 8，每轮 30 次更新，共 3000 次更新、24000 个裁块 | 同 GLP-Flow |
| 优化器 | AdamW，lr 2.8e-4，weight decay 1e-4，warmup 5 轮，余弦衰减到 1e-6，梯度裁剪 1.0 | 同 GLP-Flow |
| EMA | power-EMA，σ_rel=0.10，验证和测试都用 EMA 权重 | `glpflow_core.PowerEMA` |
| 精度 | FP32，TF32 关闭，确定性算法开启（见第5节） | 同 GLP-Flow |
| 验证 | 每 10 轮和第 100 轮，在 val25 全分辨率图上评估；512 窗口、384 步长、重叠窗口概率平均 | `glpflow_core.positions` 与相同融合方式 |
| 选模 | val25 四类全局 mIoU 最高的一轮为 best | 同 GLP-Flow |
| 扩散方法采样 | 采样种子 3407 用于选模；同一 best 再用 3408 复核一次，不重新选模 | 同 GLP-Flow |
| 测试 | 全部训练完成后，用各自 val best 在 test27 上评估一次 | test 历史已暴露，不是新的盲测 |
| 指标 | 与 GLP-Flow 完全相同：`G.classification_metrics` + `summarize_boundaries`（边界容差 2 像素），另加到边界距离分层准确率 | `glpflow_core`、`analyze_glpflow_e3r_20261022.strata` |

损失函数：
- 非扩散方法默认 **0.5 CE + 0.5 前景 Dice**（与 GLP-Flow 语义损失相同）。CE 用 `reduction='none'` 再 `.mean()`，避免 CUDA 上均值 CE 的不确定性。
- 原论文规定的辅助损失照用：PSPNet 的辅助头（权重 0.4）、Mask2Former 的匹配损失、DDP 与 CCDM 的扩散目标。
- 不额外加入类别权重、边界损失或难例采样。

## 3. 各方法规格

模型规模按原论文常用配置，不刻意对齐 GLP-Flow 的 18.2M 参数，表中如实列出参数量。实现完成后，每个模型的参数量要与原论文或官方配置报告的数字核对（4 类输出头会带来少量差异，需注明）。

| 方法 | 出处（写入前核实） | 本计划采用的配置 | 归一化 | 预计参数量 | 预计训练时长 |
|---|---|---|---|---|---|
| U-Net | Ronneberger 等，MICCAI 2015 | 原版 5 级，通道 64–1024，padding 卷积，转置卷积上采样 | BN（原文无 BN，现代复现通用，需注明） | 约 31M | 约 1 h |
| UNet++ | Zhou 等，IEEE TMI 2020 | 原文 VGG 式 backbone，通道 32–512，嵌套稠密跳连；**不用**深监督剪枝，只取最终输出 | BN | 约 9M | 约 1–1.5 h |
| PSPNet | Zhao 等，CVPR 2017 | ResNet-50 膨胀主干，输出步长 8；金字塔池化 bins 1/2/3/6；stage3 辅助头，损失权重 0.4 | BN | 约 49M | 约 1.5 h |
| DeepLabv3+ | Chen 等，ECCV 2018 | ResNet-50 主干，输出步长 16；ASPP 膨胀率 6/12/18 + 全局池化；48 通道低层特征解码器 | BN | 约 40M | 约 1–1.5 h |
| HRNet | Wang 等，TPAMI 2021 | HRNetV2-W48，四分辨率拼接后接 1×1 分割头 | BN | 约 65M | 约 1.5–2 h |
| SegFormer | Xie 等，NeurIPS 2021 | MiT-B1 编码器 + All-MLP 解码器（256 通道） | LayerNorm / BN（解码器） | 约 13.7M | 约 1–1.5 h |
| Mask2Former | Cheng 等，CVPR 2022 | ResNet-50 主干；6 层多尺度可变形注意力像素解码器；9 层掩码 Transformer 解码器；100 个查询；损失：类别 CE 2.0、掩码 BCE 5.0、Dice 5.0，12544 采样点；语义推理 = softmax(类别) × sigmoid(掩码) 求和 | BN（主干）/ LN | 约 44M | 约 2–3 h |
| DDP | Ji 等，ICCV 2023 | 图像编码器 + 轻量映射解码器；标签图编码后做连续高斯扩散，DDIM 采样。主干、编码维度、噪声日程、采样步数**按原文分割实验设置核实后冻结** | 按原文 | 待核实 | 约 2–3 h |
| CCDM | Zbinden 等，ICCV 2023 | 条件类别（多项式）扩散，图像编码条件 + 去噪网络，预测干净类别分布。T、采样步数、网络宽度**按原文核实后冻结** | 按原文 | 待核实 | 约 2–3 h |

合计约 16–22 小时（含每 10 轮验证和 test 评估）。

### 需要先核实的原文细节（实现前完成，记入 `sources.json`）

1. 9 个方法的作者、会议/期刊、年份。
2. DDP：分割实验的主干、标签图编码方式与维度、噪声日程、训练时间步分布、推理步数、是否用自对齐去噪。
3. CCDM：扩散步数 T、推理步数、去噪网络结构、条件方式、损失组成。
4. Mask2Former：查询数、损失权重、采样点数（本计划写的是原文默认值，需核对）。
5. HRNetV2-W48 与 MiT-B1 的各阶段通道、层数。

如果某个细节原文没有给出、只有官方代码里有，按官方代码默认值并注明出处；两者都没有时，在计划中写明所选值和理由，冻结后再训练。

## 4. 代码组织

- 只新增文件，不修改任何已有的 `src/`、`scripts/`、`configs/`、`work/glpflow_*.py` 或历史结果。
- 建议文件：
  - `work/baselines_20261022/models.py`：9 个模型，以及确定性双线性插值、确定性自适应池化等公用算子。
  - `work/baselines_20261022/diffusion.py`：DDP 与 CCDM 的训练目标和采样器。
  - `work/baselines_20261022/core.py`：训练循环、预测和评估。数据集、EMA、指标、`positions`、`atomic_json`、`StorageGuard`、`atomic_checkpoint` 直接复用 `glpflow_core` 和 `conditional_diffusion.runtime`。
  - `work/baselines_20261022/verify.py`：工程验证。
  - `work/launch_baselines_20261022.py`：协调器，负责串行训练、扩散方法 3408 复核、test 评估和汇总报告。
- 批次目录：`outputs/launches/p1_baselines_20261022/`（plan.json、status.json、各方法日志、engineering_verification.json、最终报告）。
- 运行目录：`outputs/P1/baselines/seed_42/p1_baselines_20261022_<method>/`，结构与 GLP-Flow 运行目录相同（checkpoints/best.pt、last.pt、metrics_*.csv、validation_epoch*.json、best_val_metrics.json、best_val_confusion.npz、tensorboard/）。
- test 输出：`outputs/inference/baselines_test_20261022/`，结构同 `glpflow_test_20261022`（masks/、color/、`<method>_confusion.npz`、test_metrics.json、test_plan.json）。
- 开训前 plan.json 冻结全部新代码、数据划分和 manifest 的 SHA256；每轮训练前校验，变化即停止。

## 5. 已知工程问题与预定处理办法

当前 `seed_all` 会调用 `torch.use_deterministic_algorithms(True)`。以下算子在 CUDA 上反向传播没有确定性实现，会直接报错：

| 算子 | 涉及方法 | 预定处理 |
|---|---|---|
| `F.interpolate(mode='bilinear')` 的反向 | DeepLabv3+、PSPNet、HRNet、SegFormer、Mask2Former、DDP | 用固定插值矩阵实现双线性缩放 `Y = A_h X A_wᵀ`（matmul，可确定），前向需与 `F.interpolate(align_corners=False)` 一致，最大绝对误差 < 1e-5 |
| `F.adaptive_avg_pool2d` 的反向（非全局） | PSPNet（bins 2/3/6） | 同样用固定池化矩阵实现；全局池化直接用 `mean` |
| 均值 CE（NLLLoss 均值） | 全部 | `reduction='none'` 再 `.mean()` |
| `grid_sample` 的反向 | Mask2Former（可变形注意力、点采样） | 先尝试确定性实现；若无法实现，**只对 Mask2Former** 改用 `use_deterministic_algorithms(True, warn_only=True)`，在报告中注明该方法训练不保证逐位可复现，并将断点恢复检查放宽为数值容差 |

其他问题：
- **显存**：batch 8、512、FP32 下 HRNet-W48、Mask2Former、DDP 可能超过 24 GiB。预定处理：保持有效 batch 8，改为 2×4 梯度累积（更新次数仍为 3000），在报告中注明 BN 的实际 batch 为 4。不得为此缩小模型或裁块。
- **磁盘**：项目当前约 14 GiB，上限 26 GiB，磁盘保留 8 GiB。last.pt 含模型、EMA 和 Adam 状态，约为参数量 × 16 字节：HRNet-W48 约 1.0 GiB，超过单 run 1 GiB 上限。预定处理：本批次单 run 上限设为 1.5 GiB，项目上限保持 26 GiB；9 个方法合计预计新增约 6–7 GiB。若剩余空间不足，在开训前停止并报告，不删除任何已有文件。
- **扩散方法的推理代价**：在报告中记录 NFE 和纯预测耗时（与 §62 同一口径：只计 `predict`，含 CPU 窗口融合，不含读图和指标）。

## 6. 工程验证（全部通过后才开始正式训练）

1. 每个模型：CPU 实例化，参数量与第3节核对；512×512 输入的前向输出形状为 4×512×512。
2. 确定性算子：自定义双线性和池化与 PyTorch 原算子前向误差 < 1e-5；梯度数值检查通过。
3. 每个模型：真实数据 batch 8 训练 10 步（3 步预热 + 7 步计时），记录样本/秒、峰值显存，损失有限。超显存时按第5节切换梯度累积并重测。
4. 断点恢复：任选 U-Net、Mask2Former、DDP、CCDM 各做一次"连续训练 2 轮"与"1 轮停止再恢复到 2 轮"（max_batches=3），模型、EMA、Adam、scheduler、RNG 逐位一致（Mask2Former 若用 warn_only，改为数值容差）。
5. 每个模型：固定 4 个训练裁块过拟合 300 次更新，训练集 mIoU ≥ 0.95（扩散方法用 3407/3408 两个采样种子）。不通过则停止并报告，不调参硬过。
6. 推理接口不接收 GT；扩散方法同一图像、同一采样种子重复推理结果一致。
7. 工程验证产生的权重全部丢弃，正式训练从头开始。

结果写入 `outputs/launches/p1_baselines_20261022/engineering_verification.json`。

## 7. 执行顺序

1. 第3节"需要先核实的原文细节"。
2. 实现与第6节工程验证。
3. 冻结 plan.json，取 `outputs/.experiments.lock` 独占锁，后台启动协调器。
4. 串行训练（先快后慢，便于尽早发现问题）：
   U-Net → UNet++ → DeepLabv3+ → PSPNet → SegFormer-B1 → HRNet-W48 → Mask2Former → DDP → CCDM。
5. DDP、CCDM 各自 best 用采样种子 3408 复核一次 val25。
6. 9 个方法各用 val best 在 test27 评估一次，保存掩码、彩色图和混淆矩阵；保存掩码与混淆矩阵逐图核对一致。
7. 生成汇总：`comparison_summary.json`、`final_analysis.md`，表中同时列出 GLP-Flow E0、E1、E3R 的 val 和 test 结果（直接读取已有文件，不重跑）。
8. 写 HANDOFF 新节，停止。

每完成一个方法，在 status.json 中更新，并在对话中报告 val best。

## 8. 汇总表格式（表3-5 的数据来源）

每个方法一行，列：参数量、val best 轮、val mIoU、val 单层/少层/多层 IoU、val 前景边界 F1、test mIoU、test 单层/少层/多层 IoU、test 前景边界 F1、到边界 0–2 像素准确率（val/test）、NFE、每图纯预测耗时、峰值显存。

## 9. 不做的事

- 不使用预训练权重，不安装新依赖，不改动 `diffusion` 环境。
- 不做多训练种子、交叉验证、超参搜索；不因结果好坏追加轮数或修改设置。
- 不根据 test 结果改选权重或方法配置。
- 不修改已有代码、配置、权重、日志或结果；PhysicalNet 原数据只读。
- 不纳入 DINO 路线结果。
- 任何一步失败：保留日志，停止并报告，不自动绕过冻结校验。

## 10. 恢复

确认没有活跃的协调器或训练进程后，重新运行同一入口 `work/launch_baselines_20261022.py`。已完成的方法根据运行目录的 status.json 跳过；未完成的从 last.pt 恢复，要求配置和 plan 哈希一致。

## 11. 修订（2026-10-23，用户批准）

执行中复核代码时发现两处与本计划不一致，用户批准"停在两方法之间修正后继续"：

1. **Mask2Former 损失**：v1 代码为 `0.5CE+0.5Dice（语义 logits）+ 0.2 × 各层平均 Mask2Former 损失`，GT 先降到 1/4 分辨率，分类 CE 按 B·Q 平均——与第 2 节"Mask2Former 的匹配损失照用"和 `sources.json` 不符。修订为原文形式：初始预测与 9 层解码器共 10 组预测各自 `2·CE（no-object 权重 0.1，加权平均）+ 5·点采样 BCE + 5·点采样 Dice` 求和，全分辨率 GT 点采样，不加额外语义损失。优化器仍按统一协议（与原文 lr 1e-4、wd 0.05、clip 0.01、160k 次迭代不同，已写入 `sources.json`）。
2. **DDP / CCDM 梯度累积**：v1 工程验证中途崩溃后续跑时跳过了第 3 项，进程内的累积设置停留在 1，`grad_accum_selected` 因此全部记为 1 并冻结进 plan.json；而第 3 项实测 DDP、CCDM 在 batch 8 时超显存（或 >22.5 GiB）、已选 2×4。修订为 `grad_accum_steps=2`（有效 batch 8、3000 次更新不变；DDP 主干 BN 实际 batch 4，CCDM 只用 GroupNorm）。`verify.py` 改为从第 3 项实测结果读取累积设置。
3. 附带修正：汇总表"推理峰值显存"改取 test27 阶段的值（训练内验证测到的是训练遗留的显存缓存，与训练峰值相同）；报告文字中的滑窗步长改为 384；验证结果额外记录 `peak_allocated_mib`。

处理方式：U-Net、UNet++、DeepLabv3+、PSPNet、SegFormer-B1 已在 v1 下完成，所用代码不受上述修改影响，不重训。旧计划与旧验证报告保留为 `plan_v1_20261022.json`、`engineering_verification_v1_20261022.json`；替换前后各文件 SHA256 见 `outputs/launches/p1_baselines_20261022/revision_20261023.json`。只对 Mask2Former、DDP、CCDM 重跑工程验证第 1、3、4、5、6 项（`verify.py --methods`，日志 `verify_rerun_20261023.log`），通过后重新冻结 plan.json（含 `supersedes` 字段）并从 HRNet-W48 继续。若 Mask2Former 在原文损失下未通过固定裁块过拟合门槛，按第 9 节停止并报告，不调参。
