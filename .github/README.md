# ConditionalDiffusionSeg

MoS₂光学图像层数分割研究代码：早期条件扩散、全局—局部条件网络（GLP-Flow）、采样轨迹对齐E3R，以及R/I/N/F受控消融。

这是2026年11月3日从AutoDL整理的源码与轻量实验记录快照。最新I/N/F批次已于2026年10月31日完成；旧文档中的“运行中”是历史记录。

## 当前代码入口

| 位置 | 用途 |
| --- | --- |
| [work/glpflow_core.py](../work/glpflow_core.py) | 全局—局部模型、数据流、EMA、滑窗推理 |
| [work/glpflow_rollout_20261022.py](../work/glpflow_rollout_20261022.py) | 历史E3R采样对齐训练 |
| [work/chapter_controlled_20261031/](../work/chapter_controlled_20261031/) | 最新R/I/N/F训练、验证、分析、测速 |
| [work/baselines_20261022/](../work/baselines_20261022/) | 外部方法适配实现 |
| [work/repela_20261024/](../work/repela_20261024/) | RepELA对照实现及模型源码副本 |
| [src/conditional_diffusion/](../src/conditional_diffusion/) | 历史直接/迭代/条件扩散平台和共享工具 |
| [scripts/](../scripts/) | 历史平台命令行入口；不是最新GLP的统一入口 |

## 收录范围

包含源码、训练配置、split清单、顶层研究文档，以及明确选取的轻量方案、环境和结果JSON。最新controlled模块导入时需要的E3R `plan.json` 保留在原目录。

不包含原始图像与mask、模型权重、NPZ计数、预测图、TensorBoard日志、预训练模型、论文Word/PDF、打包产物和备份。指向同级DINOOpticalSeg的软链接不收录；DINO路线属于独立项目。

`outputs/`默认被原有.gitignore忽略，只有本次明确选取的轻量元数据被跟踪。文件列表和原始SHA256见[快照清单](source-snapshot.json)。冻结源码、配置、根目录README及历史实验记录未改写。

## 环境与基础检查

原运行环境为Linux、Python 3.10.22、PyTorch 2.5.1+cu124、RTX 3090；记录的NumPy为2.1.3、Pillow为11.3.0、TensorBoard为2.18.0、Matplotlib为3.9.2、SciPy为1.15.3。

历史固定依赖文件中的Pillow版本是11.0.0，与现有环境不同；历史文件保留原样。最新分析额外使用SciPy，安装时需补充。以下命令用于建立主要运行依赖，不自动启动实验：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==2.1.3 Pillow==11.3.0 tensorboard==2.18.0 matplotlib==3.9.2 scipy==1.15.3
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src python -m unittest discover -s tests -v
```

2026年11月3日原服务器上68项单元测试通过。它们主要覆盖历史src平台；最新路线的CUDA等价性、恢复和实际数据检查位于独立verify脚本，运行这些脚本需要原数据和实验资产。不要批量import所有work脚本：部分历史脚本在顶层读取权重或执行分析。

## 数据与历史实验复现边界

- 四类为背景、单层、少层、多层；原划分train120/val25/test27。
- 数据清单指向`/root/autodl-tmp/PhysicalNet/Mos2_data`，split和部分历史计划使用原服务器绝对路径。
- R复用历史E3R权重；controlled的冻结计划还校验历史best权重、真实数据和验证计数。这些资产未上传，因此新克隆不能直接完整重放原协调器。
- 跨目录或跨机器使用需另外提供数据与权重，并建立新的路径配置和实验计划；不要悄悄改写历史冻结计划的含义。
- Chapter 4历史协调器还依赖同级PhysicalNet和多模型实验资产。
- 训练集m169/m56为相同RGB但mask有929像素差异；样品/母图来源独立性仍待核验。历史test不是新的盲测。

## 最新验证结果

同一val25、四类mIoU选best、训练seed42：

| 组别 | best epoch | mIoU/% | 前景BF1@2px/% | 薄层平均IoU/% |
| --- | ---: | ---: | ---: | ---: |
| R/E3R | 60 | 88.5566 | 74.0857 | 78.0081 |
| I | 100 | 89.9195 | 66.5814 | 80.9928 |
| N | 50 | 88.6189 | 73.6063 | 78.1744 |
| F | 50 | 88.1472 | 72.0962 | 77.3132 |

R对I改善边界但降低区域指标，综合门槛未通过。N/F没有随机初态；N使用Euler混合，F直接回馈预测端点。I/N/F尚无test结果。不能把R的历史test结果转移给这些组，也不能把单训练seed结果称为全面领先或统计等效。

主要记录：[最新分析](../outputs/analysis/chapter_controlled_ablation_20261031/final_report.txt)、[汇总JSON](../outputs/analysis/chapter_controlled_ablation_20261031/comparison_summary.json)、[完整状态说明](../docs/EXPERIMENT_STATUS_AND_WRITING_GUIDE_20261101.md)。历史上下文保留于[根目录README](../README.md)和[HANDOFF](../HANDOFF.md)。

本仓库初始设为私有，未额外授予开源许可证。
