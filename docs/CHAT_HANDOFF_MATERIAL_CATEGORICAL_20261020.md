> **实际开训更新：2026-10-20北京时间18:31:05启动N0。18:32核实已完成4/100轮、240更新，CUDA训练PID40200，协调器36089。冻结时间18:30:52，source57/workflow51/history501/条件metadata2全部匹配。共同trainbatch2、windowbatch16；N1/N2/N3尚未开始，已排队。实时状态优先于下方实施过程快照。**

# 材料条件类别扩散N0—N3：实际实施交接

最新用户授权：“开始启用训练”。本批已完成实现、145张RGB条件缓存、新CPU8项检查和真实512/B2 CUDA四组恢复/采样检查，目前协调器进行吞吐选型，随后自动冻结并依次训练。请先查实时状态，禁止重复启动。

项目/root/autodl-tmp/ConditionalDiffusionSeg，解释器/root/miniconda3/envs/diffusion/bin/python；原数据PhysicalNet只读。没有test、多训练seed、其他材料或条件后续队列。

## 当前入口

批次outputs/launches/p1_material_categorical_20261020；协调器启动PID36089，live以launcher.pid、status.json、launcher.log、ps/GPU为准。入口work/launch_material_categorical_20261020.py，worker为work/material_categorical_worker.py。持共享outputs/.experiments.lock。

精确顺序：真实trainbatch2/4/8(10预热+30计时)及windowbatch1/4/8/16→全四组共同设置→冷条件流水线测速→plan.json冻结→N0新增100轮→N1新增100→N2新增100→N2同best3408→N3新增100→N3同best3408→比较报告停止。无自动N4、公开方法、步数矩阵或跨材料。

## 已实现方法

仅新增src/conditional_diffusion/material_conditions.py、material_categorical.py、material_experiment.py，旧50source/config保持原样。N0直接，N1同条件确定性五步，N2全局类别关系扩散，N3可靠度调节的局部光学类别关系扩散。新网络参数4,931,460/5,941,460，N1/N2/N3初始化逐位一致；所有组共同RGB/decoder/head从历史Directbest加载，8→32条件投影零初始化。全部可学习路径正常更新，非冻结F0残差适配。

共同固定Direct=4,929,156参数，历史训练40轮(best30)，不重新训练。完整推理系统参数N0=9,860,616，N1/N2/N3=10,870,616；共享历史40与新增100预算分别报告，不写100scratch。

## 条件与类别链

conditions/保存145张train/val图的Direct整图融合P float16，5,707,589,760字节约5.3GiB；逐文件哈希index.json；先验及基底场仅用RGB，不读val标注；train标签仅用于统计。全图基底由predBG≥.95、非饱和点的线性场稳健IRLS拟合，低支持r=0，C=(B−I)/(B+eps)，相对RGB而非校准反射谱。背景条件先全图生成再按512切窗。

类别统计train每8像素采样，四类Gaussian均值/收缩协方差，固定全局关系与局部光学似然关系；不按背景频率加权。完整定义见批次README。

Q_t=(1−beta)I+beta[lambda R+(1−lambda)1P^T]，T100,beta=1/(101−t),lambda=1−t/100；Q100每行P。N2全局R，N3用r混合local/global。真实类别链：矩阵有序乘积，精确sum_y p_theta(y)q(x_s|x_t,y)Bayes混合后验，不是高斯DDIM或软标签直接代入Bayes。训练t均匀1..100，无旧高噪声/回灌。五次预测时间100/75/50/25/1，全图每阶段私有uniform场由imageID+seed固定，窗口共享，RGB每窗口编码一次。末步输出干净类概率融合。

N0/N1 loss=.5CE+.5前景Dice；N2/N3加1.0posteriorKL。保留严格确定性；CUDA预检曾遇默认CE mean和floatcumsum非确定实现，改逐像素CE再mean、四类显式CDF三个阈值。两次失败及源/日志在preflight_revisions保留。最终CPU8passed，四组CUDA三步工程更新后的model/Adam/scheduler/RNG恢复及重复采样逐位一致；工程权重不用于效果训练。

## 训练与指标

各100新epoch、seed42、原120train记录/119独立RGB，512crop，同一裁块增强顺序，12000新曝光。共同batch选后每组更新数为100*ceil(120/batch)，若2则6000。FP32，AdamW2e−4→1e−6、warmup5cosine、wd1e−4、clip1。每20完整25val，四类globalmIoU，五步3407best；N2/N3同best3408，不重选，不是独立训练。原test27已有暴露史且本批不读。

门槛N3−N0≥1pp、vsN1/N2各≥.5pp、薄层mean≥1pp，单层/少层/前景边界各下降≤.5pp，secondary正方向。四组完成报告停止，不按结果改门槛/延长/堆模块。

评估seconds含cacheIO/融合/指标；固定先验已预计算，prior_reference_seconds只列参考，不假称本次又执行。benchmark_cold真实计原RGB读取+先验前向+条件准备+新模型融合，无指标；权重已加载。工程时间不是最终best效果结论。静态计算记录outputs/analysis/material_categorical_cost_20261020/costs.json基于已核实旧模块计算加新投影/类别矩阵，所有先验计入，MAC不叫FLOP。

## 完整性与恢复

启动前snapshot501历史文件+50源码，backupwork/backups/pre_material_categorical_20261020。冻结后将是新增3模块+4config，全部source57、工作流、条件元数据/先验以及145cacheSHA受保护。epoch0last在优化前落盘；best携带矩阵修复sidecar；epochlast含优化器、scheduler、完整RNG；恢复保留CSV/TB备份/未提交临时文件。每epochsource/先验/条件metadata，阶段边界fullcache/trainRGBmask/valRGB/history。磁盘8GiB保留、project26GiB、run1GiB。

最新未完成项目：N0余下轮次、N1/N2/N3各100轮、N2/N3同best3408及最终筛选报告。吞吐与冻结已完成，第一轮完整val尚未产出。任何新对话先看实时文件，不能因这里的未完成列表重跑。

论文第3章材料判别机制，第4章RepELA精度资源取舍，第5章真实设备；正式稿不放探索失败路线，但原研究记录都保留。不能预写第3章精度优势或故意选慢扩散衬托第4章。图完整边框，中文NotoSerifSC/西文数学Tinos为用户允许替代。新架构时间以实测为准。


## 实测与使用注意

训练B2 samples/s：N0=9.409、N1=2.045、N2=5.014、N3=4.934。N1的B4/B8 OOM，其余组较大batch也未更快；所有失败档位保留。窗口16为共同合格档中几何平均原图推理时间最短，N3等仍保持至少10%显存余量。单张训练图冷条件流水线工程计时约N0=6.10s、N1=8.49s、N2=15.44s，N3见benchmark_cold_N3.json；这些包含原RGB读取、先验和条件，不含指标，不是25图均值或最终效果延迟结论。

冻结后不要改新3模块、4configs、4工作流脚本、新批次README/预检/bench报告及条件metadata。正式训练只读这些文件，source57每epoch核对。ps命令行可能仍显示--window-batch1默认值，但train/secondary由--plan完全载入冻结config，实际windowbatch16；以run/config.json和plan为准。四个DataLoader子进程不是重复训练。

首次完整N0val在第20轮，结果前不能宣称N0/N3已达到新门槛。当前15个旧完成P1不包括这4个活动/排队新run。全部主批完成后停止，后续N4/公开方法/跨材料/步数分析仍未排队。
