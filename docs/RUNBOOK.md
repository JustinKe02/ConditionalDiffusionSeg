# 运行手册

当前是首版研究代码。正式实验前仍需做小样本过拟合、完整验证集测速和协议复核；调试分数不用于论文。训练脚本一次只启动一个run，没有自动连续运行24个实验的队列。

## 1. 环境与数据

已有/root/miniconda3/bin/python环境包含PyTorch、NumPy、Pillow、TensorBoard和Matplotlib，可直接使用，不需要复制或重建conda环境。requirements.txt是最低依赖说明，不是已验证的所有未来版本组合。数据根仍为相邻PhysicalNet/Mos2_data；split采用本项目快照及SHA256检查。

```bash
cd /root/autodl-tmp/ConditionalDiffusionSeg
python scripts/check_storage.py
python scripts/audit_data.py
python -m unittest discover -s tests -v
```

数据审计会逐图验证尺寸、标签范围，输出data_manifests/full_audit.json；这不证明样品来源独立性。

## 2. 配置区别

- configs/experiment_plan.json：完整研究计划及任务登记，不能作为train.py输入。
- configs/debug.json：32×32、base8、两张训练图、一个验证裁块、每轮一个batch、两轮，用于连通性检查。
- configs/pilot.json：512×512、base32、batch2、40轮、warmup5，用于P1；验证为25张原图固定滑窗。
- configs/formal.json：相同网络，200轮、warmup10，用于通过P1后的正式实验。

模型输出四类，x0预测、cosine噪声日程；DDIM eta=0，时间索引从T−1到0等距取整，逐步clip到[-1,1]。默认T1000、采样20步、sampling_seed3407。类别权重为训练频率逆平方根、按四类均值归一化、clip到[0.25,4]；CE采用像素均值。Dice对batch与空间汇总后平均三个前景类，平滑1e−6，缺失类仍参与Dice。边界损失在3×3类别max/min不一致带上计算，权重0.1。指标边界F1采用每类3×3内边界，原图尺度Chebyshev距离2像素匹配，累计精确率/召回率计数后计算F1；无真边界且无预测边界时未定义，记null并从均值排除，一侧为空记0。该容差仍需在正式研究中做敏感性检查。

三种模型都从随机初始化训练；直接模型3通道输入，迭代/扩散7通道；每个尺度2个残差块、GroupNorm/SiLU，4个尺度。迭代模型固定5步，均匀概率初始化，最终步监督，梯度穿过全部迭代。训练每图每轮取一个随机裁块，增强随机性由seed/epoch/样本索引独立决定；三种方法同seed同epoch使用相同数据顺序与裁块。

## 3. 调试与资源检查

```bash
python scripts/train.py --config configs/debug.json --method direct
python scripts/train.py --config configs/debug.json --method iterative
python scripts/train.py --config configs/debug.json --method diffusion
python scripts/smoke_resources.py
```

这些命令会实际执行小规模运算，但不是正式训练。资源脚本对计划尺寸执行每种模型一次合成数据前向/反向/AdamW更新，记录参数量与显存；不能据此宣称收敛或完整验证速度。CPU使用需在新配置中设置device=cpu。

## 4. P1运行入口

确认P0通过之后，依次执行；不要在单张GPU上同时运行三组：

```bash
python scripts/train.py --config configs/pilot.json --method direct --seed 42
python scripts/train.py --config configs/pilot.json --method iterative --seed 42
python scripts/train.py --config configs/pilot.json --method diffusion --seed 42
```

每次自动生成独立run_id，程序打印绝对输出路径。可以指定--run-id，但已有目录会报错，防止覆盖。结束后检查status.json，不能仅凭文件夹存在判断完成。

E01/E02/E03使用formal.json与三个seed分别执行；扩展在后续有证据支持时加入--class-balance和/或--boundary。同一组比较须冻结同样的数据、裁块、预算和选模规则，不把debug与pilot混入正式表。

## 5. TensorBoard与绘图

```bash
tensorboard --logdir /root/autodl-tmp/ConditionalDiffusionSeg/outputs --host 127.0.0.1 --port 6006
```

服务器访问使用平台端口转发或SSH转发。step标量与epoch指标使用两个writer目录，避免恢复时不同横轴互相影响。分类别曲线、训练损失分项均已写入CSV与TensorBoard；图像每次验证记录固定前4张val，512最长边。原始标量不做平滑覆盖。

绘图示例（实际替换run路径；输出文件夹必须尚不存在）：

```bash
python scripts/plot_curves.py --runs /absolute/path/to/run --output /absolute/path/to/new_figure_folder
```

首版脚本导出总损失与mIoU的PDF/SVG/300dpi PNG；不同模型总损失定义不同，不能按总损失大小排名。三个seed均值±标准差、分项和采样成本图可基于CSV扩展，尚未提供完整论文绘图套件。

## 6. 恢复训练

```bash
python scripts/train.py --resume /absolute/path/to/run/checkpoints/last.pt
```

仅加载本项目可信本地checkpoint；torch.load包含Python训练状态，不用于未知来源文件。best.pt仅含权重与配置；last.pt含optimizer/scheduler/RNG/global_step/epoch，可从完成的epoch边界恢复。恢复不允许修改配置；source哈希另行记录，便于追踪代码变动。CSV与TensorBoard过期记录先备份，再按checkpoint边界裁剪/屏蔽；备份计入磁盘限额。

--stop-after-epoch N可在保留原总轮数与学习率计划的情况下有序停在某轮，便于调试恢复；它不会把计划总轮数改成N。发生磁盘异常时不冒险写大checkpoint，保留最后成功的last，最近尚未保存的进度可能损失。原子替换只能保证单文件完整；若best先于last写入而后中断，恢复会保留该已完成验证的best分数/权重，best自带来源epoch与指标。

## 7. 验证与采样研究

```bash
python scripts/evaluate.py --checkpoint /absolute/path/to/run/checkpoints/best.pt --split val --steps 10
```

--samples可设1/2/4，--sampling-seed控制初始噪声；同一图像的重叠窗口读取同一全图噪声场，逐图融合概率，默认不保存全量概率或轨迹。完整评估结果在run/evaluations/独立时间戳目录。--save-predictions额外保存单通道PNG，重复使用该参数会累积文件，需要遵守预算；默认只保存指标/混淆矩阵。

test必须使用非debug模型，并显式提供--allow-test与--protocol。protocol是验证集选定后保存的JSON，且必须恰含sampling_steps、samples_per_image、sampling_seed；不能再用命令行覆盖它。该接口促进固定设置，但不能替代研究者对测试集使用纪律与数据独立性的核查。

## 8. 空间保护的实际范围

当前训练代码检查启动预测容量、每轮、每次验证、日志写入、checkpoint保存及评估PNG写入；磁盘保留8 GiB、12 GiB告警、项目26 GiB上限、单run1 GiB上限、TensorBoard256 MiB。记录剩余磁盘、run与项目大小。checkpoint通过临时文件+fsync+原子替换保存，预留临时空间。

只读check_storage.py负责整套计划容量估算；train.py负责单run启动及运行期检查，不自动安排后续队列。外部进程可同时占盘，仍可能在检查后触发写入失败；已有checkpoint受到单文件原子保存保护。不要为了强行通过检查而直接调低保留空间。

## 9. 最新Conda环境：diffusion（2026-10-17更新）

用户要求删除的nnunet和segment环境已移除，不能再使用其原解释器。environment_nnunet_audit.json等文件仅作为历史验证记录保留。

新环境位于系统盘`/root/miniconda3/envs/diffusion`，Python3.10.22，可通过`conda activate diffusion`激活。Conda默认环境和包缓存目录也已切换到系统盘。当前新环境仅含Python、pip和基础依赖；尚需安装requirements.txt中的训练依赖并验证后，才能用于项目实验。前文base环境仍保留可用，不应将base的验证结论直接套用于新环境。

## 10. Bash实验入口（diffusion环境）

依赖安装：`bash scripts/setup_diffusion.sh`。固定PyTorch2.5.1+cu124、NumPy2.1.3、Pillow11.0.0、TensorBoard2.18.0、Matplotlib3.9.2，与此前base验证环境的主要版本一致；完整依赖列表每批另存pip-freeze。安装过程使用系统盘临时目录、不保留pip下载缓存。

从项目目录执行：

```bash
bash scripts/run_experiments.sh --stage all --batch-id diffusion_first
```

`all`先复核环境、单元测试、512/batch2显存与训练恢复，再执行P0和P1。`--stage p0`仅执行P0；`--stage pilot`要求同一个batch-id下已通过P0及完整验证集测速。相同batch-id会跳过已完成run，未完成run从last.pt恢复；配置/代码不一致或没有可用恢复checkpoint会停止，保留已有文件。全局flock防止本脚本启动多个批次争用GPU；它不能阻止其他用户或程序自行启动GPU进程。脚本不运行200轮正式实验或test评估。

P0使用4张不同训练图的固定256裁块，每类有明确覆盖，并保留边界；根据训练标签的固定网格、确定性评分选取，坐标及像素计数记录在`data_manifests/p0_fixed_crops.json`。不做增强、不使用验证或测试图。三组均为base32、batch2、300轮，每轮2次更新，共600次更新，固定lr2e-4。这是小样本记忆诊断，不是正式300轮实验。采用256裁块是为了降低诊断成本，512尺寸另经真实网络合成训练步检查。

P0每50轮从RGB（扩散为独立高斯噪声）推理，按固定训练裁块mIoU保存best。预设通过条件为四类mIoU≥0.90且每类IoU≥0.75；扩散best额外用sampling_seed3407和3408分别验证，两个种子都需通过。门槛不通过即停止，不自动降低门槛或追加搜索。训练裁块指标保存在metrics_val.csv中，使用独立的`overfit/`标量标签，run_metadata与P0报告明确其来自train；不得当作验证集或论文分数。

通过后用扩散P0权重测完整25张val、原图512窗口/384步长、20步DDIM、单次采样的耗时与内存，记录实际checkpoint/log大小。该权重仅作测速，不把它的val分数作为研究结果。benchmark_summary会校验25张和17,500次网络调用，并汇总4次P1扩散验证的实测耗时预算。

P1仍是512、batch2、40轮、seed42的direct/iterative/diffusion三组，串行执行，最终写`outputs/launches/<batch-id>/pilot_summary.json`。批次状态见同目录status.json，单run状态见各自status.json。长时运行建议将标准输出重定向到批次console.log；本次实际后台启动的PID与日志路径另记HANDOFF。控制台、TensorBoard和CSV均保留，磁盘保护仍为8 GiB预留。


## 11. Bounded P0 diffusion ablations (2026-10-17)

Run `python scripts/run_p0_ablations.py` with the diffusion interpreter. This entry point has a frozen two-experiment plan at `outputs/launches/p0_ablations_20261017/plan.json`. It runs only the following independent, from-scratch P0 diagnostics, serially under the existing GPU lock:

- `configs/p0_high_noise.json`: same fixed lr=2e-4 as original P0; each batch has one uniform draw from t=0..899 and one from t=900..999. High-noise probability changes from 10% to 50%; this deliberately changes timestep weighting, without importance reweighting.
- `configs/p0_cosine_lr.json`: original uniform t=0..999; only min_lr changes from 2e-4 to 1e-6, using the existing cosine scheduler.

Both retain seed42, four fixed train crops, base32, crop256, batch2, 300 epochs/600 updates, 20-step DDIM, and separate evaluations at seeds3407/3408. Both must be evaluated even if the first misses the gate. Completed diagnostics do not automatically start P1. The original failed batch remains preserved.

Preflight includes 13 unit tests, the existing uniform-training integration check, and a separate base8 high-noise training/resume equivalence check. Per-epoch CSV/TensorBoard now also record actual timestep mean and high-noise fraction. The uniform sampling default preserves the original randint operation and RNG behavior. This is a code/config revision, so the older batch cannot pass its frozen source-hash check under the new code.

Batch state and console are `outputs/launches/p0_ablations_20261017/status.json` and `console.log`; results are `partial_results.json` and `summary.json`. Each run keeps its own `p0_gate.json`, checkpoint and original training logs. Reusing the entry point requires matching frozen code/config and resumes only from a valid last checkpoint. No extra trials or threshold changes are automatically selected.


## 12. Registered high-noise P1 comparison

The user authorized one high-noise P1 trial and full-val sampling-seed3408 evaluations of the old and new diffusion models. Entry point: `/root/miniconda3/envs/diffusion/bin/python -u work/launch_p1_high_noise.py`. Batch `p1_high_noise_20261017` uses `configs/pilot_high_noise.json` and run `outputs/P1/diffusion/seed_42/p1_high_noise_20261017`.

This is a single timestep-coverage factor: every batch2 has one uniform t0..899 and one uniform t900..999. It deliberately reweights the timestep objective without importance correction. The P1 validator allows it only with `experiment_variant=high_noise_coverage`, diffusion method, phaseP1, debugfalse and no fixed training crops. Existing P0 and uniform configs still work. Model, losses, inference, RGB crops/order, seed42 initialization,512/batch2,40 epochs/2400 updates,warmup5 and cosine LR remain unchanged. The noise sequence also changes as a consequence of different timestep RNG calls.

Preflight checks16 unit tests and registered-P1 stratified resume equivalence on a small full-train-split integration run (base8/crop32, two controlled epochs of the40-epoch LR schedule, no val/test). Integration files are outside outputs/P1 so TensorBoard comparison is not polluted. The complete512 trial starts only after these pass. Storage reserve remains8GiB.

The baseline checkpoint is the immutable diffusion best.pt from p1_cosine_20261017 (epoch40). Source-before, frozen source/config and checkpoint SHA256 are recorded. Only trainer configuration validation and the new config differ from the prior validated source map; model/objective/diffusion/inference/data code are untouched. Current evaluation-source and old trained-source hashes are kept separately.

After training, the launcher evaluates each best checkpoint on all25 val images at seed3408,512/384,20 DDIM steps, one sample. Existing seed3407 validation still selects best checkpoints. Stage/progress: batch status.json, console.log, baseline_seed3408.log, candidate_seed3408.log. Final comparison_summary.json reports both seeds, deltas, measured high-noise exposure and the predeclared screening rule. Reusing the launcher requires the same frozen plan/source and resumes from last.pt; no extra training or formal/test stage follows completion. The current TensorBoard logdir outputs/P1 discovers this run automatically.

## 13. Registered mixed-recycling P1

User authorized the first proposed recycling experiment and baseline sampling-step measurements. Entry point: `/root/miniconda3/envs/diffusion/bin/python -u work/launch_p1_recycling.py`; batch/run ID `p1_recycling_20261017`, config `configs/pilot_recycling.json`.

Preflight checks the frozen environment, disk,22 unit tests and exact model/optimizer/scheduler/RNG resume equality through the registered P1 recycling path (full120 train split, base8/crop32, controlled stop at2 of40 epochs, no val/test). It then evaluates the existing high-noise best on full25-image val at1/5 DDIM steps and seeds3407/3408, reusing historical20-step results. This does not change the candidate's20-step checkpoint-selection protocol.

Exactly one from-scratch40-epoch/2400-update trial follows. Every batch2 retains one t0..899 and one t900..999. Independently, a random half of the batch uses recycled input: same-model terminal-time prediction from independent pure Gaussian noise, no gradients, clamped to[-1,1], then q_sample at the training time with independent noise. Other samples retain GT q_sample. Both branches have the original true-label losses; no pseudo-label supervision or teacher auxiliary loss. Architecture, loss coefficients, RGB crops/order and all other high-noise P1 settings stay fixed. Additional RNG calls alter subsequent timestep/noise assignments. Each update adds one no-gradient teacher sample forward; total2400, logged alongside actual recycling/high-noise fractions and time.

After training, candidate best is evaluated at20 steps/seed3408, then comparison_summary.json is written and the launcher stops. Baseline is the high-noise P1, with both it and the original uniform-P1 checkpoint hashes protected. Frozen training-source and historical baseline trained-source hashes are separate. No dual-head trial, class/loss change, extra training seed, formal run or test is included. Consult the batch README and plan.json for scope and recovery. First verify no launcher/train/evaluate process remains before rerunning the same launcher; never bypass source/config/hash guards.

## 14. Uniform-timestep recycling P1 and candidate step measurements

Authorized entry point: `/root/miniconda3/envs/diffusion/bin/python -u work/launch_p1_uniform_recycling.py`, batch/run `p1_uniform_recycling_20261017`. Config `configs/pilot_uniform_recycling.json` registers `experiment_variant=uniform_recycling`, paired strictly with `timestep_sampling=uniform` and mixed-recycling fraction0.5. Existing variants retain their policy checks; wrong policy/variant combinations are rejected. The recycling helper/model/loss/inference are unchanged. `high_noise_start=900` is a logging threshold only under uniform sampling.

Queue: environment/storage/tests and exact uniform-recycling model/optimizer/scheduler/RNG resume check; full25-val1/5-step evaluations of the completed high-noise+recycling best at3407/3408 (20-step results reused); one from-scratch512/batch2/base32/FP32/40epoch/2400update uniform+recycling P1; candidate20-step full-val3408; summary and stop. Best selection stays20-step/3407, every10epochs. No extra seed, architecture/loss change, formal matrix or test is queued.

The final comparison_summary.json completes the uniform/high-noise × standard/recycling2×2 table at both sampling seeds. It reports conditional effects, averaged effects and HR-H-R+U interaction descriptively, keeping historical trained/evaluation-source provenance. All three existing diffusion best weights and reused metric reports are hashed and protected. Preflight and actual trial record real recycling fraction and actual high-noise exposure; uniform coverage is expected near0.1, not forced0.5. Recovery uses the same frozen work launcher only after checking no active processes. Batch README has schedule and paths; disk reserve stays8GiB, onlybest/last kept, old outputs preserved.

## 15. Dual-head high-noise recycling P1

User authorized one dual-head ablation. Entry `/root/miniconda3/envs/diffusion/bin/python -u work/launch_p1_dual_head.py`, batch/run `p1_dual_head_20261018`, config `configs/pilot_dual_head.json`. Explicit P1-only dual_head registration retains all high-noise-recycling controls. Model forward returns a dict only for dual-head configs: logits for semantic loss and x0 for MSE. Existing single-head configs keep tensor outputs, state-dict layout and RNG initialization.

An independent deepcopy of the original output block adds196 parameters (total5132872). Initial backbone and both heads match the single-head initialization, without consuming extra RNG. CE/Dice gradients reach the semantic head, MSE reaches the denoising head, both reach shared features. Recycling and DDIM transition use clamped denoising output. At the last DDIM step, raw semantic logits provide classification/probability fusion; the legacy path still returns clipped x0. Interpret this as the complete dual-head design, including readout/calibration changes, not as a pure test of gradient competition alone.

Preflight includes full unit tests, exact old-checkpoint/source compatibility checks and registered dual-head training/resume equivalence (model/optimizer/scheduler/RNG, full120train/base8/crop32/two controlled epochs, no val/test). Then one from-scratch seed42/512/batch2/FP32/40epoch/2400update P1, unchanged high-noise50%/recycling50%,20-step3407 selection every10epochs; candidatebest3408 fullval; comparison and stop. All four prior diffusion weights and reused baseline reports are hashed and protected. No5-step sweep, other factor/seed, formal matrix or test follows. Read batch README/plan for recovery; do not restart if another process is active, and never bypass hashes.


## 16. Frozen P1 test evaluation (completed2026-10-18)

User-authorized evaluation-only batch `p1_test_20261018` compares validation-selected Direct/Iterative/best single-head Diffusion on all27 test images. Protocol and best weight/dataset/source hashes were frozen before test in outputs/launches/p1_test_20261018/{plan.json,protocol.json}. The unchanged scripts/evaluate.py uses --split test --allow-test --protocol <frozen-path> --save-predictions; no command-line sampling overrides. The coordinator work/launch_p1_test.py holds the shared GPU lock, preserves incomplete outputs and audits full prediction matrices, IDs, metric counts and hashes. Completed results are reused; incomplete or unregistered test exposures stop automatic reruns for review.

Final results: comparison_summary.json/comparison.csv/per_image_metrics.csv/final_analysis.md in the batch directory. Native prediction PNGs, raw metrics/config and confusion matrices stay under each run/evaluations/test_*. status.json is completed/finished; no further evaluation or training is queued. Test is now exposed and must not be used for model/step/seed tuning. These remain40-epochP1, single-training-seed results with sample independence unresolved.


## 17. Registered reference-fusion F0/F1 P1 (2026-10-18)

User-authorized bounded coordinator: `/root/miniconda3/envs/diffusion/bin/python -u work/launch_p1_reference_fusion.py`; batch outputs/launches/p1_reference_fusion_20261018. RGB-only estimator/audit and cache frozen before pilots. F0/F1 use identical split RGB/state encoder with ordinarymeanfusion and7 imagefeature channels; F0aux4zero,F1q*C/q. New reference_data evaluates full-image fitted fields at native crop/window coordinates, synchronized transforms; preview displays RGB only. Cache covers train/val, no test. No GT input to reference computation.

40tests, oldactualweights exactcompatibility,512batch2resource and registeredF0/F1 model/optimizer/scheduler/RNG resume equality passed. Source/config changed intentionally with full backups and newfrozen39file map; defaultlegacy paths unchanged. Inspect batchREADME/plan/status before recovery; oldbatch sourceguards must not be bypassed. Newconfigs are configs/pilot_reference_f0.json and_f1.json.

Queue: exactly two fromscratch40-epoch2400update seed42 pilots, each20step3407full25valevery10 selection, samebest3408fullval; comparison/predeclaredscreen/report/stop. NoF2/F3/Direct-reference/extra seeds/200epochs/test. SharedGPUlock and original8GiB/best-last protections remain. Current progress and any later completion must be read live.


## 18. Static semantic S0/S1/S2 batch (authorized and launched2026-10-20)

Entry: /root/miniconda3/envs/diffusion/bin/python -u work/launch_static_semantic_20261020.py. Batch outputs/launches/p1_static_semantic_20261020; full scope/current handoff docs/CHAT_HANDOFF_STATIC_TRAINING_20261020.md. Separate new schema2/model/trainer; old scripts/train.py and oldlaunchers are not this batch'sentry,never bypasshashes.

Exactsequence:S0historicalDirectreuse(completeequivalence,batch2/window1audited)→S1scratch40→S2scratch40→sameS2best3408val→report/stop. Newrunseed42,512,b2,2400updates/4800exposures;full25valevery10,S2fiveDDIM3407selects. No newtest/othermaterial/formal/reference-rescuequeue. Original120records/119uniqueRGB disclosed. Screening/state/loss details frozen inplan.

Read batch launcher.pid/status.json/launcher_canonical.log and runs under outputs/P1/static_semantic/seed_42. Trainerstdout batchtrain_S1_result.log/train_S2_result.log. Four DataLoader children inherit trainer commandline;usePPID andCUDAprocesslisting to distinguish them fromduplicatetrainers. TensorBoard outputs/P1 discoversnewruns.

Before any recoverycheck GPU/PID/children/lock;do notstart duplicatecoordinator. Sameentry checkslegacy/newhashes,reusescompletedstages,resumesidenticalfulllast. Atomicbest includesmatrices;last beginsepoch0;epoch40last completeswithoutupdates;incompleteinitialization/uncommittedtemps and failed3408results preserved. Keep8GiBfree/project26GiB/run1GiB limits. Earlierpreflightrevisionlogs arehistorical;canonicalcurrentlognamedabove. WaitfirstcompletevalidationforETA.
