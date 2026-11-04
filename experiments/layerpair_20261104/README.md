# Direct 与 LayerPair-Net 同视场参考实验（2026-11-04）

本目录归档代码版本 7f8ca3e 上的实验。训练数据为项目已登记的 MoS₂ 划分，验证集 25 张。可行性检查只使用训练图，详见 [检查记录](feasibility/README.txt)。[Direct 架构图](direct_architecture.png) 对应一步预测及有序辅助监督。

- [四组主实验](direct_batch/)：每组 150 轮、3,000 步。Direct 在第 15 轮检查点后遇到 CUDA OOM，使用原始配置、仅增加运行时设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 后续训完成。最佳权重在第 60 轮（验证四类 mIoU 88.04%）。目录包含冻结计划、配置、历史及最佳检查点验证。
- [原型实现与冻结协议](prototype/protocol.txt)：两组共享相同 Direct 最佳权重，各训练 2,465 个参数，600 次更新，每批 512 个像素。A 读取查询附近的区域，B 读取同图不同预测组件的远处参考。仅校正单层与少层的概率分配。
- [评估结果](ab_results/assessment.json) 与 [逐图计数和指标](ab_results/assessment_per_image.csv)：在 3,657,709 个真实少层像素上，Direct、A、B 分别将 624,953、633,467、629,148 个判为单层。B 相对 A 减少 4,319 个，但相对 Direct 增加 4,195 个。B 相对 A 的反向错分增加 6,652 个。本次开发集结果不支持以 B 替换 Direct。

在原 AutoDL 布局中，先确保主实验完成，使用相同数据和评估所对应的 Direct 权重，再从一个空的可写结果目录复跑。入口为 prototype/run.py，参数依次为 ConditionalDiffusionSeg 项目根目录和新结果目录，环境为 diffusion，PYTHONPATH 指向项目 src。prototype/queue.sh 保存了等待主队列结束的原始启动方式。存档数值位于 ab_results/；模型权重与原图保留在服务器，权重校验值见 backbone.json、weights.sha256 及各组评估。下一项尚未执行的检查是定位 B 对材料内部与边界像素的净改判，复核实际参考块。
