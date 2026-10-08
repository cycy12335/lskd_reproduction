## 当前研究目标：方法论实验复现（2026-10-08 用户澄清）

以受控实验检验 logit 标准化的作用，不以精确复现论文表格数值或历史环境为验收条件。本节优先于下文历史计划中的相关要求。

- 核心比较：同一 Teacher/Student、数据、训练预算和随机种子，KD 与 LS-KD 固定 T=2、CE 权重 0.1、KD 权重 9，仅切换标准化。
- Baseline 衡量蒸馏整体收益；KD 与 LS-KD 的配对差值衡量该参数设置下的标准化收益。不能据单一设置推断所有温度和权重下均有效。
- 先完成 seed 42 的顺序实验，再按原计划进行多种子与温度消融、logits 分析。报告最佳与最终准确率、配对差值、均值和标准差，避免只挑有利结果。
- Baseline < KD < LS-KD 是待检验假设，不是强制结果。反向或无提升结果须保存、排查并如实报告；不以追求预期顺序为由反复调参或丢弃实验。
- 论文数值仅作背景参考；历史 KD 73.33% 的精确参数溯源不再阻塞本项目，也不自动增加历史协议复现实验。
- INC_002 按用户明确的研究范围关闭；关闭不代表已查明历史参数。原始数据保全、源码指纹、Teacher 验证、本地 GPU 和各阶段检查仍须遵守。

# 初始实验方案与论文对照说明

本次首先验证 Teacher，再启动 seed 42 的 Student Baseline。后续 KD、LS-KD 必须在前一阶段完成并检查结果后推进。

## 固定的官方来源

官方代码：https://github.com/sunshangquan/logit-standardization-KD

固定提交：`eb569ac3783f2eed16fbc2b021101175af4fb2f2`。

Baseline 配置来源：官方 `configs/cifar100/vanilla.yaml`。Teacher 来源：mdistiller 的 checkpoints release，归档内 `resnet32x4_vanilla/ckpt_epoch_240.pth`。

## Baseline 设置

Student 为 ResNet8×4，优化器为 SGD，训练 240 epochs，batch size 64，初始学习率 0.05，momentum 0.9，weight decay 0.0005。学习率从第 151、181、211 个 epoch 开始各乘 0.1，与官方 `epoch > stage` 的实现一致。

数据增强为 32×32 随机裁剪（padding 4）和随机水平翻转。归一化均值为 (0.5071, 0.4867, 0.4408)，标准差为 (0.2675, 0.2565, 0.2761)。正式训练使用完整的 50,000 张训练图像和 10,000 张测试图像，不缩短训练或使用子集替代。

Teacher 的预期 Top-1 为 79.42%，允许绝对偏差不超过 0.5 个百分点。Baseline 最佳准确率低于 70.5% 时建立异常报告并停止进入下一 Gate；论文中的 Student 参考准确率 72.50% 不代表本地结果必然达到该值。

## KD 对照的注意事项

固定的当前官方提交中，未标准化 KD 的默认配置也是 T=2、CE 权重 0.1、KD 权重 9。当前准备的 KD、LS-KD 配置使用相同参数，仅切换标准化，以支持单变量比较。不能用计划中的 alpha 示例公式替换这两个独立权重。

不能直接假定论文表中的 Vanilla KD 73.33% 使用了当前发布配置。用户已明确目标为方法论实验复现：采用上述匹配参数设计检验标准化效果，历史表格数值仅作背景，不再要求启动前完成其参数溯源。

## 已声明的差异与局限

- 实际显卡是 RTX 3060 Laptop，显存 6 GiB，区别于原计划的 8 GB。完整 batch 64 GPU smoke test 已通过，因此未降低 batch size。
- 本地 Python 3.10.20、PyTorch 2.8.0，区别于官方说明的 Python 3.8、PyTorch 1.7.0。
- 使用 FP32，关闭 AMP 和 TF32，开启确定性算法并关闭 cuDNN benchmark；官方代码开启 cuDNN benchmark。
- 本项目起始种子为 42。论文报告四次试验的均值，项目原计划为三个种子；单次结果不能作为论文均值的统计验证。

## 运行与追踪

正式 GPU 实验仅在本地执行，GitHub Actions 仅执行 CPU 单元测试和微型合成数据 smoke test。实验目录及 ID 不复用；checkpoint 保存在本地，元数据记录代码提交、配置摘要、Teacher 摘要、环境、指标和最佳 checkpoint 摘要。

当前未提供断点续训入口。失败实验保留现场，重试须分配新的 EXP ID。项目报告统一使用中文，原始配置、日志和机器可读指标中的字段名可以保留英文。

## 2026-10-08 接手核查补充

Baseline EXP_001 已完成 240 epochs，最佳 Top-1 72.53%，最终 72.23%，Gate 2 通过。完整报告见 reports/analysis/01_baseline_analysis.md。

KD 参数溯源待完成：本地固定版本为 T=2、CE=0.1、KD=9；mdistiller 现行默认 T=4、CE=0.1、KD=0.9，不能直接认定其对应论文 73.33% 的精确历史协议。详见 reports/incidents/INC_002_kd_protocol_provenance.md。匹配参数对照与论文历史默认协议应分别报告。EXP_002、EXP_003 未启动。
