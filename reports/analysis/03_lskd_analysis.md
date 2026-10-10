# Seed 42 核心实验阶段报告：Baseline、KD 与 LS-KD

日期：2026-10-09（Asia/Shanghai）。本报告是方法论实验复现的首个种子结果，不是多种子最终结论。

## 主要结果

| 方法 | 最佳 Top-1 | 最佳 epoch | 最终 Top-1 | 最终 Top-5 |
|---|---:|---:|---:|---:|
| Baseline | 72.53% | 239 | 72.23% | 92.87% |
| KD | 73.34% | 183 | 72.64% | 92.35% |
| LS-KD | 76.51% | 190 | 75.96% | 93.18% |

LS-KD 相对 KD：最佳 Top-1 +3.17 个百分点，最终 Top-1 +3.32 个百分点。相对 Baseline：最佳 +3.98、最终 +3.73 个百分点。KD 相对 Baseline：最佳 +0.81、最终 +0.41 个百分点。

在本次 seed 42 与固定超参数条件下，最佳和最终 Top-1 均满足 Baseline < KD < LS-KD，支持标准化对分类准确率有益。不能据单次结果宣称统计显著或跨种子稳定，也不能推断所有温度、权重和架构下均有效。

## 受控对照与范围

CIFAR-100，Teacher ResNet32×4，Student ResNet8×4，完整训练集 50,000 张、测试集 10,000 张；seed 42、batch 64、240 epochs。SGD 初始 LR 0.05、momentum 0.9、weight decay 0.0005，151/181/211 起各衰减十倍。FP32，确定性算法开启。GPU 为本地 RTX 3060 Laptop 6 GiB。

KD 与 LS-KD 固定 T=2、CE=0.1、KD=9，方法处理上的差别仅为标准化开关；损失、架构、数据与优化设置已经核对。KD 在 epoch 120 后有用户主动暂停和续训，使用独立恢复入口；因此执行历史并非完全相同，已通过恢复状态测试、前段指标保全和部分日志重放一致性检查。不得隐去该操作差异。

Baseline 只使用 CE，KD/LS-KD 训练总损失含加权蒸馏项，不能直接比较三者训练总损失大小。最佳模型均按测试集 Top-1 选择，应同时报告最终 epoch；这些最佳值并非独立验证集选模后的无偏测试估计。论文表格数值仅作背景，本项目不以精确匹配历史数字为目标。

## LS-KD 完成验收

EXP_003 于 2026-10-09 15:25:07（UTC+8）完成 240 epochs，状态 completed。训练及启动进程已退出，活跃锁不存在，核查 GPU 利用率为 0%。

metrics.csv 的 240 行和 train.log 的 240 条 epoch_complete 均连续无重复；所有数值有限，LR 衰减时点、加权损失关系、最佳与最终指标均一致。error.log 和启动 stderr 为空。best/latest 模型严格加载成功，张量有限，配置、epoch、提交与源码指纹匹配。

best：epoch 190，Top-1 76.51%、Top-5 93.41%、测试 CE 3.910056307601929。
latest：epoch 240，Top-1 75.96%、Top-5 93.18%、测试 CE 4.044862754058838。

Gate 4 的完整性与分类准确率检查通过，首个种子的核心对照已完成；尚未进行多种子稳定性检验。

## logits 尺度与置信度现象

对三种方法的 best checkpoint 及 LS-KD latest 重新进行完整 10,000 张测试集推理，Top-1、Top-5、CE 均与原训练记录一致。下列数值均从未经温度校准的原始 logits 计算；标准差是逐样本在 100 个类别上计算后取均值。

| best 模型 | 原始 logits 测试 CE | 平均原始 logits 标准差 | 原始 logits 经 softmax(T=1) 的平均最高概率 |
|---|---:|---:|---:|
| Baseline | 1.03523 | 3.81550 | 80.01% |
| KD | 1.17963 | 3.69881 | 86.09% |
| LS-KD | 3.91006 | 0.11991 | 2.64% |

LS-KD 的类别排序准确率更高，但原始 logits 尺度明显更小，softmax 概率较平坦，因此 CE 较高、原始置信度明显偏低。这是 checkpoint 推理可重复观察到的现象，不是仅凭训练日志推测。标准化蒸馏项不要求匹配 Teacher 的绝对 logits 尺度，与该现象相容；具体形成原因仍需结合优化和尺度随 epoch 的变化进一步分析，不能仅凭相关性确定。

结论应限定为分类准确率改善，不能表述为概率质量全面改善。本轮没有进行温度校准，也没有使用测试集调参改变报告数字。后续如研究校准，应另行定义验证数据与方案。

## checkpoint 和溯源

LS-KD best：models/checkpoints/lskd/EXP_003/EXP_003_best.pth，9,922,435 字节。
SHA256：724aaafc5a8d279f992c00d3a3d457bbd859f834f34e6d742efcd6ea030ecd1d。

LS-KD latest：models/checkpoints/lskd/EXP_003/EXP_003_latest.pth，9,922,705 字节。
SHA256：51b54ca2763c4ca0b4d357e79a31d79311429a89317229d020a306afe2a365d0。

LS-KD 提交：38fcf3f1806142ae203445fb359b846377c6f7c6。
共同核心源码指纹：b8c6fa0e7d8b14068f72b3538fc4712865375efa1fca1bc8eecba956c5ccf3ff。
KD 恢复入口版本与哈希详见 EXP_002 的 metadata.resume_sessions；不能用核心指纹替代该入口的验证。

原始机器记录和 checkpoint 未改动。中文摘要只补充结果说明。

- 完成核查：reports/validation/EXP_003_completion_audit_20261009.json。
- 完整 checkpoint 推理与 logits 统计：reports/validation/seed42_checkpoint_reevaluation_20261009.json。
- 对照表：results/tables/main_results.csv。
- Baseline 与 KD 报告：reports/analysis/01_baseline_analysis.md、02_kd_analysis.md。

## 下一阶段

按原计划用 seed 3407、2024 分别补充三个方法，汇总均值、标准差及同 seed 的 LS-KD−KD 配对差值，再进行温度消融与更完整 logits 分析。本次完成核查与阶段总结，尚未启动新一轮训练。

## 预测概率口径复核（2026-10-09）

针对用户对 2.64% 的疑问，独立以 float64 在完整 10,000 张测试图像上重算。统计定义为：每张图在类别维度对 logits 做 softmax，取 100 类中最大概率，再对所有图像求均值。不是正确率，也不是对类别概率取平均。每张图的概率和验证为 1，百分比换算正确。

LS-KD best 原始 logits 的平均最高概率为 0.0264089913，即 2.64089913%，确认原数值正确。原表标题未明确写出“原始 logits、T=1”，容易与标准化蒸馏分支混淆，现已补充。

| 同一 LS-KD best 模型的概率计算方式 | 平均最高概率 | 测试 CE | Top-1 |
|---|---:|---:|---:|
| softmax(z)，原始 logits，T=1 | 2.64% | 3.91006 | 76.51% |
| softmax(standardize(z))，标准化后，T=1 | 63.39% | 1.03147 | 76.51% |
| softmax(standardize(z)/2)，对应蒸馏分支温度 | 15.60% | 2.20408 | 76.51% |

三种处理在所有 10,000 张图像上的 argmax 完全相同；标准化改变概率尺度，不改变该样本的类别顺序。因此很低的最高概率与较高 Top-1 可以同时出现。例：一张正确分类图像的原始最高概率仅 1.466%，但真实类别仍排第一。

后两行是解释口径的附加分析，没有更改模型、训练记录或默认评估规则，也不是在独立验证集上完成的概率校准。不能用它们替换原表的一行而保持其他模型口径未说明。对概率质量的结论必须明确所用变换；原始概率偏低不意味着所有处理后的输出都同样偏低。

复核原始数据：reports/validation/EXP_003_probability_check_20261009.json。
