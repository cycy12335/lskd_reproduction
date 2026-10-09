# EXP_002：KD 完成报告

核查日期：2026-10-09（Asia/Shanghai）。状态 completed，240/240 epochs，Gate 3 通过。训练于 2026-10-09 10:09:45（UTC+8）结束。核查时训练进程已退出，活跃锁不存在，GPU 利用率 0%。

## 方法与结果

Teacher ResNet32×4 → Student ResNet8×4；CIFAR-100 完整训练集 50,000 张、测试集 10,000 张；seed 42、batch 64、240 epochs。SGD 与 Baseline 配置一致；T=2、CE 权重 0.1、KD 权重 9、standardized=false。该结果属于匹配参数的方法论复现，不以论文历史数值为验收目标。

| 指标 | 最佳 Top-1 所在 epoch 183 | 最终 epoch 240 |
|---|---:|---:|
| 测试 Top-1 | 73.34% | 72.64% |
| 测试 Top-5 | 92.65% | 92.35% |
| 测试 CE 损失 | 1.1796309757232666 | 1.2043869707107544 |
| 训练准确率 | 95.282% | 97.282% |
| 训练总损失 | 11.10157169342041 | 9.222972548828125 |

| 方法 | 最佳 Top-1 | 最终 Top-1 |
|---|---:|---:|
| Baseline EXP_001 | 72.53% | 72.23% |
| KD EXP_002 | 73.34% | 72.64% |
| KD − Baseline（百分点） | +0.81 | +0.41 |

在 seed 42 和该配置下，KD 在最佳和最终 Top-1 上均优于 Baseline。单次结果尚不能证明跨种子的稳定性，也不能说明标准化有效；标准化效果须等 LS-KD 完成后比较。最佳 epoch 按测试集 Top-1 选择，并非独立验证集选择后的无偏测试估计。KD 的训练总损失含加权蒸馏项，不能直接与 Baseline 的训练 CE 数值比较。

## 完整性检查

240 行 epoch 连续无重复，240 条 epoch_complete 日志一一对应，数值均有限；学习率在 151/181/211 衰减，训练总损失与 0.1×CE + 9×KD 一致。最佳/最终指标与 metadata 相符。暂停前 120 行指标与归档完全一致，恢复后追加 121–240。日志中未完成的旧 epoch 121 尝试保留，有恢复标记区分，不计为完成 epoch。

用户在 epoch 120 后暂停；2026-10-09 从保存状态继续。恢复保留模型、SGD 动量、Torch/CUDA RNG 和训练 loader RNG。细节和 14 项测试证据见 reports/validation/EXP_002_resume_validation_20261009.md。首轮及恢复 stderr、实验 error.log 均为空，未发现 NaN、OOM 或 checkpoint 损坏。

best/latest 均通过 CPU weights_only 加载、模型严格加载、张量有限性及 epoch/配置/来源检查。此处准确率来自原训练评估记录，本次未重新执行完整测试集推理。

## checkpoint 与溯源

- best：models/checkpoints/kd/EXP_002/EXP_002_best.pth，epoch 183，9,922,627 字节。
  SHA256：6a0b52a3f25c4a68c730ba200a16fe9ad31918eef25aa8b63e12055deaefbe86。
- latest：models/checkpoints/kd/EXP_002/EXP_002_latest.pth，epoch 240，9,922,897 字节。
  SHA256：2a55cc7daaddbd386f3f88e67310639f24865c240a91ed184049ecf66a49af53。

两者均与 metadata 中对应哈希一致；checkpoint 留在本地。
初始提交：94d6d072bc81cd26542912cab8abcaccd8c78a29。
续训入口提交：670c65cd39fc9ae803d7e0d16f513ca24ce3d01f。
核心指纹：b8c6fa0e7d8b14068f72b3538fc4712865375efa1fca1bc8eecba956c5ccf3ff。
续训入口 SHA256：cd23133ae35e3f80fb73cd78cc31854ae05eff09d5ae2330fa219f681554f527。
两项指纹分别验证，不能用核心指纹替代入口验证。

机器核查：reports/validation/EXP_002_completion_audit_20261009.json。原配置、metrics、metadata、日志未改写。

## 下一阶段

EXP_003 LS-KD 的 Gate 检查通过；保持相同 Teacher/Student、seed、训练预算、T=2、CE=0.1、KD=9，仅 standardized=true。先完成该实验，再判断标准化作用并进行多种子验证。训练过程中的准确率只作阶段进度，不作最终结论。
