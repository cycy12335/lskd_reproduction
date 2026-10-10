# EXP_001：Baseline 完成核查报告

核查日期：2026-10-08（Asia/Shanghai）。状态：completed，240/240 epochs，Gate 2 通过。本报告是 seed 42 的完整单次结果，不代表整个复现项目完成。

开始时间 15:07:13，结束时间 17:49:09，约 2 小时 42 分钟。20:17 实时检查：训练 PID 31780 和启动 PID 35820 均已退出，无 Python 训练进程；活跃锁不存在；RTX 3060 Laptop 显存 6144 MiB，利用率 0%、显存占用 0 MiB，无 GPU 进程。last_launch.json 为历史启动记录。

## 实验设置与结果

ResNet8×4，CIFAR-100，seed 42，batch 64，240 epochs。SGD 初始 LR 0.05、momentum 0.9、weight decay 0.0005；第 151、181、211 个 epoch 起各衰减十倍。完整训练集 50,000 张、测试集 10,000 张。FP32，关闭 AMP/TF32，确定性算法开启。

| 指标 | 最佳 Top-1 所在 epoch 239 | 最终 epoch 240 |
|---|---:|---:|
| 测试 Top-1 | 72.53% | 72.23% |
| 测试 Top-5 | 92.92% | 92.87% |
| 测试损失 | 1.0352342365264893 | 1.0422475337982178 |
| 训练准确率 | 95.332% | 95.154% |
| 训练损失 | 0.23376365236282348 | 0.2368817787837982 |

最佳值超过 70.5% 验收阈值，与项目协议所列论文 Student 参考值 72.50% 相差 +0.03 个百分点。单次结果不能替代多种子统计。最佳模型按测试集 Top-1 选择，最终模型为 epoch 240，二者分别报告。

## 完整指标与异常核查

metrics.csv 共 240 行，epoch 1–240 连续且无重复；train.log 的 240 条 epoch_complete 一一对应。全部数值有限，未发现 NaN/Inf。train_kd 全零，train_ce 与 train_loss 一致。学习率、最佳和最终指标与配置及 metadata.json 一致。error.log 与启动 stderr 均为 0 字节，未发现 OOM、异常退出或 checkpoint 损坏证据。

| 阶段 | 阶段最高 Top-1 | 阶段末 Top-1 | 阶段末训练损失 | 阶段末测试损失 |
|---|---:|---:|---:|---:|
| 1–150 | 58.27% | 53.98% | 1.22412 | 1.85180 |
| 151–180 | 71.69% | 68.56% | 0.47730 | 1.19283 |
| 181–210 | 72.48% | 72.16% | 0.25392 | 1.04050 |
| 211–240 | 72.53% | 72.23% | 0.23688 | 1.04225 |

学习率衰减后准确率提高。交接时 96 epochs 是历史阶段快照，不能当作最终结果。未发现需要重跑 Baseline 的异常。

## checkpoint 与溯源

使用项目 .venv Python 3.10.20，以 torch.load(weights_only=True, map_location='cpu') 验证：两份 checkpoint 的 state_dict 严格加载成功，张量有限，epoch、配置与训练提交一致。本次未重新执行完整测试集推理，准确率来自原训练评估记录。

- best：models/checkpoints/baseline/EXP_001/EXP_001_best.pth，epoch 239，9,922,307 字节。
  SHA256：124d547eb993db2fe8165f177552f34a15baf28ef601966601b9a6f845c9c0f1，与原 metadata.json 一致。
- latest：models/checkpoints/baseline/EXP_001/EXP_001_latest.pth，epoch 240，9,922,577 字节。
  SHA256：ae040da5a937d0d5bf4a33cadca95fb1683a9217e1863d9f2cadd2887d943864，本次补充记录。

训练提交：7eb60bb7872a7f2085caa2e61fad1b035b76986c。核查 HEAD：1c3e7ed（完整值见机器记录）。
源码指纹：b8c6fa0e7d8b14068f72b3538fc4712865375efa1fca1bc8eecba956c5ccf3ff，与 EXP_001、EXP_000 和 CUDA smoke test 相同。
配置 SHA256 与原 metadata 一致。Teacher SHA256 仍为 22aa12e2b632b19ddde155ccf2388671842ddd304824b3095e0a7fff364beb4f。

checkpoint 被 Git 忽略，git ls-files 未发现已跟踪 .pth/.pt/.ckpt。现有未提交修改主要为训练完成记录及中文报告翻译，均保留。机器核查和原始文件哈希：reports/validation/EXP_001_completion_audit_20261008.json。

## 后续安排

Gate 2 通过，但 KD 论文对照协议待完成溯源，见 reports/incidents/INC_002_kd_protocol_provenance.md。EXP_002、EXP_003 均未启动；协议解决后再进入 Gate 3。未修改源码或环境，未覆盖已完成实验的原始机器记录。