# 项目交接文件

## 当前研究目标：方法论实验复现（2026-10-08 用户澄清）

以受控实验检验 logit 标准化的作用，不以精确复现论文表格数值或历史环境为验收条件。本节优先于下文历史计划中的相关要求。

- 核心比较：同一 Teacher/Student、数据、训练预算和随机种子，KD 与 LS-KD 固定 T=2、CE 权重 0.1、KD 权重 9，仅切换标准化。
- Baseline 衡量蒸馏整体收益；KD 与 LS-KD 的配对差值衡量该参数设置下的标准化收益。不能据单一设置推断所有温度和权重下均有效。
- 先完成 seed 42 的顺序实验，再按原计划进行多种子与温度消融、logits 分析。报告最佳与最终准确率、配对差值、均值和标准差，避免只挑有利结果。
- Baseline < KD < LS-KD 是待检验假设，不是强制结果。反向或无提升结果须保存、排查并如实报告；不以追求预期顺序为由反复调参或丢弃实验。
- 论文数值仅作背景参考；历史 KD 73.33% 的精确参数溯源不再阻塞本项目，也不自动增加历史协议复现实验。
- INC_002 按用户明确的研究范围关闭；关闭不代表已查明历史参数。原始数据保全、源码指纹、Teacher 验证、本地 GPU 和各阶段检查仍须遵守。


建立日期：2026-10-08（Asia/Shanghai）。项目根目录：`D:\reproduction`。

本文是交接时的状态快照。实验继续运行后会变化，接手时必须重新读取实时元数据、指标、日志并核对进程。

## 接手核查更新：2026-10-08 晚间

本节优先于下文历史快照；后续接手仍需核查实时状态。

- EXP_001 于今天 17:49:09（UTC+8）完成 240/240 epochs，最佳 Top-1 72.53%（epoch 239），最终 72.23%（epoch 240）。这是单次最终结果，不代表全项目完成。
- 20:17 原训练及启动 PID 均已退出，无 Python 训练进程，锁不存在，GPU 空闲；last_launch.json 是历史启动记录。
- 240 行指标和日志完整、数值有限、学习率时点正确；checkpoint 严格加载和 SHA256 验证通过；错误日志为空。Gate 2 通过。
- 中文报告：reports/analysis/01_baseline_analysis.md。中文摘要：experiments/baseline/EXP_001_baseline_seed42/summary.md。机器核查及原始记录哈希：reports/validation/EXP_001_completion_audit_20261008.json。
- INC_002（INVESTIGATING）：论文 KD 73.33% 的历史协议待溯源。本地 T=2/CE=0.1/KD=9 与 mdistiller 现行 T=4/CE=0.1/KD=0.9 不同，不能混作直接对照。见 reports/incidents/INC_002_kd_protocol_provenance.md。EXP_002、EXP_003 未启动，协议解决后再进入 Gate 3。
- 训练源码和环境未改变，源码指纹仍匹配 Teacher EXP_000 和 smoke test；Teacher Gate 固定 EXP_000 的限制保留。
- 接手前未提交修改已保留。registry.csv 已正确登记 completed/72.53，无需重写。原始元数据、指标、日志和 checkpoint 均未改动，未执行 reset/clean 或提交 checkpoint。

## 一、目标与用户约束

复现 CVPR 2024 论文 *Logit Standardization in Knowledge Distillation* 的 CIFAR-100 核心实验：ResNet32×4 Teacher → ResNet8×4 Student，依次完成 Teacher 验证、Baseline、KD、LS-KD，再进行多种子、温度消融、logits 分析及论文对照。

先阅读 `AGENTS.md`、`PROJECT_PLAN.md`、`reports/analysis_protocol.md`。

- 报告统一用中文，包括实验、阶段、异常和最终报告。配置、原始 JSON/CSV 字段、路径和模型名称可以保留英文。
- GitHub Actions 只负责 CI、单元测试和小规模 CPU smoke test；不得用于正式 CIFAR-100 GPU 实验，包括自托管 GPU runner。
- 正式训练在本地执行。实测显卡为 RTX 3060 Laptop、6 GiB 显存，区别于原计划假设的 8 GB。
- 大 checkpoint 保留本地，不进入普通 Git 历史。代码、配置、指标、分析和实验元数据纳入版本管理。
- 已登记 EXP ID 不复用，已完成实验的原始记录不覆盖；失败重试使用新 ID。
- 按 Gate 顺序推进。重大异常先保存现场、建立 Incident、通知用户并排查，再推进。准确率趋势是待验证目标，不能预设实验结论。

## 二、环境与版本

| 项目 | 配置 |
|---|---|
| 项目解释器 | `D:\reproduction\.venv\Scripts\python.exe` |
| Python | 3.10.20 |
| PyTorch / torchvision | 2.8.0+cu126 / 0.23.0+cu126 |
| CUDA 运行时 / 驱动 | 12.6 / 592.82 |
| 精度 | FP32，AMP 和 TF32 关闭 |
| 确定性 | 确定性算法开启，cuDNN benchmark 关闭 |
| 训练源码提交 | `7eb60bb7872a7f2085caa2e61fad1b035b76986c` |
| 官方源码提交 | `eb569ac3783f2eed16fbc2b021101175af4fb2f2` |

系统默认 Python 为 3.14，必须显式使用项目 `.venv` 解释器。精确依赖见 `requirements-lock-windows-gpu.txt`；环境验证见 `reports/environment/`。

交接前已有提交：`7eb60bb`（实现与预检）、`b24712e`（Teacher 验证与 Baseline 启动）。尚未配置 GitHub remote，未运行远程 Actions。训练中的指标和日志会持续变化，工作区存在实验记录更新属于正常现象，不能直接丢弃差异。

## 三、已完成验证

- 12 项单元测试通过，覆盖蒸馏公式、标准化、Teacher 冻结、checkpoint 往返、EXP ID 不可复用等行为。
- Baseline、KD、LS-KD 的 CPU 微型合成数据 smoke test 通过。
- 真实 CIFAR-100、batch 64、每种方法 2 个训练 batch 加小规模评估的 GPU smoke test 通过，无需降低 batch size。
- CIFAR-100 归档和 train/test 文件通过 torchvision 官方 MD5 校验。
- Gate 0（环境及 smoke test）、Gate 1（Teacher 验证）已通过。

证据：`reports/validation/PREFLIGHT.md`、`reports/validation/smoke_cpu_fake.json`、`reports/validation/smoke_cuda_cifar100.json`。

### EXP_000：Teacher 验证成功

完整测试集为 10,000 张图像。Top-1 **79.42%**、Top-5 **94.58%**、测试损失 **0.8833263866424561**。Top-1 与预期 79.42% 完全一致，验收容差为 0.5 个百分点。

报告和原始记录：`experiments/teacher/EXP_000_teacher_seed42/`。

Teacher：`models/pretrained/teacher/teacher_resnet32x4_cifar100.pth`。

SHA256：`22aa12e2b632b19ddde155ccf2388671842ddd304824b3095e0a7fff364beb4f`。

来源与下载记录见 `models/pretrained/teacher/source.json`。权重来自 mdistiller 官方 checkpoints release，严格 state-dict 加载通过。旧 checkpoint 的 NumPy 标量通过必要类型 allowlist 兼容，仍采用 `weights_only=True`。

### 已解决异常

`reports/incidents/INC_001_dataset_download.md`：官方单流下载在约 39MB 时中断。最终镜像归档与官方 MD5 完全一致，且 train/test 文件校验通过后才接受数据。完整溯源见 `data/manifest.json`。异常已解决。

## 四、EXP_001：Baseline 交接快照

建档时读取的实验元数据显示：状态为 `running`，已经完成 **96/240 个 epoch**。该数字为阶段快照，不代表接手时的最新进度。

此前已核实的第 88 个 epoch（2026-10-08 15:56:17，UTC+8）记录如下，保留其时间以免混同最新结果：阶段最佳测试准确率 56.05%（epoch 81），第 88 个 epoch 测试准确率 53.72%、训练准确率 64.32%、学习率 0.05，耗时约 34.22 秒。后续阶段最佳值以 `metadata.json` 为准。

中间结果不能作为最终复现结论。学习率从第 151、181、211 个 epoch 起各乘 0.1，不要仅凭未衰减时的阶段准确率判定失败。

正式配置：seed 42、batch 64、240 epochs、SGD 初始 LR 0.05、momentum 0.9、weight decay 0.0005；完整使用 50,000 张训练图像和 10,000 张测试图像。

2026-10-08 15:58（UTC+8）核查时，训练 PID **31780**、启动包装进程 PID **35820** 均存活，控制台错误输出为空。原始启动时间为 2026-10-08 15:07:09。PID 可能被系统复用，接手时还须核对启动时间和日志进展。

关键路径：

- 配置：`configs/baseline/resnet8x4_cifar100.yaml`。
- 实验目录：`experiments/baseline/EXP_001_baseline_seed42/`。
- 实时状态与每 epoch 指标：该目录的 `metadata.json`、`metrics.csv`。
- 训练日志与错误记录：该目录的 `train.log`、`error.log`。
- 本地 checkpoint：`models/checkpoints/baseline/EXP_001/EXP_001_best.pth`、`EXP_001_latest.pth`。
- 后台启动信息：`.tools/last_launch.json`；互斥锁：`.tools/active_experiment.lock`。
- 控制台日志：`logs/system/baseline_20261008_150709.stdout.log` 及对应 `.stderr.log`。

不要重复启动 EXP_001，不要删除活跃锁，不要因交接停止正常训练。当前没有断点续训入口；意外中断后保留现场、记录 Incident，再使用新 EXP ID 重试。锁残留时先确认进程确实退出，不能只凭 metadata 的 running 状态判断存活。

## 五、接手后的第一步

在 PowerShell 中执行只读核查：

```powershell
Set-Location D:\reproduction
Get-Content AGENTS.md -Raw -Encoding UTF8
Get-Content experiments/registry.csv -Encoding UTF8
Get-Content experiments/baseline/EXP_001_baseline_seed42/metadata.json -Raw -Encoding UTF8
Get-Content experiments/baseline/EXP_001_baseline_seed42/train.log -Tail 15 -Encoding UTF8
Get-Content experiments/baseline/EXP_001_baseline_seed42/error.log -Tail 30 -Encoding UTF8
Get-Content .tools/active_experiment.lock -Raw -Encoding UTF8
Get-Process -Id 31780,35820 -ErrorAction SilentlyContinue
nvidia-smi
git status --short
```

训练在后台执行，不依赖聊天窗口维持。当前未部署训练结束后的自动通知机制，接手者需主动检查。

本会话默认沙箱曾报 `setup refresh had errors`，不表示训练失败；必要时在用户授权范围内申请沙箱外操作。Windows PowerShell 读取中文 UTF-8 文件应显式指定 `-Encoding UTF8`，避免显示乱码。

## 六、后续任务

1. 等待 EXP_001 完成 240 epochs，检查状态、完整曲线、最佳/最终准确率、checkpoint 和 SHA256。
2. NaN、OOM、损坏、失败或严重偏差须建立 Incident 并通知用户，不直接推进。当前实现的 Baseline 最佳准确率低于 70.5% 会产生 warning；论文 Student 参考值 72.50% 不是预先认定的本地结果。
3. 整理中文最终报告。当前运行进程加载了旧英文模板，结束时可能把 `summary.md` 覆盖成英文；完成后翻译，保留原始指标，不为语言调整中断训练。后续再统一修改模板。
4. Baseline 完成并检查后，按 Gate 推进 EXP_002 KD、EXP_003 LS-KD；目前二者均未启动。
5. 当前 KD/LS-KD 配置匹配固定官方提交：T=2、CE 权重 0.1、KD 权重 9，仅切换标准化。不能直接等同于论文表格 Vanilla KD 73.33% 的历史协议；推进前解决 `reports/analysis_protocol.md` 中的对照问题。
6. 后续再进行 seed 3407/2024、多种子统计、温度消融、logits 分析和论文对照。ID 以 registry 为准，不重分配已有编号。
7. 配置 GitHub remote 后同步代码、配置、指标、报告和元数据；数据与 checkpoint 仍留在本地。

## 七、实现注意事项

训练入口：`scripts/train_baseline.py`、`train_kd.py`、`train_lskd.py`；Teacher 入口：`scripts/verify_teacher.py`；后台启动器：`scripts/start_experiment.ps1`。

训练实现：`training/runner.py`、`trainer.py`、`evaluator.py`；登记和哈希：`training/runtime.py`。

当前 Gate 检查相关 Python 文件的源码指纹。运行中不要修改训练代码，避免破坏可追踪比较。Teacher Gate 目前固定查找 EXP_000，并要求相同源码指纹：后续修改相关 Python 文件会使旧验证失效。应先设计支持新 Teacher 验证 ID 的 Gate，再按新 ID 重验；不得覆盖 EXP_000 或跳过校验。此处为后续工程事项，本次交接未修改实现。

首次环境报告中的“未开始实验”等字段属于历史快照，不能据此判断当前状态。工作区可能有未提交的报告翻译和持续更新的实验文件；先检查再选择性提交，不对活跃实验做 reset、clean 或覆盖。

## 八、参考来源

论文：https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html

官方代码：https://github.com/sunshangquan/logit-standardization-KD

Teacher release：https://github.com/megvii-research/mdistiller/releases/tag/checkpoints
