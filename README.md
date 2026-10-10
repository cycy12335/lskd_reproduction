# LS-KD 方法复现

复现 CVPR 2024 论文 *Logit Standardization in Knowledge Distillation* 的 logit 标准化方法，通过受控实验检验其在知识蒸馏中的作用。数据集为 CIFAR-100，Teacher 为 ResNet32×4，Student 为 ResNet8×4。正式训练在本地 RTX 3060 Laptop（实测 **6 GiB 显存**）执行。

## 已完成结果

seed 42 的三组实验均完成 240 epochs，并通过完整性检查和 checkpoint 全测试集推理核查。

| 方法 | 实验 | 最佳 Top-1 | 最终 Top-1 |
|---|---|---:|---:|
| Baseline | EXP_001 | 72.53% | 72.23% |
| KD | EXP_002 | 73.34% | 72.64% |
| LS-KD | EXP_003 | 76.51% | 75.96% |

Teacher 验证 Top-1 为 79.42%。KD 与 LS-KD 均固定 T=2、CE 权重 0.1、KD 权重 9，仅切换 logit 标准化。LS-KD 相对 KD 的最佳与最终 Top-1 分别提高 3.17、3.32 个百分点。

这些结果支持当前参数设置下的分类准确率收益；多种子对照和温度消融尚未完成，不能据此宣称统计显著性或跨种子稳定性。本项目以方法论实验为目标，不以精确匹配论文表格数值为验收条件。LS-KD 原始输出置信度较低（best 模型平均最高预测概率 2.64%，测试 CE 3.91006），因此准确率提升不代表概率质量全面提升。

已完成的 seed 3407 Baseline（EXP_004）最佳/最终 Top-1 为 72.67%/72.47%。2026-10-10 发布快照时，EXP_006 正在续训，其余多种子实验按队列继续；运行记录中的 running 和中间指标只代表快照时状态。

- [核心结果表](results/tables/main_results.csv)
- [Baseline 分析](reports/analysis/01_baseline_analysis.md)、[KD 分析](reports/analysis/02_kd_analysis.md)、[LS-KD 分析](reports/analysis/03_lskd_analysis.md)
- [实验登记](experiments/registry.csv)、[方法与偏差说明](reports/analysis_protocol.md)
- [交接入口](HANDOFF.md)、[多种子计划](reports/progress/multiseed_20261009/计划.md)

## 环境与执行

Python 3.10.20、PyTorch 2.8.0+cu126、torchvision 0.23.0+cu126。Windows 本地使用 `.venv` 解释器；锁定依赖见 `requirements-lock-windows-gpu.txt`，CPU CI 依赖见 `requirements-cpu.txt`。

```powershell
powershell -File scripts/setup_environment.ps1
# 或：conda env create -f environment.yml
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run_smoke_test.py --device cpu --dataset fake --batch-size 4
.\.venv\Scripts\python.exe scripts/run_smoke_test.py --device cuda --dataset cifar100 --batch-size 64
.\.venv\Scripts\python.exe scripts/verify_teacher.py
```

执行顺序为环境与 smoke test → Teacher 验证 → Baseline → KD → LS-KD。正式训练前先提交代码与配置，并通过各阶段 Gate。训练入口见 `scripts/start_experiment.ps1`；使用新的 `--experiment-id EXP_XXX`，不得复用已有实验 ID 或覆盖已完成记录。现有 EXP_000–EXP_009 配置用于历史溯源，在新环境运行时应另建 ID。

正式训练使用 batch 64、240 epochs、SGD、初始 LR 0.05、momentum 0.9、weight decay 0.0005。恢复既有实验的专用入口为 `operations/resume_kd.py`（EXP_002）与 `operations/resume_exp006.py`（EXP_006）；多种子队列入口为 `operations/multiseed_queue.py`。恢复前核查 metadata、checkpoint、源码指纹、锁与进程，具体条件见相应续训验证报告。`.tools/active_experiment.lock` 防止重复启动。

## 数据、权重与溯源

仓库包含代码、配置、原始指标、实验元数据、日志、分析和验证报告。数据集、Teacher 权重及训练 checkpoint 保留本地，不进入普通 Git 历史。克隆仓库后需重新准备数据并获取 Teacher 权重；其来源与 SHA256 见 [Teacher 来源](models/pretrained/teacher/source.json) 和 [Teacher 说明](models/pretrained/teacher/README.md)。数据完整性记录见 `data/manifest.json`。

每项实验的 `metadata.json` 记录训练提交、源码指纹、配置和 checkpoint 哈希；`metrics.csv` 保留逐 epoch 指标。历史报告的运行状态具有时间性，当前本地状态以实时文件及实际进程为准。checkpoint 未上传，因此 GitHub 中的验证记录应结合本地权重核查。

## GitHub Actions

Actions 仅运行单元测试、CPU 环境检查和小规模合成数据 CPU smoke test。禁止在 Actions 中进行正式 CIFAR-100 GPU 训练，包括自托管 GPU runner。正式实验始终在本地执行。

## 参考

- [原论文](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html)
- [官方代码](https://github.com/sunshangquan/logit-standardization-KD)
- [第三方代码声明](THIRD_PARTY_NOTICES.md)
- [原始项目计划](PROJECT_PLAN.md)
