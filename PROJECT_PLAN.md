# Logit Standardization in Knowledge Distillation
## 论文复现项目实验与工程管理计划

> Paper: **Logit Standardization in Knowledge Distillation — CVPR 2024**  
> Task: Knowledge Distillation / Image Classification  
> Dataset: CIFAR-100  
> GPU: NVIDIA RTX 3060 8GB  
> Framework: PyTorch  
> Project Type: Paper Reproduction  
> Project Status: Experimental Reproduction

---

# 0. 项目目标

本项目目标是在单张 RTX 3060 8GB GPU 环境下，复现论文 **Logit Standardization in Knowledge Distillation** 的 CIFAR-100 核心实验。

核心验证目标：

\[
Student < Vanilla\ KD < LS-KD
\]

项目需要完成：

- Student Baseline
- Teacher Verification
- Vanilla Knowledge Distillation
- Logit Standardization Knowledge Distillation
- 多随机种子实验
- Temperature Ablation
- Logits Statistical Analysis
- 原论文结果对比
- 实验异常记录
- 最终复现报告

本项目强调：

> **可复现、可追踪、可比较、可审计。**

任何实验结果必须能够追溯至：

```text
Code Version
    +
Configuration
    +
Random Seed
    +
Model Checkpoint
    +
Training Log
    +
Metrics
```

---

# 1. 项目管理原则

## 1.1 实验与代码分离

代码、模型、实验结果和报告禁止混合存放。

统一按照：

```text
Code
Models
Experiments
Results
Reports
Logs
```

进行管理。

---

## 1.2 实验必须具有唯一 ID

所有正式实验使用：

```text
EXP_XXX
```

编号。

例如：

```text
EXP_001_student_baseline_seed42
EXP_002_kd_T4_seed42
EXP_003_lskd_T4_seed42
```

禁止使用：

```text
test
test2
new
final
final2
final_final
```

---

## 1.3 不允许覆盖旧实验

已经完成的实验：

> **禁止直接覆盖。**

如果参数发生改变：

```text
EXP_003
```

不得覆盖。

必须建立：

```text
EXP_004
```

即使只是修改了一个参数。

---

# 2. GitHub Repository

建议仓库名称：

```text
lskd-cifar100-reproduction
```

GitHub Repository：

```text
lskd-cifar100-reproduction/
```

README 应说明：

```text
Paper
Dataset
Teacher
Student
Hardware
Environment
Quick Start
Experiment Status
Main Results
```

---

# 3. 项目目录结构

项目根目录：

```text
lskd-cifar100-reproduction/
│
├── README.md
├── PROJECT_PLAN.md
├── CHANGELOG.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   └── smoke_test.yml
│   │
│   └── ISSUE_TEMPLATE/
│       ├── experiment_error.md
│       └── unexpected_result.md
│
├── configs/
│   ├── baseline/
│   │   └── resnet8x4_cifar100.yaml
│   │
│   ├── kd/
│   │   └── resnet32x4_resnet8x4.yaml
│   │
│   └── lskd/
│       └── resnet32x4_resnet8x4.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── models/
│   ├── architectures/
│   │   ├── resnet.py
│   │   └── __init__.py
│   │
│   ├── pretrained/
│   │   └── teacher/
│   │
│   └── checkpoints/
│       ├── baseline/
│       ├── kd/
│       └── lskd/
│
├── losses/
│   ├── cross_entropy.py
│   ├── kd.py
│   └── lskd.py
│
├── datasets/
│   └── cifar100.py
│
├── training/
│   ├── trainer.py
│   ├── evaluator.py
│   └── scheduler.py
│
├── analysis/
│   ├── analyze_logits.py
│   ├── compare_methods.py
│   ├── aggregate_seeds.py
│   └── plot_results.py
│
├── scripts/
│   ├── setup_project.py
│   ├── verify_environment.py
│   ├── verify_teacher.py
│   ├── train_baseline.py
│   ├── train_kd.py
│   ├── train_lskd.py
│   ├── run_ablation.py
│   └── run_smoke_test.py
│
├── experiments/
│   ├── registry.csv
│   │
│   ├── baseline/
│   ├── kd/
│   ├── lskd/
│   └── ablation/
│
├── results/
│   ├── tables/
│   ├── figures/
│   ├── statistics/
│   └── comparison/
│
├── reports/
│   ├── progress/
│   ├── incidents/
│   ├── analysis/
│   └── final/
│
├── logs/
│   ├── training/
│   ├── errors/
│   └── system/
│
├── tests/
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_dataset.py
│   └── test_forward.py
│
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

# 4. 各目录职责

## `configs/`

只保存实验配置。

例如：

```text
configs/kd/resnet32x4_resnet8x4.yaml
```

包括：

```yaml
dataset: CIFAR100

teacher:
  model: resnet32x4

student:
  model: resnet8x4

training:
  epochs: XXX
  batch_size: XXX
  learning_rate: XXX
  weight_decay: XXX

distillation:
  temperature: XXX
  alpha: XXX

seed: 42
```

---

# 5. 模型存储

模型统一放：

```text
models/
```

Teacher：

```text
models/pretrained/teacher/
```

例如：

```text
teacher_resnet32x4_cifar100.pth
```

Student Baseline：

```text
models/checkpoints/baseline/
```

KD：

```text
models/checkpoints/kd/
```

LS-KD：

```text
models/checkpoints/lskd/
```

命名：

```text
EXP_001_student_resnet8x4_seed42_best.pth

EXP_002_kd_resnet32x4_resnet8x4_T4_seed42_best.pth

EXP_003_lskd_resnet32x4_resnet8x4_T4_seed42_best.pth
```

---

# 6. Git 管理大型模型

**不要直接将大型 `.pth` checkpoint 提交到普通 Git 历史。**

`.gitignore`：

```gitignore
data/raw/
models/pretrained/
models/checkpoints/

*.pth
*.pt
*.ckpt

__pycache__/
*.pyc

.venv/
venv/

wandb/
runs/

.DS_Store
```

Teacher 下载地址和校验信息记录在：

```text
models/pretrained/teacher/README.md
```

包括：

```text
Model:
Source:
Download Date:
Expected Accuracy:
SHA256:
Original Filename:
Local Filename:
```

---

# 7. Experiment Registry

建立：

```text
experiments/registry.csv
```

格式：

```text
experiment_id,
date,
method,
teacher,
student,
temperature,
alpha,
seed,
status,
best_accuracy,
git_commit,
notes
```

示例：

```text
EXP_001,2026-XX-XX,baseline,None,resnet8x4,None,None,42,completed,XX.XX,abc123,normal

EXP_002,2026-XX-XX,KD,resnet32x4,resnet8x4,4,0.9,42,completed,XX.XX,abc124,normal
```

所有正式实验必须登记。

---

# 8. 单次实验目录规范

例如：

```text
experiments/kd/EXP_002_kd_T4_seed42/
```

必须包含：

```text
EXP_002_kd_T4_seed42/
│
├── config.yaml
├── metadata.json
├── metrics.csv
├── train.log
├── summary.md
└── error.log
```

其中：

### `config.yaml`

记录全部超参数。

### `metadata.json`

记录：

```text
Experiment ID
Start Time
End Time
GPU
CUDA
PyTorch
Python
Git Commit
Random Seed
```

### `metrics.csv`

至少：

```text
epoch
train_loss
train_accuracy
test_loss
test_accuracy
learning_rate
```

### `summary.md`

记录：

```text
实验目的
实验配置
最终结果
最佳结果
是否符合预期
发现的问题
下一步
```

---

# 9. 本地环境启动

## 9.1 Clone

```bash
git clone <repository>
cd lskd-cifar100-reproduction
```

---

## 9.2 Conda 环境

```bash
conda create -n lskd python=3.10
conda activate lskd
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 9.3 环境验证

运行：

```bash
python scripts/verify_environment.py
```

需要检查：

```text
Python
PyTorch
CUDA
GPU
GPU Memory
CIFAR-100
```

预期：

```text
[OK] Python
[OK] PyTorch
[OK] CUDA available
[OK] GPU: NVIDIA RTX 3060
[OK] CIFAR-100
```

---

# 10. Smoke Test

正式训练前必须运行：

```bash
python scripts/run_smoke_test.py
```

只使用：

```text
少量 CIFAR-100 samples
1-2 batches
1 epoch
```

验证：

```text
Dataset
↓
Model
↓
Forward
↓
Loss
↓
Backward
↓
Optimizer
↓
Evaluation
```

全部正常。

Smoke Test 通过后才允许进行正式实验。

---

# 11. GitHub Actions

建立：

```text
.github/workflows/
```

至少两个 Workflow。

---

## Workflow 1：CI

文件：

```text
.github/workflows/ci.yml
```

触发：

```text
Push
Pull Request
```

运行：

```text
Install Python
↓
Install Dependencies
↓
Run Unit Tests
↓
Import Test
↓
Model Forward Test
↓
Loss Function Test
```

GitHub Actions **不要执行完整 CIFAR-100 训练**。

CI 的目标不是训练模型，而是：

> 防止代码更新后实验 Pipeline 被破坏。

---

## Workflow 2：Smoke Test

文件：

```text
.github/workflows/smoke_test.yml
```

运行 CPU 小规模测试：

```text
Fake / Tiny Dataset
↓
Teacher Forward
↓
Student Forward
↓
KD Loss
↓
LS-KD Loss
↓
Backward
```

验证：

```text
Baseline Pipeline
KD Pipeline
LS-KD Pipeline
```

---

# 12. Git Workflow

建议使用：

```text
main
│
├── dev
│
├── feature/kd
│
├── feature/lskd
│
└── experiment/temperature-ablation
```

原则：

```text
main
```

始终保持可运行。

实验开发在：

```text
dev
```

或：

```text
feature/*
```

完成测试后合并。

---

# 13. Commit 规范

建议：

```text
feat:
fix:
exp:
analysis:
docs:
test:
refactor:
```

例如：

```text
feat: implement vanilla KD loss

exp: run EXP_002 KD T4 seed42

fix: correct KL divergence temperature scaling

analysis: compare KD and LSKD results

docs: update experiment report
```

---

# 14. 完整实验流程

正式实验按照以下顺序执行。

```text
Environment Verification
          ↓
Smoke Test
          ↓
Teacher Verification
          ↓
Student Baseline
          ↓
Vanilla KD
          ↓
LS-KD
          ↓
Compare Main Results
          ↓
Multiple Seeds
          ↓
Temperature Ablation
          ↓
Logit Analysis
          ↓
Paper Comparison
          ↓
Final Report
```

---

# 15. EXP_000：Teacher Verification

Teacher：

```text
ResNet32×4
```

任务：

```text
Load Checkpoint
↓
CIFAR-100 Test
↓
Calculate Accuracy
```

运行：

```bash
python scripts/verify_teacher.py
```

记录：

```text
Teacher Accuracy
Checkpoint SHA256
Model Architecture
```

验收：

```text
Teacher Accuracy ≈ Expected Accuracy
```

如果差异明显：

> **停止后续实验。**

进入 Incident Process。

---

# 16. EXP_001：Student Baseline

模型：

```text
ResNet8×4
```

Loss：

\[
L=L_{CE}
\]

运行：

```bash
python scripts/train_baseline.py \
    --config configs/baseline/resnet8x4_cifar100.yaml
```

产出：

```text
models/checkpoints/baseline/

experiments/baseline/EXP_001_*/

logs/training/
```

记录：

```text
Train Loss
Train Accuracy
Test Accuracy
Best Accuracy
Training Time
GPU Memory
```

---

# 17. EXP_002：Vanilla KD

Teacher：

```text
ResNet32×4
```

Student：

```text
ResNet8×4
```

Loss：

\[
L=
(1-\alpha)L_{CE}
+
\alpha T^2L_{KL}
\]

运行：

```bash
python scripts/train_kd.py \
    --config configs/kd/resnet32x4_resnet8x4.yaml
```

预期：

\[
Accuracy_{KD}>Accuracy_{Baseline}
\]

---

# 18. EXP_003：LS-KD

唯一核心变量：

```text
Logit Standardization = ON
```

运行：

```bash
python scripts/train_lskd.py \
    --config configs/lskd/resnet32x4_resnet8x4.yaml
```

预期：

\[
Accuracy_{LSKD}>Accuracy_{KD}
\]

---

# 19. 第一阶段核心结果

生成：

```text
results/tables/main_results.csv
```

格式：

| Method | Teacher | Student | Accuracy | Improvement |
|---|---|---|---:|---:|
| Baseline | — | ResNet8×4 | | |
| KD | ResNet32×4 | ResNet8×4 | | |
| LS-KD | ResNet32×4 | ResNet8×4 | | |

核心验证：

\[
Baseline < KD < LSKD
\]

---

# 20. 多随机种子实验

核心实验通过后执行：

```text
Seed = 42
Seed = 3407
Seed = 2024
```

分别运行：

```text
Baseline
KD
LS-KD
```

最终计算：

\[
Mean\pm SD
\]

输出：

```text
results/statistics/multi_seed_results.csv
```

例如：

| Method | Seed 42 | Seed 3407 | Seed 2024 | Mean | SD |
|---|---:|---:|---:|---:|---:|
| Baseline | | | | | |
| KD | | | | | |
| LS-KD | | | | | |

---

# 21. Temperature Ablation

设置：

\[
T\in\{1,2,4,8\}
\]

比较：

```text
KD
vs
LS-KD
```

输出：

```text
results/tables/temperature_ablation.csv
```

以及：

```text
results/figures/temperature_ablation.png
```

---

# 22. Logits Statistical Analysis

分析：

```text
Teacher Logits
Student Logits
Standardized Teacher Logits
Standardized Student Logits
```

计算：

\[
Mean
\]

\[
Standard\ Deviation
\]

\[
Maximum
\]

必要时进一步分析：

```text
Confidence
Entropy
```

输出：

```text
results/statistics/logit_statistics.csv
```

图：

```text
results/figures/
├── teacher_student_logits_distribution.png
├── standardized_logits_distribution.png
└── logit_std_vs_epoch.png
```

---

# 23. 原论文结果比较

建立：

```text
results/comparison/paper_vs_reproduction.csv
```

格式：

| Method | Paper | Reproduction | Difference |
|---|---:|---:|---:|
| Baseline | | | |
| KD | | | |
| LS-KD | | | |

重点分析：

```text
Absolute Difference
Relative Difference
Trend Consistency
Random Variation
```

---

# 24. 异常结果处理机制

这是项目必须严格执行的部分。

以下情况视为异常：

### Level 1：运行错误

例如：

```text
CUDA OOM
NaN Loss
Checkpoint Load Failure
Dataset Error
Shape Mismatch
```

---

### Level 2：实验异常

例如：

```text
KD < Baseline

LS-KD < KD

Teacher Accuracy 明显异常

Loss 不下降

Accuracy 长期接近随机猜测

不同 Seed 差异异常大
```

---

### Level 3：严重结果偏差

例如原论文：

```text
LS-KD ≈ 76%
```

而复现：

```text
LS-KD ≈ 65%
```

这种情况：

> **禁止直接继续下一阶段实验。**

必须暂停并排查。

---

# 25. Incident Report

所有异常建立：

```text
reports/incidents/
```

命名：

```text
INC_001_teacher_accuracy_mismatch.md

INC_002_kd_underperforms_baseline.md

INC_003_nan_loss.md
```

内容：

```markdown
# Incident

## Incident ID

INC_XXX

## Related Experiment

EXP_XXX

## Date

YYYY-MM-DD

## Severity

Low / Medium / High / Critical

## Problem

描述问题。

## Expected

预期结果。

## Actual

实际结果。

## Difference

实际与预期差异。

## Environment

Python:
PyTorch:
CUDA:
GPU:
Git Commit:

## Configuration

相关超参数。

## Error Log

错误信息。

## Initial Hypothesis

可能原因。

## Investigation

已经进行的排查。

## Resolution

解决方案。

## Status

OPEN / INVESTIGATING / RESOLVED
```

---

# 26. 项目主管者通知机制

出现以下情况必须及时通知项目主管者：

- Teacher 验证失败
- 正式实验无法运行
- 连续出现 NaN
- KD 明显低于 Baseline
- LS-KD 明显低于 KD
- 与论文结果存在重大偏差
- 发现论文/官方代码与描述存在疑似不一致
- 实验数据损坏
- Checkpoint 损坏
- 关键代码实现存在疑问

通知时必须提供：

```text
Incident ID
Experiment ID
Git Commit
Expected Result
Actual Result
Error Log
Initial Diagnosis
```

不要只发送：

> “实验结果不对。”

而应该发送：

```text
INC_002

EXP_005

Expected:
KD > Baseline

Actual:
Baseline = XX.XX
KD = XX.XX

Git Commit:
abc123

Initial Diagnosis:
可能与 Temperature / KL implementation 有关。

Incident Report:
reports/incidents/INC_002_kd_underperforms_baseline.md
```

---

# 27. GitHub Issue 管理异常

建议所有重要异常同时建立 GitHub Issue。

Issue Labels：

```text
bug
experiment
unexpected-result
investigation
resolved
high-priority
```

Issue Title：

```text
[INC_002] KD underperforms student baseline
```

Issue 中链接：

```text
Experiment
Incident Report
Commit
Configuration
```

这样项目主管者可以直接通过 GitHub 查看问题。

---

# 28. 自动异常检测

训练脚本建议自动检查：

```text
Loss = NaN
Loss = Inf
Accuracy abnormal
Checkpoint missing
```

实验完成后检查：

```text
KD vs Baseline
LS-KD vs KD
```

如果明显不符合预期：

```text
STATUS = WARNING
```

并提示：

```text
[WARNING]

Experiment result differs significantly
from expected behavior.

Please create an Incident Report
and notify the project supervisor.
```

自动检测只能负责**提醒**，不能自动认定论文结论错误。

---

# 29. 分析报告

阶段性分析：

```text
reports/analysis/
```

例如：

```text
01_baseline_analysis.md

02_kd_analysis.md

03_lskd_analysis.md

04_multi_seed_analysis.md

05_temperature_analysis.md

06_logit_analysis.md

07_paper_comparison.md
```

---

# 30. Progress Report

每周：

```text
reports/progress/
```

例如：

```text
week_01.md
week_02.md
week_03.md
```

内容：

```text
Completed
Running
Failed
Incidents
Main Results
Problems
Next Week
```

---

# 31. 最终报告

最终：

```text
reports/final/
```

保存：

```text
reproduction_report.md
```

结构：

```text
1. Introduction

2. Reproduction Objective

3. Experimental Setup

4. Student Baseline

5. Vanilla KD

6. Logit Standardization

7. Main Results

8. Multi-Seed Results

9. Temperature Ablation

10. Logit Statistical Analysis

11. Comparison with Paper

12. Reproduction Problems

13. Limitations

14. Conclusion
```

---

# 32. 每次实验前 Checklist

```text
[ ] Git working tree 状态已确认

[ ] Git commit 已记录

[ ] Experiment ID 已创建

[ ] config.yaml 已保存

[ ] Random Seed 已确定

[ ] Teacher checkpoint 正确

[ ] Dataset 正确

[ ] GPU 正常

[ ] Smoke Test 通过

[ ] 实验已登记 registry.csv
```

---

# 33. 每次实验后 Checklist

```text
[ ] metrics.csv 已保存

[ ] train.log 已保存

[ ] best checkpoint 已保存

[ ] summary.md 已填写

[ ] registry.csv 已更新

[ ] Accuracy 已检查

[ ] Loss Curve 已检查

[ ] 与预期结果已比较

[ ] 是否存在异常已确认

[ ] 必要时创建 Incident

[ ] 必要时通知项目主管者
```

---

# 34. 项目阶段 Gate

不要同时开展所有实验。

## Gate 0

```text
Environment
+
Smoke Test
```

通过后：

↓

## Gate 1

```text
Teacher Verification
```

通过后：

↓

## Gate 2

```text
Student Baseline
```

通过后：

↓

## Gate 3

```text
Vanilla KD
```

确认：

\[
KD>Baseline
\]

通过后：

↓

## Gate 4

```text
LS-KD
```

确认：

\[
LSKD>KD
\]

通过后：

↓

## Gate 5

```text
Multiple Seeds
```

通过后：

↓

## Gate 6

```text
Ablation
+
Logit Analysis
```

最后：

↓

```text
Final Report
```

任何 Gate 出现重大异常：

> **暂停进入下一 Gate → 建立 Incident → 排查 → 通知主管者 → 解决后继续。**

---

# 35. 最终实验矩阵

| Experiment | Method | Seed | Temperature | Purpose |
|---|---|---:|---:|---|
| EXP_000 | Teacher Verify | — | — | 验证 Teacher |
| EXP_001 | Baseline | 42 | — | 建立基线 |
| EXP_002 | KD | 42 | Paper Default | KD 核心实验 |
| EXP_003 | LS-KD | 42 | Paper Default | LS-KD 核心实验 |
| EXP_004 | Baseline | 3407 | — | Seed |
| EXP_005 | Baseline | 2024 | — | Seed |
| EXP_006 | KD | 3407 | Default | Seed |
| EXP_007 | KD | 2024 | Default | Seed |
| EXP_008 | LS-KD | 3407 | Default | Seed |
| EXP_009 | LS-KD | 2024 | Default | Seed |
| EXP_010+ | KD / LS-KD | 42 | 1/2/4/8 | Temperature Ablation |

具体 EXP 编号一旦登记后不得重新分配。

---

# 36. 本地一键启动目标

项目最终应支持：

```bash
git clone <repository>
cd lskd-cifar100-reproduction
```

然后：

```bash
conda env create -f environment.yml
conda activate lskd
```

验证：

```bash
python scripts/verify_environment.py
```

Smoke Test：

```bash
python scripts/run_smoke_test.py
```

验证 Teacher：

```bash
python scripts/verify_teacher.py
```

Baseline：

```bash
python scripts/train_baseline.py \
    --config configs/baseline/resnet8x4_cifar100.yaml
```

KD：

```bash
python scripts/train_kd.py \
    --config configs/kd/resnet32x4_resnet8x4.yaml
```

LS-KD：

```bash
python scripts/train_lskd.py \
    --config configs/lskd/resnet32x4_resnet8x4.yaml
```

最终目标：

> **新机器 Clone Repository → 安装环境 → 下载 Teacher → Smoke Test → 启动实验。**

除 Teacher checkpoint 和 CIFAR-100 数据外，不依赖人工复制其他本地文件。

---

# 37. 项目完成标准

## Reproduction

- [ ] Teacher Verification
- [ ] Student Baseline
- [ ] Vanilla KD
- [ ] LS-KD
- [ ] Multiple Seeds
- [ ] Temperature Ablation
- [ ] Logit Analysis
- [ ] Paper Comparison

## Engineering

- [ ] Git Repository
- [ ] README
- [ ] requirements.txt
- [ ] environment.yml
- [ ] GitHub Actions
- [ ] Unit Tests
- [ ] Smoke Test
- [ ] Experiment Registry
- [ ] Config Management
- [ ] Checkpoint Management

## Traceability

- [ ] Experiment ID
- [ ] Git Commit
- [ ] Configuration
- [ ] Random Seed
- [ ] Training Log
- [ ] Metrics
- [ ] Checkpoint
- [ ] Summary

## Project Management

- [ ] Incident System
- [ ] GitHub Issues
- [ ] Supervisor Notification
- [ ] Weekly Progress Report
- [ ] Final Report

---

# 38. 核心原则

整个项目遵循：

```text
先验证
↓
再训练
↓
先 Baseline
↓
再 KD
↓
再 LS-KD
↓
单变量比较
↓
多 Seed 验证
↓
消融实验
↓
统计分析
↓
论文对照
```

实验异常时：

```text
发现异常
↓
停止下一阶段
↓
保存当前现场
↓
建立 Incident
↓
检查 Config / Code / Data / Checkpoint
↓
通知项目主管者
↓
记录排查过程
↓
解决问题
↓
重新建立新的 Experiment ID
↓
重新运行
```

最终要求：

> **任何一个结果都能回答三个问题：**
>
> 1. 它是由哪一版代码产生的？
> 2. 它使用了什么实验配置？
> 3. 它对应的日志、模型和分析在哪里？

如果这三个问题都能够在一分钟内从项目目录和 Experiment Registry 中找到答案，则项目的实验管理达到预期要求。