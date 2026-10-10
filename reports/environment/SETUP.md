# 项目环境配置验证报告

日期：2026-10-08（Asia/Shanghai）。项目根目录：`D:\reproduction`。

本报告记录首次环境配置时的验证快照；当前实验进度以 `experiments/` 中的实验元数据为准。

## 环境与硬件

- 项目独立 `.venv`，CPython 3.10.20。
- PyTorch 2.8.0+cu126、torchvision 0.23.0+cu126。
- CUDA 运行时 12.6，cuDNN 版本号 91002，NVIDIA 驱动 592.82。
- 实测显卡：NVIDIA GeForce RTX 3060 Laptop GPU，6144 MiB 显存。

原计划假定 8 GB 显存，而本机实际为 6 GB。后续已通过 batch 64 的完整 GPU smoke test，无需因此降低 batch size。原始计划保存在 `PROJECT_PLAN.md`。

## 首次环境验证

`python -m pip check` 未发现依赖冲突。环境检查脚本通过依赖导入、CUDA 矩阵乘法、卷积前向和 torchvision NMS 检查。初始环境单元测试为 2 项通过，耗时 6.75 秒；后续训练实现后的测试已扩展为 12 项，见 `reports/validation/PREFLIGHT.md`。

Python 和 PowerShell 脚本通过语法检查，环境及 workflow YAML 通过解析。已验证数据、checkpoint、虚拟环境和本地 Python 工具均被 Git 忽略；Teacher 来源说明可纳入版本管理。

精确依赖见 `requirements-lock-windows-gpu.txt`，机器可读环境记录见 `reports/environment/local.json`。

## 安装与运行边界

使用 uv 可执行 `powershell -File scripts/setup_environment.ps1` 重建环境。`environment.yml` 提供 Conda 替代方案，但本地未实际创建该 Conda 环境。

首次环境配置阶段未下载数据或 Teacher、未启动正式实验。后续已经补齐数据与训练管线并通过 Gate 0、Gate 1；这些后续状态记录在预检报告和 EXP_000、EXP_001 中。

GitHub Actions 仅使用 CPU 依赖执行检查、单元测试和小规模 smoke test；正式 GPU 实验始终在本地运行。远程 workflow 尚未执行，尚未配置 GitHub 远程仓库。

官方 PyTorch 配套版本和 CUDA 索引说明：https://pytorch.org/get-started/previous-versions/
