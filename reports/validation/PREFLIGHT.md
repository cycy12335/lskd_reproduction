# 正式实验启动前验证报告

日期：2026-10-08。

## 单元测试

执行 `.venv/Scripts/python.exe -m pytest -q`，12 项测试全部通过，耗时 4.67 秒。测试覆盖蒸馏公式、标准化性质、Teacher 冻结、检查点往返及实验 ID 不可复用等行为。

## 端到端 smoke test

Baseline、KD、LS-KD 均通过 CPU 微型合成数据测试，以及本地 CUDA 上真实 CIFAR-100 的完整管线测试。GPU 测试使用 batch size 64，每种方法训练 2 个 batch（128 张图像），再评估 128 张测试图像。

验证确认：Student 参数更新、梯度有限、Teacher 保持冻结、评估正常，且未发生显存溢出。峰值张量分配显存为 374,268,928 字节；该值不包含全部驱动和运行时保留显存。

原始记录：`smoke_cpu_fake.json`、`smoke_cuda_cifar100.json`。

源码指纹：`b8c6fa0e7d8b14068f72b3538fc4712865375efa1fca1bc8eecba956c5ccf3ff`。

## Teacher 验证

EXP_000 在完整的 10,000 张 CIFAR-100 测试图像上取得 Top-1 79.42%、Top-5 94.58%，测试损失为 0.8833263866424561。Top-1 与预期 79.42% 一致。

Gate 0（环境与 smoke test）和 Gate 1（Teacher 验证）均已通过。

正式 GPU 训练仅在本地执行；GitHub Actions 仅执行 CPU 单元测试和小规模 smoke test。
