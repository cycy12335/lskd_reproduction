# GitHub 发布说明

记录时间：2026-10-10T11:10:27.193352+08:00。仓库：https://github.com/cycy12335/lskd_reproduction；可见性：私有；分支：main。

## 发布内容

保留现有 Git 历史，发布训练与验证代码、配置、环境锁定依赖、Teacher 来源、实验指标与元数据、训练日志、中文阶段报告、完整性核查和续训归档。README 汇总 seed 42 核心对照，并说明方法论复现范围、概率质量现象和后续验证限制。

已完成 Teacher（EXP_000）、seed 42 Baseline/KD/LS-KD（EXP_001/002/003）及 seed 3407 Baseline（EXP_004）。核心最佳 Top-1 为 72.53%、73.34%、76.51%；最终 Top-1 为 72.23%、72.64%、75.96%。原始已完成实验数据保持不变。

## 进行中的记录

发布时 EXP_006 仍在本地训练；其 metadata、metrics、日志、registry 及队列 state 纳入时间快照，后续本地文件可能继续更新。快照不等同于最终验收；其余队列任务以实时状态为准。此次未停止、重启或修改正式训练。

## 发布检查

扫描 160 个候选文件及现有 Git 历史：未发现超过 10 MiB 的候选文件或历史 blob，未发现 checkpoint、数据集、虚拟环境、本地工具进入候选集合，未发现 GitHub token 或私钥特征。此检查只核对发布范围，不代表重新执行训练或全测试集推理。实验完成性证据见既有 validation 与 analysis 报告。

GitHub Actions 配置仅包含 CPU 环境验证、单元测试和合成数据 CPU smoke test，无正式 CIFAR-100 GPU 训练任务。远程运行结果以 GitHub Actions 为准。

大型 checkpoint、Teacher 权重、原始数据集、虚拟环境、实时 .tools 锁及控制台日志继续保留本地。历史暂停归档中的锁和启动信息作为溯源材料保留，不代表当前活跃状态。
