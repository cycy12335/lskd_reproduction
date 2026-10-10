# EXP_001：Baseline 实验总结

方法：Baseline。状态：completed（已完成），240/240 epochs。

最佳测试 Top-1：72.53%（epoch 239）；对应 Top-5：92.92%。
最终测试 Top-1：72.23%（epoch 240）；对应 Top-5：92.87%。

完整指标、源码指纹、配置哈希、checkpoint 结构与 SHA256 检查通过，Gate 2 通过。未发现训练异常。此为 seed 42 的单次最终结果，不代表全项目复现完成。

完整中文报告：reports/analysis/01_baseline_analysis.md。
机器核查：reports/validation/EXP_001_completion_audit_20261008.json。

下一步：解决 INC_002 的 KD 论文对照协议溯源，再推进 KD。未启动 KD 或 LS-KD。原始 metadata.json、metrics.csv、train.log、error.log 和 checkpoint 保持不变。