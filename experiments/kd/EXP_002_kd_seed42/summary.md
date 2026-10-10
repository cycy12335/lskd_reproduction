# EXP_002：KD 实验总结

状态：completed，240/240 epochs；完成时间：2026-10-09 10:09:45（UTC+8）。

最佳测试 Top-1：73.34%（epoch 183），对应 Top-5：92.65%。
最终测试 Top-1：72.64%，Top-5：92.35%。

相对 Baseline：最佳 Top-1 +0.81 个百分点，最终 Top-1 +0.41 个百分点。该结论限于 seed 42 和当前配置。

完整性、续训连续性、checkpoint 与 SHA256 检查通过，错误日志为空，Gate 3 通过。前 120 个 epoch 及暂停现场均保留。

完整报告：reports/analysis/02_kd_analysis.md。
机器核查：reports/validation/EXP_002_completion_audit_20261009.json。
下一阶段：相同参数、仅开启标准化的 EXP_003 LS-KD。
