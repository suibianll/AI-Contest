# 合法编码复核与最终输出优化执行记录

日期：2026-09-06  
父版本：v186，`solution.py` SHA256 `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`  
输入：`artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt`  
工具：`workbench/legal_codec_output_probe.py`  

## 结论

当前计划按 R0 → R1 → R2 执行至停止条件。没有产生可部署候选，根 `solution.py` 保持
v186；没有分配 v189，也没有重复提交相同 SHA 或未通过门禁的候选。目标
`linear=0.9`、`attention=0.9` 未达到，但本计划没有证据支持在当前固定机制边界内继续实现
候选。

## 阶段结果

| 阶段 | 运行 | 状态 | 关键结果 | 后继 |
|---|---|---|---|---|
| R0 | `r0/run-003` | PASS | 126 E4M3 scale、15 signed E2M1 carrier、1890 原始乘积；原始码本 BF16 逐位差异 0，独立变换 dense 差异 100/128；缺陷回归全过 | R1 |
| R1 | `r1/run-003` | PASS_TO_R2 | 最终坐标 exact-vs-parent encoder 中位 improvement `0`；正层 `0/4`；exact 违规 `0` | R2 |
| R2-L | `r2-linear/run-002` | REJECTED | 12 个 focus state 的 LOO gate `0/12`；holdout mean/median `-0.0006448984/-0.0015624767`；L1 `0.0027736778` | 关闭固定 8 码分支 |
| R2-A | `r2-attention/run-001` | ORACLE_ONLY | Q/K output oracle mean gain `-0.0001224617/+0.0017271334`，median 均为 0；无在线通用规则 | 不进入 R3 |

R2-L 的 35 个 barrier witness 只证明联合候选与单码变化的局部差异，不能抵消 LOO 和
holdout 门失败。R2-A 是 calibration 离线块替换，不是可提交的动态 Q/K 联合搜索。

## 复现命令

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_legal_codec_output_probe.py
.venv\Scripts\python.exe workbench\legal_codec_output_probe.py --stage r0 --parent solution.py --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt --config workbench\legal_codec_output_probe_config.json --output-dir artifacts\proxy_v3\legal-codec-output-20260905\r0\run-003 --device cuda
.venv\Scripts\python.exe workbench\legal_codec_output_probe.py --stage r1 --parent solution.py --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt --config workbench\legal_codec_output_probe_config.json --output-dir artifacts\proxy_v3\legal-codec-output-20260905\r1\run-003 --device cuda
.venv\Scripts\python.exe workbench\legal_codec_output_probe.py --stage r2-linear --parent solution.py --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt --config workbench\legal_codec_output_probe_config.json --output-dir artifacts\proxy_v3\legal-codec-output-20260905\r2-linear\run-002 --device cuda
.venv\Scripts\python.exe workbench\legal_codec_output_probe.py --stage r2-attention --parent solution.py --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt --config workbench\legal_codec_output_probe_config.json --output-dir artifacts\proxy_v3\legal-codec-output-20260905\r2-attention\run-001 --device cuda
```
