# 修正版合法离散网格与输出目标优化执行记录

日期：2026-09-06  
父版本：根 `solution.py` v186，SHA256 `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`  
输入：`artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt`  
工具：`workbench/legal_codec_output_probe.py`、`workbench/corrected_legal_lattice_oracle.py`

## 结论

本计划按 R0 → R1 执行至停止条件。R0 的参考 codec 与五字段回归全部通过；R1 在真实
NVFP4 输入、最终部署输出目标和合法共享层级约束下没有材料余量，状态为
`NO_SUPPORTED_MECHANISM`。不进入 R2，不产生候选版本，不提交官方；根 `solution.py`
保持 v186。下一步切换到独立的 Attention source-scale proposal 机制。

## 阶段结果

| 阶段 | 运行 | 状态 | 关键结果 | 后继 |
|---|---|---|---|---|
| R0 | `artifacts/proxy_v3/corrected-legal-lattice-20260906/r0/result.json` | PASS | 126 个 E4M3 scale、15 个 signed E2M1 carrier、1890 个原始乘积；BF16 原始码本差异 0；变换 dense 差异 100/128；5 组回归全过 | R1 |
| R1 | `artifacts/proxy_v3/corrected-legal-lattice-20260906/r1-fixed-panel/result.json` | NO_SUPPORTED_MECHANISM | 28 个 state、112 个 holdout case、14,622,720 个固定合法模式；LOO mean/median `-0.0031395/-0.0002933`；holdout mean/median `-0.0003232/-0.0000542`；L1 `0.0004041`；正 case `10/112`；四层 median 全负 | 关闭 |

R1 的每个 64 元素块只使用一个共享 E6M2 scale，并枚举合法 lv2/lv3 与 0.25 mantissa
舍入模式；输出目标使用实际 cache 的固定父激活输出和真实部署形状。它不是旧 cb1/cb2
自定义解码器，也没有用 `A@W` 反推激活 state。R1 结果因此只关闭本次固定联合离散
oracle，不把旧错误结果或本次 oracle 解释为整个 HiF4 空间的格式证明。

## 复现命令

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_legal_codec_output_probe.py
.venv\Scripts\python.exe workbench\legal_codec_output_probe.py --stage r0 --parent solution.py --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt --config workbench\legal_codec_output_probe_config.json --output-dir artifacts\proxy_v3\corrected-legal-lattice-20260906\r0 --device cuda
.venv\Scripts\python.exe workbench\corrected_legal_lattice_oracle.py --parent solution.py --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt --output-dir artifacts\proxy_v3\corrected-legal-lattice-20260906\r1-fixed-panel --layers 0 8 15 23 --roles q k v o fc_gate fc_up proj --rows 2 --blocks 1 --token-limit 128 --calibration-windows 0 1 --holdout-windows 1 2 6 7 --device cuda
```

## 归档状态

- 状态：`CLOSED / R1_NO_SUPPORTED_MECHANISM`。
- R2 未执行，未分配版本号，未改变根 `solution.py`。
- 当时的后继计划现已归档为 `docs/superpowers/archive/plans/2026-09-06-attention-source-scale-proposal-plan-rejected.md`。
