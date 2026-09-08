# v191 — attn-block-triangular-transport

## 状态

- 机制：在父版本最终 Q/K 坐标中固定搬运两个不重叠的 64 维块对 `0→1`、`2→3`。
- 父 SHA256：`12352efdd4e23cc5e1e17953008664fbaa5ea5d693373635fdafc4d28ce4e24e`
- 候选 SHA256：`eb8c3cebb081546dc17c0f0c6470e3ffcaa60ce672865b617cfb1cddd1ff9666`
- 官方状态：`unregistered/NA`；根 `solution.py` 未替换。

每个块使用父版本最终 Attention 输出误差梯度的最大奇异向量，并取沿该方向第一次触发真实
HiF4 码变化的步长。部署为 `Q_parent*T`、`K_parent*T^{-T}`，其中 `T=I+N`、`N²=0`，
GQA 组内共享；在线路径只执行固定块变换。

## 检查

- `check_math_and_import.py`：PASS（`N²=0`、QK dot invariance、rank-1、六 API、有限输出、隔离导入）。
- `verify.py`：PASS（解析 Attention backward、QK 不变性、强制可达、父回退、状态和输出契约）。

## 4B shard0

- eval-v3 Attention-only、Qwen3.5-4B proxy-v2、CUDA、shard 0：12 cases，
  candidate mean `0.570137778826948`，父 mean `0.5702418426766733`，paired delta
  `-0.0001040638497253`（7 正、5 负），`reasonableness_issues=0`。
- 候选 API total `9.371834s`，其中 calibration `8.940114s`；本地时间和 proxy 只作诊断，
  不换算官方分数/时间，也不构成本地晋级门。

## 校准审计

- `tri_attempted=1`、`tri_accepted=1`、`tri_arm=accepted`、fit windows `0,1,2`、gate windows `3,4`。
- 8/8 个块找到首个码边界并实际可达；边界探针累计 Q 改码 `1`、K 改码 `9`。
- 两个 gate 均通过；窗口 3 的 parent/candidate mean 为 `0.0004609261086443439`，
  窗口 4 为 `0.0005083594151074067`，对应 causal/non-causal 也相同。
- 主奇异值/步长：

  | GQA group | block pair | singular value | step |
  |---:|---:|---:|---:|
  | 0 | 0→1 | `1.44055e-11` | `0.000615625` |
  | 0 | 2→3 | `1.10851e-11` | `0.000858750` |
  | 1 | 0→1 | `3.84086e-11` | `0.001965000` |
  | 1 | 2→3 | `3.29568e-11` | `0.001167500` |
  | 2 | 0→1 | `1.11318e-11` | `0.000703750` |
  | 2 | 2→3 | `1.19626e-11` | `0.000957500` |
  | 3 | 0→1 | `3.24713e-11` | `0.001165000` |
  | 3 | 2→3 | `3.66819e-11` | `0.000229063` |

评测记录：`artifacts/proxy_v3/full_solution/attn-block-triangular-transport-shard0/candidate/manifest.json`。
