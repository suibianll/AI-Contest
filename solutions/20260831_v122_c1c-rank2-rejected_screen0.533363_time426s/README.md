# v122 C1c structured rank-2 rejected

- 状态：screen rejected；只改变 `_ACT_STRUCTURED_LRH_COMPONENTS: 4→2`，其余保持 v121（sweep2、`max_blocks=4`）。
- source `solution.py` SHA256（规范 LF）：`86402f6e73b44acf3cb100142e17c34d4c94dc992e4d6e4080974accb1220432`。
- Qwen 五层×七 role screen：Linear mean `0.53336284`，较 v121 `0.5333964596`（`−0.00003362`），也低于 v118 `0.5333753185`。
- `proj` role 为 `0.41334546`，低于 v121 `0.4135807679`；其他 role 基本持平。
- elapsed `425.699s`（screen）；未进入 full-layer。
- 结论：rank=2 的结构化 kernel 表达能力不足，保留 v121 rank=4 parent，不扩大该 rank-2 组合。
