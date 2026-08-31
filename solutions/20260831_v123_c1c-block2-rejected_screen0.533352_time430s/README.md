# v123 C1c max-blocks-2 rejected

- 状态：screen rejected；只改变 `_ACT_STRUCTURED_LRH_BLOCKS: 4→2`，其余保持 v121（rank=4、sweep2）。
- source `solution.py` SHA256（规范 LF）：`b498a6d5cf025b58807c9751d0e98e237f0ea9d5bdf2a11fc5be8f2aa7ae1801`。
- Qwen 五层×七 role screen：Linear mean `0.53335171`，较 v121 `0.5333964596`（`−0.00004475`），也低于 v118 `0.5333753185`。
- `proj` role 为 `0.41326755`，低于 v121 `0.4135807679`；其余 role 基本持平。
- elapsed `429.949s`（screen）；未进入 full-layer。
- 结论：减少 selected block 会损失跨 block proposal 覆盖，保留 v121 `max_blocks=4`，不扩大该组合。
