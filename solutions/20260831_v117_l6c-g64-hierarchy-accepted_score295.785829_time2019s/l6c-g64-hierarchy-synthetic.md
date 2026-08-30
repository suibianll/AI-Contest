# L6c complete `G_64` hierarchy-coordinate synthetic validation

- 日期：2026-08-31
- 候选：L6c activation-side fixed-scale hierarchy coordinate sweep
- 根源码规范 LF SHA256：`8746b8026495cb56a3dc1d622e463f89226b23e3206e2202bd468f45530d952c`
- 父版本：v116 L6b wide rank-4（规范 LF SHA `8fa4db38ac96ca0957e1b1cee61d0c5bd248cf3a4df5d24fa04bedc9239b25f4`）
- 定向命令：

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -q tests/test_global_activation_lrh.py tests/test_l5a_joint_transform.py tests/test_linear_compliance_guard.py tests/test_linear_error_decomposition.py tests/test_expansive_cat.py
  ```

- 结果：`33 passed in 3.92s`。
- 新增测试覆盖：
  - 固定 `scale_factor`，逐个尝试 `lv2/lv3∈{1,2}`，完整 `64×64` Gram 的精确
    `ΔJ=2eᵀGΔq+ΔqᵀGΔq` 非增门禁；
  - 单 block 一 sweep 与独立坐标暴力参考逐字段/重建值一致；
  - 不超过每行 4 个高损 block，不增加 state 节点，不修改五字段 schema。
- `guard_solution_file('solution.py')`：`violations=[]`、`static_violations=[]`，
  `contraction_count=22`；所有 A@W-like contractions 仍只在离线权重校准人工复核中。
- 结论：L6c 合成数值、合法 hierarchy 写回与退化路径通过，允许进入 Qwen screen；
  该日志尚不代表 full-layer 精度收益。
