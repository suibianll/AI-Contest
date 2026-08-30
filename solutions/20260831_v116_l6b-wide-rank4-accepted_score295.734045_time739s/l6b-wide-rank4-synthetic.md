# L6b wide-input rank-4 synthetic validation

- 日期：2026-08-31
- 候选：L6b wide-input compressed cross-block factor
- 根源码规范 LF SHA256：`8fa4db38ac96ca0957e1b1cee61d0c5bd248cf3a4df5d24fa04bedc9239b25f4`
- 父版本：v115 L6a rank-16（规范 LF SHA `043e5401c7d8cf68339e9faec3f60943c11821e3b51bb1563d2ecd8a812f22e5`）
- 运行命令：

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -q tests/test_global_activation_lrh.py tests/test_l5a_joint_transform.py tests/test_linear_compliance_guard.py tests/test_linear_error_decomposition.py tests/test_expansive_cat.py
  ```

- 定向回归：`32 passed in 4.41s`。
- 新增覆盖：`d=2048/4096/4864/8192`，rank=`4`，验证 factor shape、finite、
  channel cap fallback；默认窄路径仍在 `d<=1024`，不会接受宽 factor。
- 一次宽形状校准 probe：`rows=64,d=4864`，`global_lrh.shape=(4864,4)`，
  CPU state tensor count `5`（新增 factor 与既有静态 Gram/部署 Gram 共存，不复制
  第二份 dense Gram），合法五字段 `scale_factor/scale_lv2/scale_lv3/sign/mant` 全部存在。
- 直接 range probe 输出：

  ```text
  2048 (2048, 4) True
  4096 (4096, 4) True
  4864 (4864, 4) True
  8192 (8192, 4) True
  ```

- 结论：synthetic 数值与 state/codec 边界通过，允许进入 Qwen 五层×七 role screen；
  该结果不代表精度增益，screen 仍以 v115 为唯一 parent。
