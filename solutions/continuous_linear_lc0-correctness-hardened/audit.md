# LC0 Linear correctness audit

日期：2026-09-08

LC0 是从 `solutions/continuous_linear_l28-proj-vectorized/solution.py` 复制出的独立候选。L28 原文件、根 `solution.py` 和其他历史归档未修改。

父源码 SHA256：

```text
44D7E964F82646331E4509996FFFEAB84E4FE0B3870528D45F25ED953EC93AB5
```

候选源码 SHA256：

```text
E702D7C1F087097906E752DEFA5CBFF05C2E41585883B0EC8D5E4F937AA4AF3D
```

## 修复内容

1. L23 A@W 校准目标改为 evaluator 同口径的逐 case `MSE / MSE_STD` 等权目标。每个 fold 的残差权重为 `1 / (F * numel(Y_f) * MSE_STD_f)`，不再使用 `1 / ||Y_f||²`。
2. 校准期间 activation 解码形状与 NVFP4 输入不一致时立即抛错，删除危险的自动 `reshape`。
3. L28 static activation GPTQ 路径删除静默 fallback；非法 block order、缺失 `h_inv` 或内部补偿求解失败现在显式报错。
4. L23 写回先生成 canonical signed code，再生成 mantissa/sign，保证 `mant == 0` 时 `sign == 0`。
5. 将 L28 的 `torch.round` 投影改为无网格 materialization 的 `ceil(z - 0.5)`，保留旧显式升序 legal grid 的 first-on-tie 语义，包括负半格边界。

## 校验结果

- 六个公开 API：导入检查 PASS。
- `py_compile`：PASS。
- `tests/test_reference_hif4.py` 与 `tests/test_static_activation_gptq_order.py`：`10 passed`。
- `correctness_battery.py`：PASS，覆盖 half-grid tie、非法 GPTQ state、shape mismatch、same-input replay、zero-code、五字段解码布局、MSE_STD 目标等价式和增量残差等价式。
- LC0 标准 HiF4 编解码与 evaluator reference：随机 3x128 输入 `max_abs = 0`，逐元素一致。
- 当前 4B `proxy-v3` Linear shard0：候选 `status=ok`，56 cases，无 reasonableness issue；校准缓存命中，scoring API 42.583s。

## 结果边界

4B shard0 的 LC0 local proxy mean 为 `0.2884021593`，同口径 L28 父结果为 `0.2910968103`，paired delta 为 `-0.002695`，L1 为 `0.008212`。该结果仅用于本地正确性/风险诊断，不能换算或宣称为官方分数；因此 LC0 不晋级为新的官方父版本，官方状态保持 `unregistered/NA`。
