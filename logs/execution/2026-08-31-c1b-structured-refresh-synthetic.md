# C1b structured gradient refresh — synthetic and screen decisions

> 日期：2026-08-31
> parent：v119 C1a structured proposal vectorization

## 实现与合成检查

- `block` 模式：按初始 top-k block 顺序处理，每完成一个 block 就重新计算冻结
  structured `R e` proposal gradient；最终仍用完整部署 `G_q` row gate。
- `sweep2` 模式：重复上述 block-refresh sweep 两轮；不改变 HiF4 五字段、state
  来源或 exact gate。
- 随机宽层合成（seed `716`，`dense=[4,256]`、`deployment=[96,256]`、2 blocks）
  两个模式均 finite，且完整部署二次型逐行不增；C1b 定向集合共 `38 passed`。

## Qwen screen

固定 cache、层 `[0,5,11,17,23]`、七 role、CPU：

| 变体 | Overall Linear mean | vs v118 | proj | elapsed | 裁决 |
|---|---:|---:|---:|---:|---|
| per-block refresh | `0.5333730058` | `−0.0000023127` | `0.4134165913` | `419.630s` | rejected |
| two refresh sweeps | `0.5333964596` | `+0.0000211411` | `0.4135807679` | `431.513s` | enter full |

Per-block 版本在 `proj` 回退，不能进入 full；两轮版本仅在 Qwen screen 通过后进入
full-layer，不能从 screen 直接推断最终 panel。

