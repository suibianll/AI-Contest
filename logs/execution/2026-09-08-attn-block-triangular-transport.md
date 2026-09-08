# 2026-09-08 `v191 / attn-block-triangular-transport`

## 结论

候选从当前完整根直接构建，严格实现计划方向二：固定 `0→1`、`2→3` 两个 64D 非重叠块对，
每块采用最终 Attention 输出误差梯度的最大奇异向量和首个真实 HiF4 码边界步长；Q 使用
`T=I+N`，K 使用 `T^{-T}`。数学、六 API、状态合法性、父回退和可达性检查均通过。

4B eval-v3 Attention shard0 无实现异常。8/8 块可达且 gate 接受，候选与父方案的 paired proxy
delta 为 `-0.0001040638497253`（7 正、5 负），因此该值只作为诊断记录；根保持不变，官方状态
为 `unregistered/NA`。

## 固定配置

- 父 SHA256：`12352efdd4e23cc5e1e17953008664fbaa5ea5d693373635fdafc4d28ce4e24e`
- 候选 SHA256：`eb8c3cebb081546dc17c0f0c6470e3ffcaa60ce672865b617cfb1cddd1ff9666`
- block pairs：`0→1`、`2→3`；block dim `64`；head dim `256`。
- fit windows：`0,1,2`；gate windows：`3,4`；候选数量 `1`。
- 首码边界：初始步长 `1e-5`、固定增长 16 次、固定二分 10 次。

## 执行结果

```text
check_math_and_import.py: PASS
verify.py: PASS
eval-v3 shard0: protocol ok, reasonableness_issues=0
tri_attempted=1, tri_accepted=1, tri_reachable_blocks=8
```

候选 API total `9.371834s`，父 API total `0.745168s`；本地时间仅作诊断，不作为官方时间结论。
候选已归档于 `solutions/20260908_v191_attn-block-triangular-transport_scoreNA_timeNA/`，
待官方提交与回传后再裁决是否替换根。
