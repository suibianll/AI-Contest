# v028 — C28 Activation Scale-Code Refinement (REJECTED BY FEASIBILITY PROBE)

- Date: 2026-08-27
- Candidate ID: `C28`（预注册见执行日志同日条目）
- Parent: `C21-C` / v025，SHA256
  `83AB4864254F80D221BB491BDEF89F8C9AB8E83534FD62D4DD5E0C1C292FEA12`
- Source SHA256: 无——**未产生任何 solution.py 改动**（实现前被探针否决）
- Local status: `rejected`（feasibility-probe rejection，0 代码变更）
- Official status: `unavailable`

## 预注册机制（未实施）

`hif4_dynamic_quantize_activation` Linear 路径上对每 64 块做层级
scale 码字（`scale_factor` E6M2 码 × `scale_lv2`{1,2} × `scale_lv3`{1,2}）
的批量坐标下降/beam 精修，目标为 A 自身块重构误差（operand-local）。

## 可行性探针（evaluator 侧，实施前）

对每个 64 块求**合法码空间内的精确最优**（穷举）：`scale_factor`
扫 E6M2 网格（amax/7 ±4 个 octave × 4 个 mantissa 步），固定 s 后
lv3 按子组、lv2 按 8 组精确解耦取最优（选择可分解性使穷举在张量
批量下可行），mantissa 取最近合法码。该 oracle 是任何 scale 码字
精修在"变换坐标系无权激活误差"指标上的**严格上界**。结果（真实
GPT-2 数据，amax6，与此前诊断同一套变换坐标系）：

```text
 comp  player_rel  code_oracle  headroom
    q     0.07682      0.07533     1.95%
    k     0.07854      0.07462     4.99%
    v     0.07562      0.07498     0.84%
    o     0.07613      0.07390     2.93%
   fc     0.07682      0.07568     1.47%
 proj     0.08350      0.07810     6.46%
TOTAL energy ratio (oracle/player): 0.9190
```

## Decision

`rejected per preregistered gate 3`。门 3 要求激活误差能量降幅
≥15%，而穷举合法码空间的精确上限仅 **8.1% 能量（~4% relRMSE）**。
结合预注册依据的修正：

1. **player 激活码字已在合法码空间内接近最优**（headroom 0.8~6.5%
   relRMSE）——现有 search_offsets + quadratic8 精修已基本榨干
   scale 码字机制。
2. 此前诊断的 26~38% per-4 自由 scale 余量是**格式限制**而非拟合
   缺口：lv2/lv3 码字只有 {1,2}，合法组合无法逼近任意连续 per-4
   scale。激活侧残差的下界由格式决定，不由拟合决定。
3. 因此 C28 的唯一机制（scale 码字精修）天花板 ~8% 能量 < 15% 门，
   实现只会浪费 CPU 预算（C23 教训），按纪律在实现前否决。
   根目录 solution.py 无任何改动（git 无 diff）。

## 遗留结论（供后续候选）

- 激活侧 Linear 残差（占 ~50%）在"HiF4 码字 + 现有变换"框架内
  已近极限；进一步提升需要改变**变换坐标系**（更强的平滑/旋转/
  置换，C22 类）或**两侧联合**机制，或等待官方确认更宽格式解释。
- 权重侧（C23 类 Hessian 精修）同样受 CPU 预算约束。两侧边际
  收益均 <2pp，22000~25000 主目标在当前合规框架下缺乏已验证的
  可行路径；Champion 维持 C21-C（v025），官方提交以建立锚点
  （预期低于 16043）是当前最高价值动作。
