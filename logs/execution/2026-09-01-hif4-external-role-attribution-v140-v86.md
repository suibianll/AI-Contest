# hif4 外部逐 role 归因：v140 Linear 相对 v86

- 日期：2026-09-01
- 外部仓库：<https://github.com/youxilee/hif4>
- 外部脚本版本：commit `dd5ee6515323169dbd4133b3d4fd1ff1cb7be646`
- 模型：本地 GPT-2 small，12 layers，hidden 768，12×64 MHA
- 命令：

  ```powershell
  .\.venv\Scripts\python.exe -u real_data_eval.py `
    --layers 12 --seq 128 --calib 2 --test 2 `
    --mode amax6 --config current
  ```

`real_data_eval.py` 的 Linear `q/k/v` 指 GPT-2 fused `c_attn` 拆出的三个**投影层**，
不是 Attention 输出里的动态 Q/K/V 控制臂；后者单独汇总为 Attention mean。

## v140 − v86 外部 role 差分

基线逐 role 均值为：

| 版本 | q | k | v | o | fc | proj | Attention |
|---|---:|---:|---:|---:|---:|---:|---:|
| v86 | 0.6221 | 0.6341 | 0.6133 | 0.5578 | 0.5558 | 0.5373 | 0.4727 |
| v140 | 0.6630 | 0.7241 | 0.6218 | 0.5560 | 0.5107 | 0.5221 | 0.4661 |
| Δ | +0.0409 | +0.0900 | +0.0085 | −0.0018 | **−0.0452** | **−0.0153** | −0.0066 |

逐层符号统计（12 层）：

| role | 正向层数 | 负向层数 | 最大负差 | 结论 |
|---|---:|---:|---:|---|
| q | 12 | 0 | — | 稳定改善 |
| k | 12 | 0 | — | 稳定改善 |
| v | 11 | 1 | −0.0007 | 基本中性略正 |
| o | 5 | 7 | −0.0648（L5） | 局部回归，均值近中性 |
| fc | 0 | **12** | −0.0575（L10） | **全层系统性回归** |
| proj | 6 | 6 | −0.1634（L2） | **混合但有严重层级回归** |

这说明 v140/v147 的外部 Linear 负担不在 q/k/v；第一嫌疑是 fc，第二嫌疑是 proj，o
只应作为局部异常处理。

## 角色化临时消融

以下均为 v140 的临时副本，只改变一个指定形状/调用槽位，其他参数与上表相同；结果仍是
hif4 外部脚本的候选私有标准 codec 分数，不能当官方分数。

| 消融 | q | k | v | o | fc | proj | Attention | 观察 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v140 baseline | .6630 | .7241 | .6218 | .5560 | .5107 | .5221 | .4661 | — |
| 关闭 proj 的 ROAB（rows < channels） | .6630 | .7241 | .6218 | .5560 | .5107 | **.5430** | .4661 | proj `+0.0209`，直接确认 ROAB 对 proj 有害 |
| 关闭 fc 的 ROAB（rows > channels） | .6630 | .7241 | .6218 | .5560 | .5107 | .5221 | .4661 | 无变化 |
| 关闭 fc 的 CAT balance | .6630 | .7241 | .6218 | .5560 | **.5144** | .5221 | .4661 | 只恢复 `+0.0037` |
| 跳过 fc 的 pre-output HSDQ | .6630 | .7241 | .6218 | .5560 | .5107 | .5221 | .4661 | 无变化 |
| 跳过 fc 的 output cross-fold | .6630 | .7241 | .6218 | .5560 | .5107 | .5221 | .4661 | 无变化 |
| 关闭 fc 的 BOAT | .6630 | .7241 | .6218 | .5560 | **.4599** | .5221 | .4661 | BOAT 对 fc 有益，不能整体删除 |
| 关闭 o 的 ROAB（第 4 个 square 调用槽） | .6630 | .7241 | .6218 | **.5512** | .5107 | .5221 | .4661 | ROAB 对 o 有益，o 回归来自其他路径 |

因此不能用“关闭一个 refine 开关”修复 fc。fc 需要保留 BOAT 的结构收益，同时重做其
expansive 编码/scale 决策；proj 可以先去掉 ROAB，作为明确的负向控制。

## 与其他本地评测的冲突

- Qwen `proxy-v2` 把 v140 的七个 Linear role 都判为上升，但同一批官方锚点的排序出现
  3/6 反转，不能用它选择 v140 的角色路径。
- 独立 `cross_model_eval.py` 使用 WikiText、评测器自有标准 codec 和非 causal full-softmax，
  也把 v140 的所有 GPT-2 role 判为上升；这与 hif4 外部脚本冲突，原因是数据、Attention
  公式和分母均不同。
- 外部 hif4 证据最适合回答“历史 v140 代码在 GPT-2 causal real-data 下哪个 role 退化”，
  但不证明官方隐藏 case 的绝对权重。因此角色结论按置信度使用：fc 高、proj 中高、o 低，
  q/k/v 不作为第一修复目标。

## 决策

1. Linear 下一轮冻结 q/k/v；不再对 square qkv 统一替换编码器。
2. proj 先执行“ROAB off”负向控制，再测试解耦 encoder；只接受跨 fold/跨层不回归的方案。
3. fc_gate/fc_up 保留 BOAT，针对 expansive shape 单独设计 Activation-first encoder/scale
   teacher；不要整体禁用 BOAT。
4. o 暂不改动；只有出现跨模型、跨 fold 的稳定负差才新增 O 专属路由。
5. Attention 动态 Q/K/V 继续保持 v86 的 Q/K 配对路径，V 不引入新搜索。
