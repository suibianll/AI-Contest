# 当前规则与候选身份清理（2026-09-08）

本记录修正当前入口和状态，不改写历史 JSON、原始评测报告或历史裁决。

## 统一规则

1. 优先级固定为：`AGENTS.md` → `docs/4b-panel-testing-guide.md` → 唯一活动计划及当前工作包
   → `workbench/*/state.json` / `queue.md`。状态文件不得新增门禁。
2. 全部新测试只使用 Qwen3.5-4B。Linear 直接评价全校准数据合法部署 fit；独立窗口只记录。
   Attention 的 4B paired 只作合法性、可达性和风险判读，不换算或预测官方分数。
3. 官方 300s 是唯一时间硬限。本地 API 秒数、按层成本和父官方时间只标记 `time-risk` 和安排
   降时优先级，不恢复 `<280s` 或按层外推提交门。
4. 官方结果必须绑定实际计分 SHA；骨架、复制目录和真实机制实现分别登记，不继承结果。

## 身份清理

- 根运行父：v189 `17616/275s`，SHA `26120224...17AF`。
- compiled sample-energy：用户回传 `17636/264s`，归档 SHA `D66128A6...B0F6`；官方计分 SHA
  尚未单独核验，状态统一为 `REPORTED_BETTER / IDENTITY_PENDING`。核验前不替换根父。
- Linear：L28 `4611/286s` 为 score parent，完整 fit `0.948587`；L4 `4607/247s` 为
  time reference。下一步先完成上述完整候选身份核验，再在 L31/L32 中只注册一张。
- Attention：A2 `14440/274s` 为 score parent，R3 `14405/238s` 为 time parent，AC0
  `14395/258s` 为 correctness reference。`continuous_attention_a29-boundary-output` 是 AC0
  逐位骨架；真实 A29 `D8BAAB46...126A5` 官方 TIMEOUT。当前下一卡为 A30。

## 历史冲突的处理

- `2026-09-08-a29-official-results.md` 中新增的“外推 >280s 拦截”只保留为当时记录，当前无效；
  A29 超时仍是有效官方事实。
- `2026-09-08-la0a1-stop-diagnosis.md` 中以独立窗口退化否决 Linear 的做法不适用于当前 Linear
  工作包；其中 fit、合法投影和成本数据仍可作为诊断证据。
- 旧计划移入 `docs/superpowers/archive/plans/`，活动目录只保留 README 和唯一活动总计划。
