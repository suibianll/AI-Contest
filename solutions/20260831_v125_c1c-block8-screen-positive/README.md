# v125 C1c rank-8 / max-blocks-8 screen 归档

日期：2026-08-31  
父版本：v124 C1c rank-8 / max-blocks-4  
状态：**screen positive；已进入 Qwen full，正式结论以 full 归档为准**

固定 Qwen2.5-0.5B、五层 `{0,5,11,17,23}`、七个 Linear role、`seq=128`、
`calib=2`、`test=4`、`amax6`、CPU、只读 cache。screen Linear mean 为
`0.53358298`，较 v124 screen `0.53343639` 提升 `+0.00014659`，无 role 回退，
因此按 active plan 晋级 full。full 结果见
[`v125 precision-only archive`](../20260831_v125_c1c-block8-precision-only_score295.847849_time2654s/)。

本目录保存 screen JSON 与当时完整源码；规范 LF source SHA256：
`c9b419717e38bcec69d907d1cab6638409f1fa9a3072892dde9494ef9da3cc8e`。
