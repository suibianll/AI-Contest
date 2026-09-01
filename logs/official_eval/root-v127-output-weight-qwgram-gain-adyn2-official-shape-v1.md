# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `16.990`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| root-v127-output-weight-qwgram-gain-adyn2 | ok | 0.483610 | 0.834256 | 28775.385 | 365.818 | False | 397.341 | False | unregistered |

> **时间质量更正（2026-09-01）**：该次运行时有其他程序同时占用机器；API `365.818s`
> 与 wall `397.341s` 是受干扰观测，仅保留原始记录，不用于超时判定或候选排序。需要在
> 机器空闲时重跑后，才能形成可比较的本地时间观测。
