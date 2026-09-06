# candidate/ 目录说明

本目录存放 L 侧候选源码。

- **L0 零点候选** = `../baseline/solution.py`（v162 标准文件，SHA
  `56101559D267D962084CD67A9F9AF8EB924501B17AB408EAF676081876CC000A`）。
- **L1 候选（l1-v160-stack-recovery）**：工作区安全钩子禁止用 Bash 复制 `.py`
  源文件，而该文件 414KB 无法经 Write 工具整篇落盘；因此 L1 评测以归档源码
  **原位只读引用**作为候选：
  `solutions/20260903_v163_v160-linear_standard-attn_scoreNA_timeNA/solution.py`。
  评测 manifest 会记录实际路径与 SHA256；运行前先核验 SHA 与归档一致。
  L2 起的自研候选将直接以 Write/Edit 落盘为 `solution.py` 或带版本号的
  `lN_*.py`，不再使用原位引用。
