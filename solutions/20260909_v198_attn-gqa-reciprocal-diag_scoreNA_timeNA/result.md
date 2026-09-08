# v198 — attn-gqa-reciprocal-diag

## 状态

- 机制：GQA 组共享互逆对角重参数化（设计文档 §8–§16）。每 KV group、每 channel 一个 `u_j`，
  group 内全部 Q head 共享 `D_g=diag(e^u)`；解析初始化 `u=½log((k+eps)/(q+eps))`（clamp
  ±log2），Stage A 以 τ=8、λ=1e-3 的 64-block smooth-max range 目标手写梯度 5 步（无
  Attention 前向），Stage B 在最后一个留出校准窗口用真实编码+真实前向比较父/候选最终输出
  MSE，严格改善才接受，否则回退父 state。严格冻结 V；K learned center 同步乘 D⁻¹。
  校准增量成本：一次统计 pass + 5 步小梯度 + 2 次门控前向，远低于 v190–v192 的 32 步训练。
- 父（当前根 v195）SHA256：`839adb1e617c3115c6b55071a34b281c5db0ff2aa070adbbc71fd1549e761d7f`
- 候选 SHA256：`8486773e0120a78a653e02543aef60a503f8e508d646a0cc8d5b6f2fe4f30f84`
- 官方状态：`unregistered/NA`。

## 检查

- `verify.py` 全 PASS：分组 QK^T 互逆误差 2.0e-7；K-center 同步编译精确；spike 数据上门控真实
  接受（`rd_attempted=1, rd_accepted=1`，父 MSE 9.714e-4 → 候选 9.432e-4）；退化数据 u≡0 精确
  回退父；u≡0 control 与父逐位一致；合法 state、有限输出、脱离仓库六 API 导入通过。

## 4B shard0

- eval-v3、Qwen3.5-4B proxy-v2、CUDA、Attention-only、shard 0、calibration-cache-mode write：12 cases。
- candidate mean `0.5704842850`，父 mean `0.5721770802`；paired delta mean **−0.0016928**、
  median −0.0002091、min −0.0254413、max +0.0106140，+/-/0 = `6/6/0`，L1 `0.006240`；
  `reasonableness_issues=0`。非 no-op、无运行错误；本地小幅正负不用于挑参数，交官方裁决。
- calibration API：父 `5.947389s` → 候选 `5.003227s`（同机噪声水平，无超时风险信号）。

## 证据位置

- 归档：`solutions/20260909_v198_attn-gqa-reciprocal-diag_scoreNA_timeNA/`
- 评测：`artifacts/proxy_v3/full_solution/attn-gqa-reciprocal-diag-shard0/candidate/manifest.json`
- 工作脚本：`workbench/full_solution/attn-gqa-reciprocal-diag/`

## 官方结果

- `unregistered/NA`。候选非 no-op、合法、可达，可按当前规则（新 Attention 机制与标准 Linear
  配对的单次诊断提交）交官方裁决。
