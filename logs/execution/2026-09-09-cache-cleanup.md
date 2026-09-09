# artifacts/official_eval/cache 清理（2026-09-09）

## 背景

用户反馈 `artifacts/official_eval/cache` 空间占用过大。只读扫描结果：

| 项 | 清理前 |
|---|---|
| `cache/proxy-v3-calibration/` | 671.79 GB / 797 个 `.pt` |
| `cache/qwen3.5-4b-proxy-v2.pt`（+`.panel.json`） | 6.28 GB |
| **cache 合计** | **678.07 GB** |
| `artifacts/` 整体 | 约 694 GB（cache 占 99.1%） |

## 为什么会长到 672 GB

校准缓存键 = `<solution SHA256 前16位>-<linear|attention|both>-<配置哈希>.pt`，
**按 solution 源码身份命名**。每个新候选的 Linear 六 shard 写约 5.6 GB，Attention 侧约
2 MB×6。几十个已废弃候选累积后即达数百 GB。最大的单个死候选 v189（`261202248a0146a2`）
独占 44.97 GB。

死候选的缓存键**永远不会再次命中**；缺失时 `--calibration-cache-mode auto` 会自动重建，
只是多花时间。因此删除是安全的。

## 清理前核对（三项，全部通过才动手）

1. `cache/` 被 `.gitignore:19` 忽略 → 删除不影响 git 历史与任何证据（`git status` 无变化已验证）。
2. 保留前缀确实命中：`56dc805d6e5a3aef`（当前根 v202）6 个、`839adb1e617c3115`（回退根 v195）12 个。
3. dense 主缓存 `qwen3.5-4b-proxy-v2.pt` 存在且不在删除范围。
4. 最近一次缓存写入为 09-09 02:58，执行时 7 小时内无新写入 → 无在跑评估。
5. 生成一次性清单（779 个 / 604.53 GB）供复核，已在清理完成后删除，
   后续一律用 `prune_calibration_cache.py` 现算，不再保留会过期的静态清单。

## 执行（用户确认方案 A）

保留 `qwen3.5-4b-proxy-v2.pt`(+panel) + 当前根 v202 + 回退根 v195，删除其余全部。

- 实际删除 **730 个文件**，释放 **540.51 GB**，失败 0 个。
- 待删清单生成时为 779 个；执行时只剩 730 个（49 个在执行前已被其他进程移除），
  故本次与清单合计的差额约 64 GB 不由本脚本释放。

## 清理后

| 项 | 清理后 |
|---|---|
| 校准缓存 | 18 个 / 67.25 GB（v202 6 个 + v195 12 个） |
| dense 主缓存 | 6.28 GB |
| **cache 合计** | **73.53 GB** |
| `artifacts/` 整体 | 约 91 GB |

**共释放约 604 GB。**

## 清理后 2 小时复涨 11.21 GB（并发 session 造成，非漏删）

12:40 复核：`cache` 由 73.53 GB 回到 85 GB，新增 22−18=4 个文件，全部来自**并发 session 的活候选**：

| 写入时间 | solution 前缀 | 候选 | 大小 |
|---|---|---|---|
| 12:31 | `d61dfbb85a39334f` | v211 linear-aw8-output-code-group | 5.74 GB |
| 12:26 | `11d7bbcd049092e7` | v204 linear-no-rank2-residual | 5.74 GB |
| 12:28 | `52a79bd20cd3848d` | v205 attn-no-c764-rotation-search | 2 MB |

**这三份不能删**（正在被使用）。由此暴露清理脚本的隐患：本项目多 session 并发，盲跑
`prune` 会删掉别人正在写的缓存 → 已为脚本加 `--min-age-hours` 保护。

## 防复涨

新增清理脚本 `workbench/cache_cleanup/prune_calibration_cache.py`：
按 solution SHA 前缀保留、默认 dry run、`--apply` 才删除；空保留集时拒绝执行；
硬编码保护 dense 主缓存。已在 `AGENTS.md` §6 写入定期清理规则与命令。

当前根前缀取法：

    python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('solution.py').read_bytes()).hexdigest()[:16])"
