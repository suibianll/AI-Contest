"""合成 Attention 安全评测器（E1，S0 预注册实现）。

背景：官方评测含合成 Attention 场景（saturated logits 等），本地此前只有
真实 GPT-2 路径，无法在提交前预筛此类场景的退化（variantH 教训）。本
评测器按执行日志 E1 预注册的冻结矩阵生成确定性合成数据，并复用
real_data_eval 的评分口径与 nvfp4_sim 的参考编码：

- reference = NVFP4 反量化（nvfp4_encode + solution 反量化）；
- standard  = 朴素 HiF4（solution._dense_to_hif4，无 search offsets）；
- candidate = solution 动态量化输出；
- score     = (mse_std − mse_player) / mse_std，causal/non-causal 双轨。

场景参数、维度网格、seed 一经冻结不得按结果回调；本工具只作本地安全
诊断，不外推官方绝对分数，不参与比赛提交。

用法：
    python evaluator/synthetic_attention_eval.py --solution solution.py
    python evaluator/synthetic_attention_eval.py --cases saturated_logits --seeds 0
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path
from types import ModuleType

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nvfp4_sim import nvfp4_encode  # noqa: E402
from real_data_eval import load_solution, score_attention  # noqa: E402

# ---- 冻结矩阵（E1 预注册；禁止按结果修改） ----

SCENARIOS = (
    "balanced",
    "saturated_logits",
    "near_uniform",
    "v_outlier",
    "qk_dynamic_imbalance",
    "k_mean_shift",
    "heavy_tail",
    "qk_correlated",
)
TOPOLOGIES = ((4, 4), (4, 2))  # (q_heads, kv_heads)：MHA 与 GQA
HEAD_DIMS = (64, 128)
SEQS = (32, 128)
MODES = ("amax6", "amax4", "pow2")
SEEDS = (0, 1, 2)
CALIB_BATCHES = 2
TEST_BATCHES = 2
MASKS = ("causal", "non-causal")

_HIF4_PARAM_KEYS = frozenset(
    ("sign", "mant", "scale_lv3", "scale_lv2", "scale_factor")
)


def _heavy_tail(shape: tuple[int, int]) -> torch.Tensor:
    """randn·m，m=10 若 u<0.1 否则 1（u~U(0,1) 逐元素）。"""
    base = torch.randn(shape)
    u = torch.rand(shape)
    mult = torch.where(u < 0.1, 10.0, 1.0)
    return base * mult


def generate_qkv(
    scenario: str, q_dim: int, k_dim: int, seq: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按冻结的场景配方生成一组 [seq, D] 的 q/k/v 稠密张量。

    RNG 消耗顺序固定为 q → k → v，保证同 case 复跑逐位一致。
    """
    q_shape = (seq, q_dim)
    k_shape = (seq, k_dim)
    v_shape = (seq, k_dim)
    if scenario == "balanced":
        q = torch.randn(q_shape)
        k = torch.randn(k_shape)
        v = torch.randn(v_shape)
    elif scenario == "saturated_logits":
        q = 4.0 * torch.randn(q_shape)
        k = 4.0 * torch.randn(k_shape)
        v = torch.randn(v_shape)
    elif scenario == "near_uniform":
        q = 0.05 * torch.randn(q_shape)
        k = 0.05 * torch.randn(k_shape)
        v = torch.randn(v_shape)
    elif scenario == "v_outlier":
        q = torch.randn(q_shape)
        k = torch.randn(k_shape)
        v = torch.randn(v_shape)
        outlier = (torch.arange(k_dim) % 20) == 7
        v[:, outlier] = v[:, outlier] * 50.0
    elif scenario == "qk_dynamic_imbalance":
        q = torch.randn(q_shape)
        wide = (torch.arange(q_dim) % 2) == 1
        q[:, wide] = q[:, wide] * 64.0
        k = torch.randn(k_shape)
        v = torch.randn(v_shape)
    elif scenario == "k_mean_shift":
        q = torch.randn(q_shape)
        k = torch.randn(k_shape) + 5.0
        v = torch.randn(v_shape)
    elif scenario == "heavy_tail":
        q = _heavy_tail(q_shape)
        k = _heavy_tail(k_shape)
        v = torch.randn(v_shape)
    elif scenario == "qk_correlated":
        q = torch.randn(q_shape)
        k = 0.8 * q[:, :k_dim] + 0.6 * torch.randn(k_shape)
        v = torch.randn(v_shape)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return q, k, v


def generate_case_data(
    scenario: str,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    seq: int,
    seed: int,
) -> tuple[list[dict[str, torch.Tensor]], list[tuple[torch.Tensor, ...]]]:
    """生成一个 case 的 calib/test 稠密数据（每 batch [seq, D]）。

    生成顺序：calib batch 0 → calib batch 1 → test batch 0 → test batch 1，
    每 batch 内 q → k → v；case 开始时重播种保证确定性。
    """
    torch.manual_seed(seed)
    q_dim = q_heads * head_dim
    k_dim = kv_heads * head_dim
    calib: list[dict[str, torch.Tensor]] = []
    for _ in range(CALIB_BATCHES):
        q, k, v = generate_qkv(scenario, q_dim, k_dim, seq)
        calib.append({"q": q, "k": k, "v": v})
    tests: list[tuple[torch.Tensor, ...]] = []
    for _ in range(TEST_BATCHES):
        tests.append(generate_qkv(scenario, q_dim, k_dim, seq))
    return calib, tests


def encode_case(
    calib: list[dict[str, torch.Tensor]],
    tests: list[tuple[torch.Tensor, ...]],
    mode: str,
) -> tuple[list[dict[str, tuple[torch.Tensor, ...]]], list[tuple[tuple[torch.Tensor, ...], ...]]]:
    """把稠密 calib/test 编码为 NVFP4 (carrier, scale) 对。"""
    calib_pairs = [
        {
            "q": nvfp4_encode(batch["q"], mode),
            "k": nvfp4_encode(batch["k"], mode),
            "v": nvfp4_encode(batch["v"], mode),
        }
        for batch in calib
    ]
    test_pairs = [
        (
            nvfp4_encode(q, mode),
            nvfp4_encode(k, mode),
            nvfp4_encode(v, mode),
        )
        for q, k, v in tests
    ]
    return calib_pairs, test_pairs


def check_state_tree(value: object, path: str, failures: list[str]) -> None:
    """校准 state 合法性：CPU、strided、连续、无梯度、有限、叶子类型。"""
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            failures.append(f"{path}: device={value.device}")
        if value.layout != torch.strided:
            failures.append(f"{path}: layout={value.layout}")
        if not value.is_contiguous():
            failures.append(f"{path}: not contiguous")
        if value.requires_grad:
            failures.append(f"{path}: requires_grad")
        if not bool(torch.isfinite(value.to(torch.float32)).all()):
            failures.append(f"{path}: non-finite values")
    elif isinstance(value, dict):
        for key, child in value.items():
            check_state_tree(child, f"{path}.{key}", failures)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            check_state_tree(child, f"{path}[{index}]", failures)
    elif value is not None and not isinstance(value, (bool, int, float, str)):
        failures.append(f"{path}: illegal leaf {type(value).__name__}")


def check_dynamic_params(
    solution: ModuleType,
    test_pairs: list[tuple[tuple[torch.Tensor, ...], ...]],
    states: dict,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    case_label: str,
    failures: list[str],
) -> None:
    """动态量化输出五字段（sign/mant/scale_lv3/scale_lv2/scale_factor）合法性。"""
    q_pair, k_pair, v_pair = test_pairs[0]
    try:
        params = (
            ("q", solution.hif4_dynamic_quantize_q(
                *q_pair, q_heads, head_dim, states["q_state"]
            )),
            ("k", solution.hif4_dynamic_quantize_k(
                *k_pair, kv_heads, head_dim, states["k_state"]
            )),
            ("v", solution.hif4_dynamic_quantize_v(
                *v_pair, kv_heads, head_dim, states["v_state"]
            )),
        )
    except Exception as exc:  # noqa: BLE001
        failures.append(f"{case_label}: dynamic-quantize raised {exc!r}")
        return
    for side, side_params in params:
        keys = set(side_params)
        if keys != set(_HIF4_PARAM_KEYS):
            failures.append(
                f"{case_label}: params[{side}] keys {sorted(keys)}"
            )
            continue
        for key, tensor in side_params.items():
            if tensor.device.type != "cpu":
                failures.append(f"{case_label}: params[{side}][{key}] device")
            if tensor.requires_grad:
                failures.append(f"{case_label}: params[{side}][{key}] grad")
            if not bool(torch.isfinite(tensor.to(torch.float32)).all()):
                failures.append(f"{case_label}: params[{side}][{key}] non-finite")


def evaluate(solution: ModuleType, args: argparse.Namespace) -> int:
    torch.set_grad_enabled(False)
    modes = tuple(args.modes)
    seeds = tuple(args.seeds)
    results: list[tuple[str, str, int, float, float]] = []
    failures: list[str] = []
    total_time = 0.0
    for scenario in SCENARIOS:
        for q_heads, kv_heads in TOPOLOGIES:
            for head_dim in HEAD_DIMS:
                for seq in SEQS:
                    name = f"{scenario}_h{q_heads}_kv{kv_heads}_d{head_dim}_s{seq}"
                    if args.cases and args.cases not in name:
                        continue
                    for mode in modes:
                        for seed in seeds:
                            start = time.perf_counter()
                            calib, tests = generate_case_data(
                                scenario, q_heads, kv_heads, head_dim, seq, seed
                            )
                            calib_pairs, test_pairs = encode_case(
                                calib, tests, mode
                            )
                            states = solution.hif4_calibration_attention(
                                calib_pairs, q_heads, kv_heads, head_dim
                            )
                            for side in ("q_state", "k_state", "v_state"):
                                check_state_tree(
                                    states[side], f"{name}/{side}", failures
                                )
                            check_dynamic_params(
                                solution,
                                test_pairs,
                                states,
                                q_heads,
                                kv_heads,
                                head_dim,
                                f"{name}/mode={mode}/seed={seed}",
                                failures,
                            )
                            scores = score_attention(
                                solution,
                                test_pairs,
                                states["q_state"],
                                states["k_state"],
                                states["v_state"],
                                q_heads,
                                kv_heads,
                                head_dim,
                                masks=MASKS,
                            )
                            causal = float(scores["causal"])
                            noncausal = float(scores["non-causal"])
                            for label, value in (
                                ("causal", causal),
                                ("non-causal", noncausal),
                            ):
                                if not math.isfinite(value):
                                    failures.append(
                                        f"{name} mode={mode} seed={seed}: "
                                        f"{label}={value}"
                                    )
                            elapsed = time.perf_counter() - start
                            total_time += elapsed
                            print(
                                f"CASE {name} mode={mode} seed={seed} "
                                f"causal={causal:.9f} "
                                f"noncausal={noncausal:.9f} "
                                f"time={elapsed:.2f}s",
                                flush=True,
                            )
                            results.append(
                                (name, mode, seed, causal, noncausal)
                            )
    # ---- 汇总（只聚合本次实际运行的 case） ----
    scenario_summary: dict[str, list[float]] = {}
    scenario_summary_nc: dict[str, list[float]] = {}
    for name, _mode, _seed, causal, noncausal in results:
        scenario = name.split("_h")[0]
        scenario_summary.setdefault(scenario, []).append(causal)
        scenario_summary_nc.setdefault(scenario, []).append(noncausal)
    print(f"SUMMARY cases={len(results)} total_time={total_time:.1f}s")
    for scenario in SCENARIOS:
        if scenario not in scenario_summary:
            continue
        causal_values = scenario_summary[scenario]
        noncausal_values = scenario_summary_nc[scenario]
        print(
            f"SCENARIO {scenario} n={len(causal_values)} "
            f"causal_mean={sum(causal_values) / len(causal_values):.6f} "
            f"noncausal_mean="
            f"{sum(noncausal_values) / len(noncausal_values):.6f}"
        )
    if results:
        all_causal = [r[3] for r in results]
        all_noncausal = [r[4] for r in results]
        worst_causal = min(results, key=lambda r: r[3])
        worst_noncausal = min(results, key=lambda r: r[4])
        print(
            f"OVERALL causal_mean={sum(all_causal) / len(all_causal):.6f} "
            f"noncausal_mean={sum(all_noncausal) / len(all_noncausal):.6f}"
        )
        print(
            f"WORST causal={worst_causal[3]:.6f} "
            f"({worst_causal[0]} mode={worst_causal[1]} seed={worst_causal[2]})"
        )
        print(
            f"WORST noncausal={worst_noncausal[4]:.6f} "
            f"({worst_noncausal[0]} mode={worst_noncausal[1]} "
            f"seed={worst_noncausal[2]})"
        )
    if failures:
        print(f"FAILURES {len(failures)}")
        for line in failures:
            print(f"FAIL {line}")
        return 1
    print("RESULT ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solution",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "solution.py",
        help="solution.py to evaluate",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="substring filter on frozen case names (default: all)",
    )
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help=f"comma-separated NVFP4 modes subset of {MODES}",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in SEEDS),
        help=f"comma-separated seeds subset of {SEEDS}",
    )
    args = parser.parse_args(argv)
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    if not modes or set(modes) - set(MODES):
        parser.error(f"--modes must be a non-empty subset of {MODES}")
    if not seeds or set(seeds) - set(SEEDS):
        parser.error(f"--seeds must be a non-empty subset of {SEEDS}")
    args.modes = modes
    args.seeds = seeds

    solution = load_solution(args.solution)
    digest = hashlib.sha256(args.solution.read_bytes()).hexdigest().upper()
    print(
        f"CONFIG solution={args.solution} sha256={digest} "
        f"cases_filter={args.cases or 'all'} modes={','.join(modes)} "
        f"seeds={','.join(str(s) for s in seeds)} "
        f"calib={CALIB_BATCHES} test={TEST_BATCHES} masks={','.join(MASKS)}",
        flush=True,
    )
    return evaluate(solution, args)


if __name__ == "__main__":
    raise SystemExit(main())
