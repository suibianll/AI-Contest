"""Root component time audit for solution.py (read-only instrumentation).

Loads the root solution as a module, wraps key component functions with
synchronized timers, and line-traces the two big calibration bodies
(Linear base at ~7802, v189 Attention at ~9250) plus the two final
wrappers, so inline segments (rank-1/rank-2 residual, adaptive-reg
cholesky loop, adaptive-offsets loop, ...) get attributed without
modifying solution.py.  Nothing is written back to the repo; results go
to results.json next to this script.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(r"D:\工作内容\AI竞赛")
sys.path.insert(0, str(ROOT / "evaluator"))
import official_eval as v2  # noqa: E402

OUT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda")

spec = importlib.util.spec_from_file_location("root_solution", ROOT / "solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

# --------------------------------------------------------------------------
# synchronized function wrappers
# --------------------------------------------------------------------------
func_stats: dict[str, list[float]] = {}
_active: set[str] = set()

WRAP_NAMES = [
    # linear shared
    "_dequantize_nvfp4_float32",
    "_dense_to_hif4",
    "_dequantize_hif4",
    "_linear_output_candidate_metrics",
    "_linear_output_candidate_metrics_combos",
    "_linear_candidate_metrics",
    "_linear_smooth_hybrid_metrics",
    "_block_swap_optimize",
    "_linear_pair_transform",
    "_transformed_covariance",
    "_gptq_quantize_weight",
    "_weight_e2e_refine",
    "_activation_gptq_quantize",
    "_v202_sample_energy_block_order_from_calibration",
    "_combined_sample_energy_block_order",
    "_loss_capture_ratio",
    # attention shared
    "_attention_candidate_metrics",
    "_attention_deployed_mse",
    "_attention_forward",
    "_solve_k_center_scale_aware",
    "_fit_attention_pair_matrix_smooth",
    "_fit_attention_logit_gain",
    "_attention_qk_fisher_importance",
    "_attention_rotation_signs",
    "_a2_train_rotation",
    "_a2_true_path_gate_loss",
]


def _sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


for _name in WRAP_NAMES:
    _fn = getattr(sol, _name, None)
    if _fn is None:
        continue
    func_stats[_name] = [0.0, 0]

    def _make(fn, name):
        def wrapped(*args, **kwargs):
            if name in _active:
                return fn(*args, **kwargs)
            _active.add(name)
            t0 = _sync_time()
            try:
                return fn(*args, **kwargs)
            finally:
                func_stats[name][0] += _sync_time() - t0
                func_stats[name][1] += 1
                _active.discard(name)

        return wrapped

    setattr(sol, _name, _make(_fn, _name))

# --------------------------------------------------------------------------
# line tracer for the calibration bodies
# --------------------------------------------------------------------------
linear_base = sol._COMBINED_LINEAR_BASE_CALIBRATION          # def at 7802
linear_wrapper = sol.hif4_calibration_and_quantize_weight    # def at 11645
attn_v189 = sol._V189_CALIBRATION_ATTENTION                  # def at 9250
attn_wrapper = sol.hif4_calibration_attention                # def at 11529


def _collect_codes(fn, qual, out):
    # @torch.no_grad() decorations share one decorate_context code object;
    # unwrap to the real function code before tracing.
    fn = getattr(fn, "__wrapped__", fn)
    code = fn.__code__
    out[code] = qual
    for const in code.co_consts:
        if hasattr(const, "co_name") and hasattr(const, "co_consts"):
            out[const] = f"{qual}.{const.co_name}"
            _recurse_codes(const, f"{qual}.{const.co_name}", out)


def _recurse_codes(code, qual, out):
    for const in code.co_consts:
        if hasattr(const, "co_name") and hasattr(const, "co_consts"):
            out[const] = f"{qual}.{const.co_name}"
            _recurse_codes(const, f"{qual}.{const.co_name}", out)


TRACE_CODES: dict = {}
_collect_codes(linear_base, "linear_base", TRACE_CODES)
_collect_codes(linear_wrapper, "linear_wrapper", TRACE_CODES)
_collect_codes(attn_v189, "attn_v189", TRACE_CODES)
_collect_codes(attn_wrapper, "attn_wrapper", TRACE_CODES)

line_stats: dict[str, dict[int, float]] = {}
line_counts: dict[str, dict[int, int]] = {}
TRACE_SYNC = os.environ.get("AUDIT_TRACE_SYNC", "1") == "1"
TRACE_ON = os.environ.get("AUDIT_TRACE", "1") == "1"


def _stamp() -> float:
    if TRACE_SYNC:
        return _sync_time()
    return time.perf_counter()


def _global_trace(frame, event, arg):
    if event != "call" or frame.f_code not in TRACE_CODES:
        return None
    qual = TRACE_CODES[frame.f_code]
    table = line_stats.setdefault(qual, {})
    counts = line_counts.setdefault(qual, {})
    state = {"last": None, "t": _stamp()}

    def _local(fr, ev, ar):
        if ev in ("line", "return", "exception"):
            now = _stamp()
            if state["last"] is not None:
                table[state["last"]] = table.get(state["last"], 0.0) + (now - state["t"])
                counts[state["last"]] = counts.get(state["last"], 0) + 1
            state["last"] = fr.f_lineno if ev == "line" else None
            state["t"] = now
        return _local

    return _local


# --------------------------------------------------------------------------
# case construction (mirrors evaluator/proxy_v3_eval.py calibration inputs)
# --------------------------------------------------------------------------
def move_pair(pair):
    return (pair[0].to(DEVICE), pair[1].to(DEVICE))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    raw = v2.load_pack(ROOT / "artifacts" / "official_eval" / "cache" / "qwen3.5-4b-proxy-v2.pt")
    print(f"layers={raw.layers} roles={raw.roles} q_heads={raw.q_heads} "
          f"kv_heads={raw.kv_heads} head_dim={raw.head_dim} "
          f"calib_windows={len(raw.calibration_windows)}")
    attn_pool = raw.metadata.get("attention_layers")
    print(f"attention_layers={attn_pool}")

    # shape inventory for linear
    shapes = {}
    for layer in range(raw.layers):
        for role in raw.roles:
            w = raw.weights[layer][role]
            shapes.setdefault((role, tuple(w.shape)), []).append(layer)
    for key, layers in sorted(shapes.items(), key=lambda kv: -kv[0][1][0] * kv[0][1][1]):
        print(f"  {key[0]:24s} shape={key[1]} layers={layers[:6]}{'...' if len(layers) > 6 else ''}")

    # pick up to 5 linear cases covering distinct shapes, largest d_in first
    role_filter = os.environ.get("AUDIT_ROLES")
    if role_filter:
        wanted = set(role_filter.split(","))
        picked_linear = [
            (layers[0], role, shape)
            for (role, shape), layers in shapes.items()
            if role in wanted
        ]
    else:
        picked_linear = []
        seen_roles = set()
        for (role, shape), layers in sorted(shapes.items(), key=lambda kv: -kv[0][1][1]):
            if role in seen_roles:
                continue
            seen_roles.add(role)
            picked_linear.append((layers[0], role, shape))
            if len(picked_linear) >= 5:
                break

    attn_layers = sorted(int(x) for x in (attn_pool if attn_pool is not None else range(raw.layers)))
    step = max(1, len(attn_layers) // 3)
    picked_attn = [attn_layers[i] for i in range(0, len(attn_layers), step)][:3]

    print(f"picked_linear={picked_linear}")
    print(f"picked_attn={picked_attn}")

    config = {k: getattr(sol, k, None) for k in (
        "_WEIGHT_QUADRATIC", "_WEIGHT_QUADRATIC_MAX_FEATURES", "_WIDE_LAYER_MIN_DIM",
        "_BLOCK_SMOOTH_SIZES", "_BLOCK_SMOOTH_SEEDS", "_LINEAR_SMOOTH_END_TO_END",
        "_WEIGHT_GPTQ", "_ACTIVATION_GPTQ", "_ADAPTIVE_OFFSETS",
        "_ADAPTIVE_OFFSET_CANDIDATES", "_ADAPTIVE_ACT_GPTQ_REG",
        "_ADAPTIVE_ACT_GPTQ_REG_CANDIDATES", "_WEIGHT_RESIDUAL_RANK",
        "_ATTN_OUTPUT_SELECTOR", "_ATTN_FISHER_IMPORTANCE",
        "_ATTN_PAIR_MATRIX_SMOOTH", "_ATTN_LOGIT_GAIN", "_A2_TRAIN_STEPS",
        "_ATTN_SCALE_AWARE_CENTER", "_LINEAR_SMOOTH_BLOCK_JOINT",
        "_WEIGHT_SMOOTH_ALPHAS", "_WEIGHT_SMOOTH_ALPHAS_WIDE",
        "_WEIGHT_E2E_REFINE", "_ACTIVATION_QUADRATIC", "_DATA_DRIVEN_RATIO",
        "_BLOCK_SWAP_ROUNDS", "_PERMUTATION_BASES", "_V_IMPORTANCE_CANDIDATES",
    )}
    config = {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in config.items()}

    results = {"config": config, "cases": []}

    def run_traced(fn, *args):
        global line_stats, line_counts
        line_stats = {}
        line_counts = {}
        for key in func_stats:
            func_stats[key][0] = 0.0
            func_stats[key][1] = 0
        t0 = _sync_time()
        if TRACE_ON:
            sys.settrace(_global_trace)
        try:
            out = fn(*args)
        finally:
            if TRACE_ON:
                sys.settrace(None)
        wall = _sync_time() - t0
        lines = {qual: {str(ln): t for ln, t in sorted(tbl.items())}
                 for qual, tbl in line_stats.items()}
        counts = {qual: {str(ln): c for ln, c in sorted(tbl.items())}
                  for qual, tbl in line_counts.items()}
        funcs = {k: {"seconds": v[0], "calls": v[1]}
                 for k, v in func_stats.items() if v[1] > 0}
        return out, wall, lines, counts, funcs

    for layer, role, shape in picked_linear if only in ("all", "linear") else []:
        weight_pair = move_pair(v2._pair(raw.weights[layer][role]))
        calib = [
            move_pair(v2._pair(raw.calibration_activations[role][sample][layer]))
            for sample in range(min(2, len(raw.calibration_windows)))
        ]
        _, wall, lines, counts, funcs = run_traced(
            sol.hif4_calibration_and_quantize_weight, weight_pair[0], weight_pair[1], calib
        )
        print(f"[linear] layer={layer} role={role} shape={shape} wall={wall:.3f}s")
        results["cases"].append({
            "kind": "linear", "layer": layer, "role": role, "shape": list(shape),
            "wall_seconds": wall, "functions": funcs, "lines": lines, "counts": counts,
        })
        del weight_pair, calib
        gc.collect()
        torch.cuda.empty_cache()

    for layer in picked_attn if only in ("all", "attention") else []:
        calib = []
        for sample in range(len(raw.calibration_windows)):
            q, k, v = raw.calibration_qkv[sample][layer]
            calib.append({
                "q": move_pair(v2._pair(q)),
                "k": move_pair(v2._pair(k)),
                "v": move_pair(v2._pair(v)),
            })
        _, wall, lines, counts, funcs = run_traced(
            sol.hif4_calibration_attention, calib, raw.q_heads, raw.kv_heads, raw.head_dim
        )
        print(f"[attention] layer={layer} wall={wall:.3f}s")
        results["cases"].append({
            "kind": "attention", "layer": layer,
            "wall_seconds": wall, "functions": funcs, "lines": lines, "counts": counts,
        })
        del calib
        gc.collect()
        torch.cuda.empty_cache()

    out_name = "results.json" if only == "all" else f"results-{only}.json"
    with open(OUT / out_name, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {out_name}")


if __name__ == "__main__":
    main()
