"""VK candidate verification on real calibration windows.

  A. the candidate is standalone and exposes the six APIs;
  B. with no kernel in v_state, the V path is bitwise the parent's -- the
     fallback is real, not nominal;
  C. state legality: the kernel is one CPU tensor inside a legal state;
  D. per layer: which arm calibration chose, and how many codes moved;
  E. V-side reachability on a test window;
  F. deploy cost: the V API's wall time against the parent's, same windows.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCHIVE = ROOT / "solutions/20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
LAYERS = (0, 1, 5, 8, 15, 22)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    parent = load("vk_parent", ARCHIVE)
    candidate = load("vk_candidate", CANDIDATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    dim = int(pack["head_dim"])

    apis = (
        "hif4_calibration_and_quantize_weight", "hif4_dynamic_quantize_activation",
        "hif4_calibration_attention", "hif4_dynamic_quantize_q",
        "hif4_dynamic_quantize_k", "hif4_dynamic_quantize_v",
    )
    missing = [a for a in apis if not hasattr(candidate, a)]
    print(f"[A] six APIs present: {not missing}  missing={missing}")
    if missing:
        return 1

    rows = []
    for layer in LAYERS:
        # The calibration API takes dicts keyed "q"/"k"/"v"; a bare tuple makes
        # `_check_attention_state` see a degenerate state.
        windows = [
            {
                role: tuple(
                    t.to(device)
                    for t in v2._pair(pack["calibration_qkv"][s][layer][i].to(torch.float32))
                )
                for i, role in enumerate(("q", "k", "v"))
            }
            for s in range(len(pack["calibration_qkv"]))
        ]
        parent_states = parent.hif4_calibration_attention(windows, q_heads, kv_heads, dim)
        cand_states = candidate.hif4_calibration_attention(windows, q_heads, kv_heads, dim)

        arm = cand_states["v_state"].get("vk_arm")
        kernel = cand_states["v_state"].get("vk_kernel")
        # C. state legality of the kernel entry
        kernel_ok = (
            kernel is None
            or (torch.is_tensor(kernel) and kernel.device.type == "cpu"
                and bool(torch.isfinite(kernel).all())
                and kernel.dtype in (torch.float32, torch.float64))
        )

        # B/D/E on the first test window that exists
        entry = pack["test_qkv"][0][layer]
        moved = None
        fallback_bitwise = None
        time_parent = time_candidate = None
        if entry is not None:
            pairs = [v2._pair(entry[i].to(torch.float32)) for i in range(3)]
            gate = windows[3]
            # B. no kernel -> parent bitwise
            stripped = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cand_states.items()}
            stripped["v_state"] = {k: v for k, v in cand_states["v_state"].items() if k != "vk_kernel"}
            p_par = candidate.hif4_dynamic_quantize_v(*gate["v"], kv_heads, dim, parent_states["v_state"])
            p_str = candidate.hif4_dynamic_quantize_v(*gate["v"], kv_heads, dim, stripped["v_state"])
            fallback_bitwise = all(torch.equal(p_par[k], p_str[k]) for k in p_par)

            v_par = candidate.hif4_dynamic_quantize_v(*pairs[2], kv_heads, dim, parent_states["v_state"])
            v_cand = candidate.hif4_dynamic_quantize_v(*pairs[2], kv_heads, dim, cand_states["v_state"])
            moved = int((v_cand["mant"] != v_par["mant"]).sum())

            for _ in range(3):
                t0 = time.perf_counter()
                candidate.hif4_dynamic_quantize_v(*pairs[2], kv_heads, dim, parent_states["v_state"])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                time_parent = min(time_parent or 1e9, time.perf_counter() - t0)
                t0 = time.perf_counter()
                candidate.hif4_dynamic_quantize_v(*pairs[2], kv_heads, dim, cand_states["v_state"])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                time_candidate = min(time_candidate or 1e9, time.perf_counter() - t0)

        rows.append(
            {
                "layer": layer,
                "vk_arm": arm,
                "kernel_legal": bool(kernel_ok),
                "gate_parent": cand_states["v_state"].get("vk_gate_parent"),
                "gate_candidate": cand_states["v_state"].get("vk_gate_candidate"),
                "vk_changed": cand_states["v_state"].get("vk_changed"),
                "error": cand_states["v_state"].get("vk_error"),
                "fallback_bitwise": fallback_bitwise,
                "test_window_moved": moved,
                "sec_parent": time_parent,
                "sec_candidate": time_candidate,
            }
        )
        print(
            f"[L{layer:>2}] arm={arm} kernel_legal={kernel_ok} "
            f"gate={rows[-1]['gate_parent']} -> {rows[-1]['gate_candidate']} "
            f"changed={rows[-1]['vk_changed']} fallback_bitwise={fallback_bitwise} "
            f"test_moved={moved} "
            f"sec p/c={time_parent if time_parent is None else round(time_parent, 4)}/"
            f"{time_candidate if time_candidate is None else round(time_candidate, 4)}"
            + (f"  error={rows[-1]['error']}" if rows[-1]["error"] else ""),
            flush=True,
        )

    arms = [r["vk_arm"] for r in rows]
    print()
    print(f"[D] arms: {arms}")
    print(f"[B] fallback bitwise on every layer: {all(r['fallback_bitwise'] for r in rows if r['fallback_bitwise'] is not None)}")
    print(f"[C] kernel entry legal on every layer: {all(r['kernel_legal'] for r in rows)}")
    tp = [r["sec_parent"] for r in rows if r["sec_parent"]]
    tc = [r["sec_candidate"] for r in rows if r["sec_candidate"]]
    if tp and tc:
        print(f"[F] V API sec: parent mean={statistics.mean(tp):.4f}  candidate mean={statistics.mean(tc):.4f}  "
              f"delta={statistics.mean(tc) - statistics.mean(tp):+.4f}")

    (HERE / "verify.json").write_text(json.dumps({"layers": list(LAYERS), "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {HERE / 'verify.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
