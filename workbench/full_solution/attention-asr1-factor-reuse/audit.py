"""A-SR1 audit: can the spectral factors really stand in for the next step's inverse?

The card's whole proposal is:

    _agr1_project spends one SVD per step and throws the factors away;
    the next step immediately computes inv(m) where m is built from that
    projection; so keep U/s/Vh and build P = M^{-T} = U diag(1/s) Vh instead.

The plan names the crux of it before anything is written (audit item 1):

    "核对实际投影与下一步M：现有代码存N=M-I，再重建I+N，浮点下未必与投影M逐位
     相同。明确这一差异，不能凭代数公式宣称等价。"

That is exactly what this script measures, on real calibration data, for every
full-attention layer.  The loop is

    for step in 1..32:
        m = eye + n
        m_inv = torch.linalg.inv(m)
        p = m_inv.transpose(-1, -2)
        ...
        n = _agr1_project(eye + n - lr * update) - eye

so at step k+1 the matrix the parent inverts is

    m_{k+1} = eye + (M'_k - eye)        (M'_k = what step k's projection returned)

while the card would take its factors from M'_k itself.  The two agree
algebraically and need not agree in float32.

Three things are measured per layer, none of them assumed:

  R  the reconstruction gap: is `eye + (M' - eye)` bitwise `M'`?
  F  factor fidelity: does `(u * sv) @ vh` from a fresh SVD of the projection's
     input reproduce the projection's own output bitwise?  (If not, this audit's
     factor capture would be measuring the wrong factors.)
  P  the divergence that decides the card: `inv(m_{k+1})^T` versus the
     factor-built `U diag(1/s) Vh` from M'_k, per step, in absolute and relative
     terms.

The plan's bar for this card is bitwise identity of the step-by-step path, so a
non-zero R or P is not a curiosity -- it is the answer.  The script reports the
numbers and states plainly which way it falls; it does not decide the card, and
it writes no candidate.

CPU only, real data from the 4B pack.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

CONTROL = HERE / "control" / "agr1-on-v237.py"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def real_windows(pack, layer, pair):
    return [
        {
            "q": pair(pack["calibration_qkv"][s][layer][0]),
            "k": pair(pack["calibration_qkv"][s][layer][1]),
            "v": pair(pack["calibration_qkv"][s][layer][2]),
        }
        for s in range(len(pack["calibration_qkv"]))
    ]


def audit_layer(module, windows, heads, layer):
    q_heads, kv_heads, head_dim = heads
    states = module._AGR1_PARENT_CALIBRATION(windows, q_heads, kv_heads, head_dim)
    fit_windows = windows[: len(windows) - len(module._AGR1_GATE_WINDOWS)]

    original_project = module._agr1_project
    projections = []

    def project_spy(m):
        out = original_project(m)
        u, sv, vh = torch.linalg.svd(m)
        sv = sv.clamp(min=module._AGR1_SINGULAR_LO, max=module._AGR1_SINGULAR_HI)
        projections.append(
            {
                "input": m.detach().clone(),
                "output": out.detach().clone(),
                "u": u,
                "sv": sv,
                "vh": vh,
            }
        )
        return out

    # The inverse is taken on the step's m; capture both m and what inv returned
    # so the comparison is against the value the parent actually uses.
    original_inv = torch.linalg.inv
    inverses = []

    def inv_spy(m):
        out = original_inv(m)
        inverses.append({"m": m.detach().clone(), "inv": out.detach().clone()})
        return out

    module._agr1_project = project_spy
    torch.linalg.inv = inv_spy
    try:
        tq, tk, center, info = module._agr1_train(
            fit_windows, states, q_heads, kv_heads, head_dim, torch.device("cpu")
        )
    finally:
        module._agr1_project = original_project
        torch.linalg.inv = original_inv

    eye = torch.eye(int(head_dim), dtype=torch.float32)

    # F: the factors this audit captured must reproduce the projection's own
    # output, otherwise everything below is measuring the wrong factors.
    factor_fidelity = []
    for record in projections:
        rebuilt = (record["u"] * record["sv"].unsqueeze(-2)) @ record["vh"]
        factor_fidelity.append(raw_bytes(rebuilt) == raw_bytes(record["output"]))
    if not all(factor_fidelity):
        raise AssertionError(
            f"layer {layer}: the captured factors do not reproduce _agr1_project's "
            "own output; the audit would be measuring the wrong factors"
        )

    # R: what the next step uses versus what the projection produced.
    gaps = []
    for record in projections:
        rebuilt_m = eye + (record["output"] - eye)
        diff = (rebuilt_m - record["output"]).abs()
        scale = record["output"].abs().amax().item()
        gaps.append(
            {
                "bitwise_equal": raw_bytes(rebuilt_m) == raw_bytes(record["output"]),
                "max_abs": float(diff.max().item()),
                "max_rel_to_scale": float(diff.max().item() / max(scale, 1e-30)),
            }
        )

    # P: parent's p versus the factor-built p, step by step.  `inverses[i]` is the
    # inverse taken at loop step i+1; the projection that precedes it is
    # `projections[i-1]`, and the one after the loop is `projections[-1]`.
    divergences = []
    for index, record in enumerate(inverses):
        # Step 1 inverts `eye` (n starts at zero) -- the card keeps I there too.
        if index == 0:
            continue
        source = projections[index - 1]
        parent_p = record["inv"].transpose(-1, -2)
        card_p = (
            source["u"] * (1.0 / source["sv"]).unsqueeze(-2)
        ) @ source["vh"]
        diff = (parent_p - card_p).abs()
        denom = parent_p.abs().amax().item()
        divergences.append(
            {
                "step": index + 1,
                "bitwise_equal": raw_bytes(parent_p) == raw_bytes(card_p),
                "max_abs": float(diff.max().item()),
                "max_rel": float(diff.max().item() / max(denom, 1e-30)),
                "m_matches_projection": gaps[index - 1]["bitwise_equal"],
            }
        )

    n_gap = sum(1 for g in gaps if not g["bitwise_equal"])
    n_div = sum(1 for d in divergences if not d["bitwise_equal"])
    worst_gap = max(g["max_abs"] for g in gaps)
    worst_rel = max(d["max_rel"] for d in divergences)

    print(
        f"[layer {layer}] projections={len(projections)} inverses={len(inverses)} "
        f"| R: {n_gap}/{len(gaps)} steps where eye+(M'-eye) != M' (worst abs {worst_gap:.3e}) "
        f"| P: {n_div}/{len(divergences)} steps where factor-P != inv(m)^T "
        f"(worst rel {worst_rel:.3e})"
    )
    return {
        "layer": layer,
        "projections": len(projections),
        "inverses": len(inverses),
        "factor_fidelity_all_bitwise": all(factor_fidelity),
        "reconstruction": {
            "steps_compared": len(gaps),
            "steps_not_bitwise": n_gap,
            "worst_max_abs": worst_gap,
        },
        "p_divergence": {
            "steps_compared": len(divergences),
            "steps_not_bitwise": n_div,
            "worst_max_rel": worst_rel,
            "first_step_not_bitwise": next(
                (d["step"] for d in divergences if not d["bitwise_equal"]), None
            ),
        },
    }


def main() -> int:
    torch.set_grad_enabled(False)
    if not PACK.exists():
        raise SystemExit("the 4B pack is required for this audit")
    module = load_module(CONTROL, "asr1_control")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    heads = (int(pack["q_heads"]), int(pack["kv_heads"]), int(pack["head_dim"]))
    layers = [int(layer) for layer in pack["metadata"]["attention_layers"]]
    print(f"[real] heads={heads} layers={layers}")

    results = []
    for layer in layers:
        windows = real_windows(pack, layer, v2._pair)
        results.append(audit_layer(module, windows, heads, layer))

    payload = {
        "control": str(CONTROL.relative_to(ROOT)).replace("\\", "/"),
        "control_sha256": __import__("hashlib").sha256(CONTROL.read_bytes()).hexdigest(),
        "heads": heads,
        "layers": layers,
        "results": results,
    }
    (HERE / "audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    every_gap = all(r["reconstruction"]["steps_not_bitwise"] == 0 for r in results)
    every_p = all(r["p_divergence"]["steps_not_bitwise"] == 0 for r in results)
    print()
    if every_gap and every_p:
        print(
            "VERDICT: on every layer, eye + (M' - eye) is bitwise M', and the "
            "factor-built P is bitwise the parent's inv(m)^T. The substitution is "
            "exactly equivalent on this data and the card can proceed."
        )
    else:
        print(
            "VERDICT: the substitution is NOT bitwise equivalent on real data. "
            f"Reconstruction fails on {sum(r['reconstruction']['steps_not_bitwise'] for r in results)} "
            f"step/layer pairs and P diverges on "
            f"{sum(r['p_divergence']['steps_not_bitwise'] for r in results)}. "
            "The plan's bar for this card is bitwise identity of the step-by-step "
            "path, so this is the answer the audit was run to get; the card does "
            "not clear its own equivalence requirement as specified."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
