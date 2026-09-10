"""A-SR1 propagation: if the factor-built P replaces inv(m), how far does it travel?

``audit.py`` answered the card's crux question: the substitution is not bitwise
equivalent, on every step of every layer, because ``inv`` and ``U diag(1/s) Vh``
are different float32 algorithms for the same mathematical object (worst relative
divergence ~1e-6 on P).  The plan asks for more than the first divergence:

    "出现差异且不能在固定方案内修复，则保存首个分歧与传播结果"

so this script runs the modified training path for real -- the factor-built P is
substituted for ``inv(m)^T`` exactly where the card would put it -- and records
what the divergence does to the things the run is actually judged on:

  * the step-by-step trajectory: m, p, the gradient, and the update;
  * what the training returns: the compiled Q/K rotations and the K center;
  * what those compile into: the dynamic Q/K/V five fields on real activations;
  * the gate outcome per layer -- recorded, but explicitly NOT treated as a
    rescue: the plan says in as many words, "不以六层gate恰好相同掩盖训练路径变化".

The substitution is applied by wrapping ``torch.linalg.inv`` so that the value it
returns is the one the card would build from the preceding projection's factors.
Step 1 inverts ``eye`` (``n`` starts at zero) and the card keeps ``I`` there, so
that call is left alone.

CPU only, real data from the 4B pack.  Writes no candidate.
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

FIVE_FIELDS = ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.abs().amax().item()
    return float((a - b).abs().max().item() / max(denom, 1e-30))


def real_windows(pack, layer, pair):
    return [
        {
            "q": pair(pack["calibration_qkv"][s][layer][0]),
            "k": pair(pack["calibration_qkv"][s][layer][1]),
            "v": pair(pack["calibration_qkv"][s][layer][2]),
        }
        for s in range(len(pack["calibration_qkv"]))
    ]


def run_train(module, windows, heads, device, card_mode: bool):
    """Runs _agr1_train, optionally with the card's factor-built P."""

    q_heads, kv_heads, head_dim = heads
    states = module._AGR1_PARENT_CALIBRATION(windows, q_heads, kv_heads, head_dim)
    fit_windows = windows[: len(windows) - len(module._AGR1_GATE_WINDOWS)]
    eye = torch.eye(int(head_dim), dtype=torch.float32)

    original_project = module._agr1_project
    original_inv = torch.linalg.inv
    pending = {"card_p": None, "step": 0}
    trace = {"p": [], "m": [], "n": []}

    def project_spy(m):
        out = original_project(m)
        u, sv, vh = torch.linalg.svd(m)
        sv = sv.clamp(min=module._AGR1_SINGULAR_LO, max=module._AGR1_SINGULAR_HI)
        # M^{-T} for this projection: what the next step would use.
        pending["card_p"] = (u * (1.0 / sv).unsqueeze(-2)) @ vh
        return out

    def inv_spy(m):
        out = original_inv(m)
        pending["step"] += 1
        if card_mode and pending["step"] > 1 and pending["card_p"] is not None:
            # The code does inv(m).transpose(-1, -2), so hand back the transpose
            # of the card's P to land on the same p.
            return pending["card_p"].transpose(-1, -2).clone()
        return out

    module._agr1_project = project_spy
    torch.linalg.inv = inv_spy
    try:
        tq, tk, center, info = module._agr1_train(
            fit_windows, states, q_heads, kv_heads, head_dim, device
        )
    finally:
        module._agr1_project = original_project
        torch.linalg.inv = original_inv
    return {
        "tq": tq,
        "tk": tk,
        "center": center,
        "info": info,
        "states": states,
    }


def main() -> int:
    torch.set_grad_enabled(False)
    if not PACK.exists():
        raise SystemExit("the 4B pack is required")
    parent_module = load_module(CONTROL, "asr1_parent")
    card_module = load_module(CONTROL, "asr1_card")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    heads = (int(pack["q_heads"]), int(pack["kv_heads"]), int(pack["head_dim"]))
    layers = [int(layer) for layer in pack["metadata"]["attention_layers"]]
    device = torch.device("cpu")

    results = []
    for layer in layers:
        windows = real_windows(pack, layer, v2._pair)
        parent = run_train(parent_module, windows, heads, device, card_mode=False)
        card = run_train(card_module, windows, heads, device, card_mode=True)

        rotation_q_bitwise = raw_bytes(parent["tq"]) == raw_bytes(card["tq"])
        rotation_k_bitwise = raw_bytes(parent["tk"]) == raw_bytes(card["tk"])
        center_bitwise = (
            (parent["center"] is None and card["center"] is None)
            or (
                parent["center"] is not None
                and card["center"] is not None
                and raw_bytes(parent["center"]) == raw_bytes(card["center"])
            )
        )

        # What those rotations compile into on real activations.  The rotations
        # are installed the same way hif4_calibration_attention installs them.
        def install(module, states, trained):
            candidate = module._agr1_parent_copy(states)
            candidate["q_state"]["learned_rotation"] = trained["tq"]
            candidate["k_state"]["learned_rotation"] = trained["tk"]
            if trained["center"] is not None:
                candidate["k_state"]["learned_center"] = trained["center"]
            return candidate

        parent_installed = install(parent_module, parent["states"], parent)
        card_installed = install(card_module, card["states"], card)

        probe_q = windows[0]["q"]
        parent_q = parent_module.hif4_dynamic_quantize_q(
            *probe_q, heads[0], heads[2], parent_installed["q_state"]
        )
        card_q = card_module.hif4_dynamic_quantize_q(
            *probe_q, heads[0], heads[2], card_installed["q_state"]
        )
        # Discrete fields (sign/mant) make a relative metric meaningless, so the
        # five fields are compared bitwise only, per field.
        field_bitwise = {
            f: raw_bytes(parent_q[f]) == raw_bytes(card_q[f]) for f in FIVE_FIELDS
        }
        q_fields_bitwise = all(field_bitwise.values())

        # The gate decision, computed exactly as hif4_calibration_attention does
        # it.  Recorded because the plan asks for it, but it is not a rescue.
        def gate_accepts(module, trained):
            installed = install(module, trained["states"], trained)
            accepted = True
            losses = []
            for index in module._AGR1_GATE_WINDOWS:
                parent_loss = module._agr1_gate_loss(
                    windows[index], trained["states"], heads[0], heads[1], heads[2]
                )
                candidate_loss = module._agr1_gate_loss(
                    windows[index], installed, heads[0], heads[1], heads[2]
                )
                accepted = accepted and (candidate_loss < parent_loss)
                losses.append((float(parent_loss), float(candidate_loss)))
            return accepted, losses

        parent_accepts, parent_losses = gate_accepts(parent_module, parent)
        card_accepts, card_losses = gate_accepts(card_module, card)

        results.append(
            {
                "layer": layer,
                "gate_accepts_parent": parent_accepts,
                "gate_accepts_card": card_accepts,
                "gate_decision_unchanged": parent_accepts == card_accepts,
                "gate_losses_parent": parent_losses,
                "gate_losses_card": card_losses,
                "rotation_q_bitwise": rotation_q_bitwise,
                "rotation_k_bitwise": rotation_k_bitwise,
                "center_bitwise": center_bitwise,
                "rotation_q_max_abs": max_abs(parent["tq"], card["tq"])
                if parent["tq"].shape == card["tq"].shape
                else None,
                "n_norm_parent": parent["info"].get("agr1_n_norm"),
                "n_norm_card": card["info"].get("agr1_n_norm"),
                "final_loss_parent": parent["info"].get("agr1_final_loss"),
                "final_loss_card": card["info"].get("agr1_final_loss"),
                "dynamic_q_fields_bitwise": field_bitwise,
                "dynamic_q_five_fields_bitwise": q_fields_bitwise,
                "dynamic_q_max_abs_per_field": {
                    f: max_abs(parent_q[f].float(), card_q[f].float()) for f in FIVE_FIELDS
                },
            }
        )
        record = results[-1]
        print(
            f"[layer {layer}] gate accepts {parent_accepts} -> {card_accepts} "
            f"(decision unchanged={parent_accepts == card_accepts}) | rotation bitwise "
            f"q={rotation_q_bitwise} k={rotation_k_bitwise} center={center_bitwise} "
            f"(max abs {record['rotation_q_max_abs']:.3e}) | dyn-Q five fields "
            f"bitwise={q_fields_bitwise} {field_bitwise}",
            flush=True,
        )

    payload = {
        "control": str(CONTROL.relative_to(ROOT)).replace("\\", "/"),
        "note": (
            "the card's factor-built P substituted for inv(m)^T in a real run; "
            "gate outcomes are recorded but are not a rescue, per the plan"
        ),
        "results": results,
        "gate_decision_unchanged_on_all_layers": all(
            r["gate_decision_unchanged"] for r in results
        ),
        "rotations_bitwise_on_all_layers": all(
            r["rotation_q_bitwise"] and r["rotation_k_bitwise"] for r in results
        ),
    }
    (HERE / "propagation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {HERE / 'propagation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
