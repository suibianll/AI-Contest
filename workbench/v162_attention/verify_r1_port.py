"""R1 port verification: Linear == v162 standard, Attention == v189, bitwise."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
import official_eval as v2  # noqa: E402
import reference_hif4 as ref  # noqa: E402

V162 = ROOT / "solutions/20260903_v162_standard-baseline-both_scoreNA_timeNA/solution.py"
V189 = ROOT / "solutions/20260906_v189_static-actorder-hdiag_recovered_scoreNA_timeNA/solution.py"
PORT = ROOT / "workbench/v162_attention/candidate_v2/solution.py"
CACHE = ROOT / "artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt"

FIELDS = ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    torch.manual_seed(0)
    base = load(V162, "r1_v162")
    v189 = load(V189, "r1_v189")
    port = load(PORT, "r1_port")
    pack = torch.load(CACHE, map_location="cpu", weights_only=False)

    failures = 0
    checked_linear = 0
    for layer in (0, 7, 15, 23):
        for role in ("q", "k", "v", "o", "fc_gate", "fc_up", "proj"):
            w_quant, w_scale = v2._pair(pack["weights"][layer][role])
            a = base.hif4_calibration_and_quantize_weight(w_quant, w_scale, [])
            b = port.hif4_calibration_and_quantize_weight(w_quant, w_scale, [])
            for key in FIELDS:
                if not torch.equal(a["weight_params"][key], b["weight_params"][key]):
                    print(f"LINEAR FAIL weight L{layer} {role} {key}")
                    failures += 1
            if b["activation_state"] != {}:
                print(f"LINEAR FAIL activation_state not empty L{layer} {role}")
                failures += 1
            act_quant, act_scale = v2._pair(pack["calibration_activations"][role][0][layer])
            a = base.hif4_dynamic_quantize_activation(act_quant, act_scale, {})
            b = port.hif4_dynamic_quantize_activation(act_quant, act_scale, {})
            for key in FIELDS:
                if not torch.equal(a[key], b[key]):
                    print(f"LINEAR FAIL activation L{layer} {role} {key}")
                    failures += 1
            checked_linear += 2
    print(f"linear control: {checked_linear} comparisons, {failures} failures")

    checked_attn = 0
    attn_failures = 0
    for layer in (0, 7, 15, 23):
        windows = []
        for sample in range(len(pack["calibration_windows"])):
            q, k, v = pack["calibration_qkv"][sample][layer]
            windows.append({"q": v2._pair(q), "k": v2._pair(k), "v": v2._pair(v)})
        states_189 = v189.hif4_calibration_attention(
            windows, pack["q_heads"], pack["kv_heads"], pack["head_dim"]
        )
        states_port = port.hif4_calibration_attention(
            windows, pack["q_heads"], pack["kv_heads"], pack["head_dim"]
        )
        for name in ("q_state", "k_state", "v_state"):
            ref.validate_state(states_port[name])
        q, k, v = pack["test_qkv"][pack["metadata"].get("panel_window_indices", [0, 1, 2, 3, 4])[0]][layer] \
            if "test_qkv" in pack and pack["test_qkv"][0][layer] is not None else (None, None, None)
        # use a calibration window as the probe input if test qkv absent in raw pack
        if q is None:
            q, k, v = pack["calibration_qkv"][1][layer]
        for api, pair, heads, state_name in (
            ("q", q, pack["q_heads"], "q_state"),
            ("k", k, pack["kv_heads"], "k_state"),
            ("v", v, pack["kv_heads"], "v_state"),
        ):
            quant, scale = v2._pair(pair) if not isinstance(pair, tuple) or not torch.is_tensor(pair[1]) else pair
            out_189 = getattr(v189, f"hif4_dynamic_quantize_{api}")(
                quant, scale, heads, pack["head_dim"], states_189[state_name]
            )
            out_port = getattr(port, f"hif4_dynamic_quantize_{api}")(
                quant, scale, heads, pack["head_dim"], states_port[state_name]
            )
            for key in FIELDS:
                if not torch.equal(out_189[key], out_port[key]):
                    print(f"ATTN FAIL L{layer} {api} {key}")
                    attn_failures += 1
            checked_attn += 1
    print(f"attention control: {checked_attn} API probes, {attn_failures} failures")
    if failures or attn_failures:
        return 1
    print("R1 PORT VERIFICATION PASS (Linear==v162, Attention==v189, bitwise)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
