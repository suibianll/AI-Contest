"""v184 timeout attribution via the official time model (2026-09-04).

T_official ~= 170.3 + 0.115*W_calib + 0.694*A_calib + 0.734*dyn_act
              - 1.58*dyn_qkv   (R^2=0.799, MAE 10.1s)

API seconds are read from the local JSON timing dicts.
"""
import json


def api_times(path):
    d = json.load(open(path, encoding="utf-8"))
    t = d["results"][0].get("timing", {})
    api = t.get("api_seconds") or t.get("apis") or {}
    if isinstance(api, dict):
        return api
    # fallback: per-api entries
    return {k: v for k, v in t.items() if k.startswith("hif4_")}


W = "hif4_calibration_and_quantize_weight"
A = "hif4_calibration_attention"
DA = "hif4_dynamic_quantize_activation"
QK = ["hif4_dynamic_quantize_q", "hif4_dynamic_quantize_k", "hif4_dynamic_quantize_v"]


def model(name, path):
    api = api_times(path)
    w = api.get(W, 0.0)
    a = api.get(A, 0.0)
    da = api.get(DA, 0.0)
    qkv = sum(api.get(k, 0.0) for k in QK)
    T = 170.3 + 0.115 * w + 0.694 * a + 0.734 * da - 1.58 * qkv
    print("%-28s W=%.1f A=%.1f dyn_act=%.1f dyn_qkv=%.1f -> T_pred=%.1fs"
          % (name, w, a, da, qkv, T))
    return T, a


print("=== time-model attribution (local API seconds -> official prediction) ===")
base = model("v182/v180 attn (4-code)", r"artifacts\official_eval\v180-attn-default.json")
v184 = model("v184 dual-window", r"artifacts\official_eval\v184-attn-default.json")
print()
print("A_calib local increment: %+.1fs (4->5 dual-window calibration x2)"
      % (v184[1] - base[1]))
print("official increment estimate: 0.694 * %+.1f = %+.1fs"
      % (v184[1] - base[1], 0.694 * (v184[1] - base[1])))
print()
print("=== predicted official totals ===")
print("v182 parent:            273.0s (actual)")
print("v184 from v182:  273 + %+.1f = %.1fs  (actual: >300 TIMEOUT)"
      % (0.694 * (v184[1] - base[1]), 273 + 0.694 * (v184[1] - base[1])))
print("v180 parent:            242.0s (actual)")
print("v184 from v180:  242 + %+.1f = %.1fs  (<280 submit gate: %s)"
      % (0.694 * (v184[1] - base[1]), 242 + 0.694 * (v184[1] - base[1]),
         "PASS" if 242 + 0.694 * (v184[1] - base[1]) < 280 else "FAIL"))
