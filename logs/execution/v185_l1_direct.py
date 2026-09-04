"""Compute L1 = mean |delta_gain| per case for probe vs v180 baseline."""
import json


def gains(path):
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for c in d["results"][0]["case_scores"]["attention"]:
        key = (c["layer"], c.get("test_length", c.get("length")), c.get("test_split"), c.get("test_window"))
        out[key] = c["gain"]
    return out


for name in ("diag-plus4-attn-default", "v184-attn-default"):
    base = gains(r"artifacts\official_eval\v180-attn-default.json")
    cand = gains(r"artifacts\official_eval\%s.json" % name)
    common = [k for k in base if k in cand]
    deltas = [cand[k] - base[k] for k in common]
    l1 = sum(abs(x) for x in deltas) / len(deltas)
    mean = sum(deltas) / len(deltas)
    print("%s: n=%d L1=%.4f mean=%+.6f  gate L1<0.02: %s"
          % (name, len(deltas), l1, mean, "PASS" if l1 < 0.02 else "FAIL"))
