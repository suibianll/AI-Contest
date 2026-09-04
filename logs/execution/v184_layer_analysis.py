import json
d = json.load(open(r"artifacts\official_eval\v184-attn-default.json", encoding="utf-8"))
pe = d["paired_effect"]["attention"]
ov = pe["overall"]
print("v184 gated: mean %+.6f median %+.6f pos/neg/zero %d/%d/%d" % (
    ov["mean_delta_gain"], ov["median_delta_gain"],
    ov["positive_cases"], ov["negative_cases"], ov["zero_cases"]))
bl = pe.get("by_layer", {})
top = sorted(bl.items(), key=lambda kv: -kv[1]["mean_delta_gain"])
print("by layer:")
for k, v in top:
    print("  L%s: %+.5f (%d/%d changed? pos=%d neg=%d zero=%d)" % (
        k, v["mean_delta_gain"], v["positive_cases"], v["case_count"],
        v["positive_cases"], v["negative_cases"], v["zero_cases"]))
print()
# compare with probe
d2 = json.load(open(r"artifacts\official_eval\diag-plus4-attn-default.json", encoding="utf-8"))
pe2 = d2["paired_effect"]["attention"]
bl2 = pe2.get("by_layer", {})
print("probe (all-layers +4) by layer for the 7 accepted layers:")
accepted = [k for k, v in bl.items() if v["zero_cases"] < v["case_count"]]
for k in accepted:
    v1 = bl[k]; v2 = bl2.get(k, {})
    print("  L%s: v184 %+.5f (pos %d neg %d zero %d) | probe %+.5f" % (
        k, v1["mean_delta_gain"], v1["positive_cases"], v1["negative_cases"],
        v1["zero_cases"], v2.get("mean_delta_gain", float("nan"))))
