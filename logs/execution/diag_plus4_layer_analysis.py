import json
for name in ("diag-plus4-attn-default", "diag-oracle-attn-default"):
    d = json.load(open(r"artifacts\official_eval\%s.json" % name, encoding="utf-8"))
    pe = d["paired_effect"]["attention"]
    ov = pe["overall"]
    print("===", name, "===")
    print("mean %+.6f median %+.6f pos/neg/zero %d/%d/%d" % (
        ov["mean_delta_gain"], ov["median_delta_gain"],
        ov["positive_cases"], ov["negative_cases"], ov["zero_cases"]))
    bl = pe.get("by_layer", {})
    top = sorted(bl.items(), key=lambda kv: -kv[1]["mean_delta_gain"])
    print("top layers:", [(k, round(v["mean_delta_gain"], 4)) for k, v in top[:5]])
    print("bot layers:", [(k, round(v["mean_delta_gain"], 4)) for k, v in top[-5:]])
