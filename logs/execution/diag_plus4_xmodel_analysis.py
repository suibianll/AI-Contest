import json
for name in ("diag-plus4-gpt2-attn-default", "diag-plus4-opt-attn-default"):
    d = json.load(open(r"artifacts\official_eval\%s.json" % name, encoding="utf-8"))
    pe = d["paired_effect"]["attention"]
    ov = pe["overall"]
    print("===", name, "===")
    print("mean %+.6f pos/neg %d/%d" % (ov["mean_delta_gain"], ov["positive_cases"], ov["negative_cases"]))
    bl = pe.get("by_layer", {})
    top = sorted(bl.items(), key=lambda kv: -kv[1]["mean_delta_gain"])
    print("top:", [(k, round(v["mean_delta_gain"], 4)) for k, v in top[:6]])
    print("bot:", [(k, round(v["mean_delta_gain"], 4)) for k, v in top[-6:]])
