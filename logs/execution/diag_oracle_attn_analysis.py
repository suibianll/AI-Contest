import json
d = json.load(open(r"artifacts\official_eval\diag-oracle-attn-default.json", encoding="utf-8"))
pe = d["paired_effect"]["attention"]
ov = pe["overall"]
print("mean", ov["mean_delta_gain"], "median", ov["median_delta_gain"],
      "min", ov["min_delta_gain"], "max", ov["max_delta_gain"])
bl = pe.get("by_layer", {})
print("--- by layer ---")
for k, v in sorted(bl.items(), key=lambda kv: -kv[1]["mean_delta_gain"]):
    print("  L%s: %+.5f (%d/%d)" % (k, v["mean_delta_gain"], v["positive_cases"], v["case_count"]))
for key in ("by_test_length", "by_length"):
    bl = pe.get(key, {})
    if bl:
        print("--- %s ---" % key)
        for k, v in sorted(bl.items(), key=lambda kv: -kv[1]["mean_delta_gain"]):
            print("  len%s: %+.5f (%d/%d)" % (k, v["mean_delta_gain"], v["positive_cases"], v["case_count"]))
wc = pe.get("worst_cases", [])
bc = pe.get("best_cases", [])
print("worst:", [(w.get("layer"), w.get("test_length", w.get("length")),
                  round(w.get("delta_gain", 0), 5)) for w in wc[:5]])
print("best:", [(w.get("layer"), w.get("test_length", w.get("length")),
                 round(w.get("delta_gain", 0), 5)) for w in bc[:10]])
