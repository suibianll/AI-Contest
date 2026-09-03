import json

for name, label in [
    (r"artifacts\official_eval\legacy-v1\root-v127-fixed-attn-budget-sweep1-official-shape-v1.json", "v129 sweep1"),
    (r"artifacts\official_eval\legacy-v1\v138-attention-static-v86-budget-official-shape-v1.json", "v138 static"),
    (r"artifacts\official_eval\legacy-v1\root-v127-fixed-attn-budget-official-shape-v1.json", "v128 full"),
]:
    j = json.load(open(name, encoding="utf-8"))
    r = j["results"][0]
    t = r["timing"]
    print(f"=== {label}  total {t['api_total_seconds']:.1f}s")
    for k, v in t["api_seconds"].items():
        calls = t["api_calls"][k]
        print(f"  {k}: {v:.1f}s ({calls} calls, {v / max(calls, 1):.3f}s/call)")
    s = r.get("score", {})
    print("  attn_mean:", s.get("attention_mean"), " linear_mean:", s.get("linear_mean"))
