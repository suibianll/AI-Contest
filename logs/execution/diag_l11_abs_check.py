import json
# probe L11 case-level detail
d = json.load(open(r"artifacts\official_eval\diag-plus4-attn-default.json", encoding="utf-8"))
cases = d["results"][0]["case_scores"]["attention"]
print("=== probe (+4 all layers) L11/L16 case detail ===")
for c in cases:
    if c["layer"] in (11, 16):
        print("L%d len=%s split=%s gain_std=%.4f gain_player_probe=%.4f mse_std=%.6f"
              % (c["layer"], c.get("test_length", c.get("length")), c.get("test_split"),
                 c["gain"] if "gain_std" not in c else c["gain_std"], c["gain"],
                 c["mse_standard"]))
# wall times
for name in ("diag-plus4-attn-default", "v180-attn-default", "v184-attn-default"):
    dd = json.load(open(r"artifacts\official_eval\%s.json" % name, encoding="utf-8"))
    t = dd["results"][0].get("timing", {})
    print(name, "wall:", t.get("wall_seconds"), "api:", t.get("api_total_seconds"))
