"""L1 overfit gate for the single-window +4 probe (new AGENTS rule: L1 < 0.02)."""
import json

for name in ("diag-plus4-attn-default", "v184-attn-default"):
    d = json.load(open(r"artifacts\official_eval\%s.json" % name, encoding="utf-8"))
    pe = d["paired_effect"]["attention"]
    ov = pe["overall"]
    # reconstruct per-case |delta| from best/worst and by-layer data is not enough;
    # use case-level scores from results + baseline? Use paired JSON case deltas if present.
    cases = d["results"][0]["case_scores"]["attention"]
    # try to find per-case delta in paired_effect
    pc = pe.get("per_case") or pe.get("case_deltas")
    if pc:
        l1 = sum(abs(x) for x in pc) / len(pc)
        print("%s: L1=%.4f (gate <0.02: %s), mean=%+.6f"
              % (name, l1, "PASS" if l1 < 0.02 else "FAIL", ov["mean_delta_gain"]))
    else:
        # approximate from available distribution stats
        print("%s: per-case deltas not stored; mean=%+.6f pos/neg=%d/%d"
              % (name, ov["mean_delta_gain"], ov["positive_cases"], ov["negative_cases"]))
        print("  keys:", list(pe.keys()))
