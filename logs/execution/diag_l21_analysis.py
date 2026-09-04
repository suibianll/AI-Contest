"""Analyze diag_l21_offsets.jsonl: per-layer winning-offset distributions."""
import json
from collections import defaultdict

# 1) case order from the dump run
d = json.load(open(r"artifacts\official_eval\diag-l21-dump-attn-default.json", encoding="utf-8"))
cases = d["results"][0]["case_scores"]["attention"]
order = [(c["layer"], c.get("test_length", c.get("length"))) for c in cases]
assert len(order) == 120

# 2) aggregate jsonl into (api,hist) segments
segments = []  # list of (api, merged_hist, n_calls)
cur_api, cur_hist, cur_calls = None, defaultdict(int), 0
for line in open(r"logs\execution\diag_l21_offsets.jsonl", encoding="utf-8"):
    rec = json.loads(line)
    if rec["api"] != cur_api:
        if cur_api is not None:
            segments.append((cur_api, dict(cur_hist), cur_calls))
        cur_api, cur_hist, cur_calls = rec["api"], defaultdict(int), 0
    cur_calls += 1
    for k, v in rec["hist"].items():
        cur_hist[int(k)] += v
if cur_api is not None:
    segments.append((cur_api, dict(cur_hist), cur_calls))

print("segments:", len(segments), "api cycle:",
      [s[0] for s in segments[:6]])
# evaluation phase is the LAST 360 segments: strict (q,k,v) x 120;
# everything before is calibration-internal dynamic-API traffic.
segments = segments[-360:]
assert [s[0] for s in segments[:3]] == ["q", "k", "v"], [s[0] for s in segments[:3]]
# 3) map segments to cases: every 3 segments = 1 case (q,k,v)
per_case = {}
for i in range(120):
    seg = segments[3 * i: 3 * i + 3]
    assert [s[0] for s in seg] == ["q", "k", "v"], (i, [s[0] for s in seg])
    per_case[order[i]] = {s[0]: s[1] for s in seg}

# 4) per-layer merged histogram (over lengths)
lay_q = defaultdict(lambda: defaultdict(int))
lay_k = defaultdict(lambda: defaultdict(int))
lay_v = defaultdict(lambda: defaultdict(int))
for (layer, ln), h in per_case.items():
    for off, cnt in h["q"].items():
        lay_q[layer][off] += cnt
    for off, cnt in h["k"].items():
        lay_k[layer][off] += cnt
    for off, cnt in h["v"].items():
        lay_v[layer][off] += cnt


def summarize(name, hist):
    total = sum(hist.values())
    if total == 0:
        print(f"  {name}: (no hard blocks)")
        return
    out5 = sum(v for k, v in hist.items() if k > 4)      # beyond +4 window
    out4 = sum(v for k, v in hist.items() if k > 3)      # beyond old 4-code
    neg = sum(v for k, v in hist.items() if k < 0)
    top = sorted(hist.items(), key=lambda kv: -kv[1])[:6]
    print(f"  {name}: hard={total}  off>+4: {out5} ({out5/total*100:.1f}%)  "
          f"off<0: {neg} ({neg/total*100:.1f}%)  top={top}")


print("=== L21 (oracle residual +0.091@10 / +0.024@1024) ===")
summarize("q", lay_q[21]); summarize("k", lay_k[21]); summarize("v", lay_v[21])
print("=== L23 (biggest remaining-error layer, gate-rejected) ===")
summarize("q", lay_q[23]); summarize("k", lay_k[23]); summarize("v", lay_v[23])
print("=== L11 (+4 captured +0.13) ===")
summarize("q", lay_q[11]); summarize("k", lay_k[11]); summarize("v", lay_v[11])
print("=== L16 (+4 captured +0.27@10) ===")
summarize("q", lay_q[16]); summarize("k", lay_k[16]); summarize("v", lay_v[16])
print("=== all-layer beyond-+4 rate ===")
rows = []
for l in range(24):
    for nm, h in (("q", lay_q[l]), ("k", lay_k[l]), ("v", lay_v[l])):
        t = sum(h.values())
        if t:
            rows.append((l, nm, sum(v for k, v in h.items() if k > 4) / t, t))
for l, nm, r, t in sorted(rows, key=lambda x: -x[2])[:12]:
    print(f"  L{l} {nm}: {r*100:.1f}% of {t} hard blocks pick offset>+4")
