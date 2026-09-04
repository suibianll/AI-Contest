"""Oracle-methodology round 4 (2026-09-04, post-v186): E1 verification + remaining
error re-concentration.  Pure JSON analysis, zero GPU, zero quota.

A)  oracle(+/-24) vs deployed(+4) window residual: global / L21 / per-layer
A2) timing-like numbers in both diag JSONs (wide-window calibration cost)
B)  calibration state diff oracle vs v186 (E1: state quality under wide window)
B2) per-layer l2 rel-diff of tensor state keys
C)  remaining-error concentration on plus4 (current attention parent)
"""
import json
from collections import defaultdict

ROOT = "artifacts/official_eval"


def by_cell(path):
    d = json.load(open(f"{ROOT}/{path}", encoding="utf-8"))
    cells = {}
    for c in d["results"][0]["case_scores"]["attention"]:
        cells[(c["layer"], c.get("test_length", c.get("length")))] = c["gain"]
    return cells


oracle = by_cell("diag-oracle-attn-default.json")
plus4 = by_cell("diag-plus4-attn-default.json")
base = by_cell("v180-attn-default.json")
common = [k for k in oracle if k in plus4 and k in base]
print(f"[A] common cells: {len(common)}")

res = [oracle[k] - plus4[k] for k in common]
print(f"[A] GLOBAL residual oracle-minus-plus4: "
      f"mean={sum(res) / len(res):+.6f}  "
      f"pos={sum(1 for r in res if r > 0)}  neg={sum(1 for r in res if r < 0)}")
go = sum(oracle[k] - base[k] for k in common) / len(common)
gp = sum(plus4[k] - base[k] for k in common) / len(common)
print(f"[A] GLOBAL gain vs v180-base: oracle={go:+.6f}  plus4={gp:+.6f}")

print()
print("[A] L21 per-length")
for ln in sorted(k[1] for k in common if k[0] == 21):
    k = (21, ln)
    print(f"  len={ln:5d}  oracle={oracle[k]:+.6f}  plus4={plus4[k]:+.6f}  "
          f"res={oracle[k] - plus4[k]:+.6f}")

print()
print("[A] per-layer mean residual (sorted desc)")
rl = defaultdict(list)
for k in common:
    rl[k[0]].append(oracle[k] - plus4[k])
for layer, v in sorted(rl.items(), key=lambda kv: -sum(kv[1])):
    print(f"  L{layer:2d}  mean_res={sum(v) / len(v):+.6f}")

print()
print("[A2] timing-like numbers in diag JSONs (case_scores skipped, cap 30)")


def find_num(obj, path, out):
    if len(out) >= 30 or "case_scores" in path:
        return
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            v = obj[k]
            kl = str(k).lower()
            if (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and ("time" in kl or "duration" in kl or "elapsed" in kl)):
                out.append((f"{path}.{k}", v))
            elif isinstance(v, (dict, list)):
                find_num(v, f"{path}.{k}", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_num(v, f"{path}[{i}]", out)


for name, path in [("oracle", "diag-oracle-attn-default.json"),
                   ("plus4", "diag-plus4-attn-default.json")]:
    d = json.load(open(f"{ROOT}/{path}", encoding="utf-8"))
    out = []
    find_num(d, "", out)
    print(f"  {name}:")
    for p, v in out:
        print(f"    {p} = {v}")

print()
print("[C] remaining-error concentration on plus4 (current attention parent)")
d = json.load(open(f"{ROOT}/diag-plus4-attn-default.json", encoding="utf-8"))
cases = d["results"][0]["case_scores"]["attention"]
total = sum(c["mse_player"] for c in cases)
srt = sorted(cases, key=lambda c: -c["mse_player"])
for k in (6, 12, 24, 60):
    share = sum(c["mse_player"] for c in srt[:k]) / total
    print(f"  top {k:3d}/120 hold {share * 100:5.1f}% of remaining error")
blen = defaultdict(float)
for c in cases:
    blen[c.get("test_length", c.get("length", -1))] += c["mse_player"]
print("  by length:", [(l, f"{v / total * 100:.1f}%") for l, v in sorted(blen.items())])
bw = defaultdict(float)
for c in cases:
    bw[c["layer"]] += c["mse_player"]
print("  worst layers:",
      [(l, f"{v / total * 100:.1f}%") for l, v in sorted(bw.items(), key=lambda kv: -kv[1])[:6]])

print()
print("[B] calibration state diff: oracle vs v186 (assumes row index == layer)")


def load_states(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


so = load_states("logs/execution/diag_l21_state_oracle.jsonl")
sv = load_states("logs/execution/diag_l21_state_v186.jsonl")
print(f"  rows: oracle={len(so)}  v186={len(sv)}")
print(f"  row0 top keys: {sorted(so[0].keys())}")
print(f"  q keys: {sorted(so[0]['q'].keys())}")


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", "ONE_SIDE_ONLY"))
            else:
                walk(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"LEN {len(a)} vs {len(b)}"))
        else:
            for i, (xa, xb) in enumerate(zip(a, b)):
                if xa != xb:
                    out.append((f"{path}[{i}]", f"{xa} vs {xb}"))
    elif a != b:
        out.append((path, f"{a} vs {b}"))


diffs = {}
for i in range(min(len(so), len(sv))):
    out = []
    for side in ("q", "k", "v"):
        walk(so[i].get(side, {}), sv[i].get(side, {}), side, out)
    diffs[i] = out

print()
print("[B] per-layer differing leaf count + key families")
for i in sorted(diffs):
    if diffs[i]:
        fams = sorted(set(p.split("[")[0] for p, _ in diffs[i]))
        print(f"  L{i:2d}: {len(diffs[i]):5d} leaf diffs -> {fams}")
print("  identical layers:", [i for i in sorted(diffs) if not diffs[i]])

print()
print("[B2] per-layer l2 rel-diff for tensor state keys")
for i in range(min(len(so), len(sv))):
    parts = []
    for side in ("q", "k", "v"):
        for k in sorted(so[i].get(side, {})):
            va = so[i][side].get(k)
            vb = sv[i].get(side, {}).get(k)
            if isinstance(va, dict) and isinstance(vb, dict) and "l2" in va and "l2" in vb:
                la, lb = va["l2"], vb["l2"]
                m = max(abs(la), abs(lb), 1e-12)
                parts.append(f"{side}.{k}={abs(la - lb) / m:.2e}")
    if parts:
        print(f"  L{i:2d}: " + " ".join(parts))

print()
print("[B] L21 detail (first 80 leaf diffs)")
for p, v in diffs.get(21, [])[:80]:
    print(f"  {p}: {v}")
if len(diffs.get(21, [])) > 80:
    print(f"  ... +{len(diffs[21]) - 80} more")
