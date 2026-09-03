import json
import torch

c = torch.load(r'artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt', map_location='cpu', weights_only=False)
j = json.load(open(r'artifacts\official_eval\v160-a2-attn-default-candidate.json', encoding='utf-8'))
r = j['results'][0]
k_only_by_layer = {}
for cs in r['case_scores']['attention']:
    k_only_by_layer.setdefault(cs['layer'], []).append(cs['gain_k_only'])

rows = []
for layer in range(24):
    kw = c['weights'][layer]['k'].to(torch.float32)          # (128, 896)
    # dense K (post-projection) across all calibration folds
    k_dense = torch.cat([c['calibration_qkv'][f][layer][1] for f in range(5)], dim=0).to(torch.float32)  # (T,128)
    T = k_dense.shape[0]
    ch_rms = k_dense.square().mean(dim=0).sqrt()             # (128,)
    kv, hd = 2, 64
    head_rms = ch_rms.reshape(kv, hd)
    # structural features
    spread = float(ch_rms.max() / ch_rms.median())           # channel outlier ratio
    head_ratio = float(head_rms.mean(dim=1).max() / head_rms.mean(dim=1).min())
    # per-head-dim imbalance: mean over heads of (max/median within head)
    within = (head_rms.max(dim=1).values / head_rms.median(dim=1).values).mean().item()
    # weight row structure
    w_rms = kw.square().mean(dim=1).sqrt()
    w_spread = float(w_rms.max() / w_rms.median())
    # K-only gain for this layer (average over its 5 length cases)
    kl = sum(k_only_by_layer[layer]) / len(k_only_by_layer[layer])
    rows.append((layer, kl, spread, head_ratio, within, w_spread))

print(f"{'ly':>3} {'K-only':>10} {'ch_spread':>10} {'head_ratio':>10} {'within':>8} {'w_spread':>9}")
for row in rows:
    print(f"{row[0]:>3} {row[1]:>10.2f} {row[2]:>10.2f} {row[3]:>10.3f} {row[4]:>8.3f} {row[5]:>9.2f}")

kl = torch.tensor([r[1] for r in rows])
feats = {'ch_spread': 2, 'head_ratio': 3, 'within': 4, 'w_spread': 5}
print('\nSpearman-style rank corr with K-only gain (more negative gain = worse):')
k_rank = kl.argsort().argsort().to(torch.float32)
for name, idx in feats.items():
    v = torch.tensor([r[idx] for r in rows])
    v_rank = v.argsort().argsort().to(torch.float32)
    corr = torch.corrcoef(torch.stack([k_rank, v_rank]))[0, 1].item()
    # also on layer 21-23 (deep tail) vs rest
    deep = torch.tensor([r[idx] for r in rows[21:]])
    rest = torch.tensor([r[idx] for r in rows[:21]])
    print(f"  {name:>10}: rankcorr={corr:+.3f}   deep(21-23) mean={deep.mean():.3f}  rest mean={rest.mean():.3f}")
