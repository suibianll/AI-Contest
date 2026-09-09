"""Q1/Q3/Q4: read A-G1 (v227) and root calibration caches, extract per-layer A2 audit.

Pure CPU. Reads proxy-v3 calibration cache files:
  artifacts/official_eval/cache/proxy-v3-calibration/<sha16>-attention-<hash>.pt
Each file holds one shard's attention_states (one full-attention layer).
"""
import glob
import json
import math
import os

import torch

CACHE_DIR = r"artifacts/official_eval/cache/proxy-v3-calibration"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_calibration_audit.json")

CAND_PREFIX = "165e1a6bc50abd41"   # v227 / A-G1
ROOT_PREFIX = "56dc805d6e5a3aef"   # current root (v202 linear + v195 attention)

# panel full-attention layer -> shard (shard_layers: range(shard, 24, 6))
LAYER_TO_SHARD = {0: 0, 1: 1, 8: 2, 15: 3, 22: 4, 5: 5}
# panel layer -> model layer (panel_model_layers from candidate-attention-shard0.json)
PANEL_TO_MODEL = {0: 3, 1: 7, 5: 11, 8: 19, 15: 23, 22: 31}


def load_layer_records(prefix):
    records = {}
    for path in sorted(glob.glob(os.path.join(CACHE_DIR, prefix + "-attention-*.pt"))):
        pack = torch.load(path, map_location="cpu", weights_only=False)
        for item in pack["attention_states"]:
            layer = int(item["layer"])
            states = item["states"]
            qs = states["q_state"]
            ks = states["k_state"]
            rec = {
                "cache_file": os.path.basename(path),
                "layer": layer,
                "model_layer": PANEL_TO_MODEL.get(layer),
                "shard": LAYER_TO_SHARD.get(layer),
                "a2_arm": qs.get("a2_arm"),
                "a2_gate_loss_identity": qs.get("a2_gate_loss_identity"),
                "a2_gate_loss_rotation": qs.get("a2_gate_loss_rotation"),
                "a2_train_loss": qs.get("a2_train_loss"),
                "a2_steps": qs.get("a2_steps"),
                "a2_ortho_error": qs.get("a2_ortho_error"),
                "a2_scale_l2": qs.get("a2_scale_l2"),
                "a2_scale_max_abs": qs.get("a2_scale_max_abs"),
                "a2_scale_nonzero": qs.get("a2_scale_nonzero"),
                "has_learned_rotation": "learned_rotation" in qs,
                "has_learned_scale": "learned_scale" in qs,
                "has_learned_center": "learned_center" in ks,
            }
            li, lr = rec["a2_gate_loss_identity"], rec["a2_gate_loss_rotation"]
            if isinstance(li, float) and isinstance(lr, float) and li > 0:
                rec["gate_rel_improvement"] = (li - lr) / li
            else:
                rec["gate_rel_improvement"] = None
            s = qs.get("learned_scale")
            if torch.is_tensor(s):
                s = s.to(torch.float32)
                lim = math.log(2.0)
                rec["scale_shape"] = list(s.shape)
                rec["scale_numel"] = int(s.numel())
                rec["scale_clamp_frac"] = float((s.abs() > lim - 1e-4).float().mean())
                rec["scale_mean_abs"] = float(s.abs().mean())
                rec["scale_std"] = float(s.std())
                # per-channel dispersion: std across the head_dim axis per group,
                # then summarize; also zero-mean check per group row
                if s.ndim == 2:
                    rec["scale_row_mean_absmax"] = float(s.mean(dim=-1).abs().max())
                    rec["scale_perrow_std_mean"] = float(s.std(dim=-1).mean())
                    rec["scale_perrow_std_max"] = float(s.std(dim=-1).max())
                flat = s.reshape(-1)
                rec["scale_p05"] = float(flat.quantile(0.05))
                rec["scale_p95"] = float(flat.quantile(0.95))
            rot = qs.get("learned_rotation")
            if torch.is_tensor(rot):
                rec["rotation_shape"] = list(rot.shape)
            records[layer] = rec
    return records


def main():
    cand = load_layer_records(CAND_PREFIX)
    root = load_layer_records(ROOT_PREFIX)
    out = {"candidate_v227": cand, "root": root}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)

    hdr = (f"{'layer':>5} {'shard':>5} {'arm':>14} {'gate_id':>10} {'gate_rot':>10} "
           f"{'rel_impr':>9} {'train_loss':>10} {'|s|max':>7} {'clamp%':>7} {'|s|mean':>8}")
    for tag, recs in (("CANDIDATE v227", cand), ("ROOT", root)):
        print(f"== {tag} ==")
        print(hdr)
        for layer in sorted(recs):
            r = recs[layer]
            ri = r.get("gate_rel_improvement")
            print(f"{r['layer']:>5} {r['shard']:>5} {str(r['a2_arm']):>14} "
                  f"{r['a2_gate_loss_identity'] if r['a2_gate_loss_identity'] is not None else float('nan'):>10.6f} "
                  f"{r['a2_gate_loss_rotation'] if r['a2_gate_loss_rotation'] is not None else float('nan'):>10.6f} "
                  f"{ri if ri is not None else float('nan'):>9.5f} "
                  f"{r['a2_train_loss'] if r['a2_train_loss'] is not None else float('nan'):>10.6f} "
                  f"{r.get('a2_scale_max_abs') if r.get('a2_scale_max_abs') is not None else float('nan'):>7.4f} "
                  f"{(r.get('scale_clamp_frac') or 0.0) * 100:>6.1f}% "
                  f"{r.get('scale_mean_abs') if r.get('scale_mean_abs') is not None else float('nan'):>8.4f}")
        print()


if __name__ == "__main__":
    main()
