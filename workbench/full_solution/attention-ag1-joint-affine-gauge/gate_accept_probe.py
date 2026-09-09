import importlib.util, os, sys, torch
REPO = r"D:\工作内容\AI竞赛"
sys.path.insert(0, REPO)
from evaluator.reference_hif4 import validate_state

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

cand = load(os.path.join(REPO, "workbench/full_solution/attention-ag1-joint-affine-gauge/candidate/solution.py"), "cand")
MAG = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0])
QH, KVH, HD, T, NW = 4, 2, 64, 96, 4

def mk(shape, seed, channel_gain=None):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 8, shape, generator=g)
    sign = torch.randint(0, 2, shape, generator=g)*2-1
    q = (MAG[idx]*sign).float()
    if channel_gain is not None:
        q = q * channel_gain
    s = (torch.rand(shape[:-1]+(shape[-1]//16,), generator=g)*0.02+0.002).float()
    return q, s

accepted = 0
for seed in range(12):
    gain_q = torch.ones(QH*HD); gain_q[::3] = 4.0
    gain_k = torch.ones(KVH*HD); gain_k[1::3] = 0.25
    wins = [{"q": mk((T,QH*HD), seed*10+w, gain_q),
             "k": mk((T,KVH*HD), seed*10+w+100, gain_k),
             "v": mk((T,KVH*HD), seed*10+w+200)} for w in range(NW)]
    st = cand.hif4_calibration_attention(wins, QH, KVH, HD)
    arm = st["q_state"].get("a2_arm")
    print(f"seed={seed} arm={arm} "
          f"gate_id={st['q_state'].get('a2_gate_loss_identity'):.8f} "
          f"gate_cand={st['q_state'].get('a2_gate_loss_rotation'):.8f} "
          f"||s||={st['q_state'].get('a2_scale_l2'):.4f} "
          f"nz={st['q_state'].get('a2_scale_nonzero')}")
    if arm == "rotation+scale":
        accepted += 1
        ls_q = st["q_state"]["learned_scale"]; ls_k = st["k_state"]["learned_scale"]
        validate_state(st["q_state"]); validate_state(st["k_state"]); validate_state(st["v_state"])
        print(f"   ACCEPT: learned_scale cpu={ls_q.device.type} dtype={ls_q.dtype} "
              f"shape={tuple(ls_q.shape)} finite={bool(torch.isfinite(ls_q).all())} "
              f"q_vs_k_equal={torch.equal(ls_q, ls_k)} zero_mean_max={float(ls_q.mean(-1).abs().max()):.2e} "
              f"max|s|={float(ls_q.abs().max()):.4f} (limit {0.6931:.4f})")
        qq, qs = mk((T,QH*HD), 4242, gain_q)
        p = cand.hif4_dynamic_quantize_q(qq, qs, QH, HD, st["q_state"])
        print(f"   dynamic_q finite={all(bool(torch.isfinite(t.float()).all()) for t in p.values())}")
if accepted == 0:
    print("NO_ACCEPT_IN_12_SEEDS")
