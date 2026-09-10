import importlib.util, sys, torch
REPO = r"D:\工作内容\AI竞赛"
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
root = load(REPO + "/solution.py", "r")
cand = load(REPO + "/workbench/full_solution/attention-afix1-train-deploy-align/candidate/solution.py", "c")
MAG = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0])
QH, KVH, HD, T, NW = 4, 2, 64, 96, 4
def mk(shape, seed):
    g = torch.Generator().manual_seed(seed)
    q = (MAG[torch.randint(0,8,shape,generator=g)] * (torch.randint(0,2,shape,generator=g)*2-1)).float()
    s = (torch.rand(shape[:-1]+(shape[-1]//16,), generator=g)*0.02+0.002).float()
    return q, s
for seed in range(10):
    wins = [{"q": mk((T,QH*HD), seed*100+100+w), "k": mk((T,KVH*HD), seed*100+200+w), "v": mk((T,KVH*HD), seed*100+300+w)} for w in range(NW)]
    sr = root.hif4_calibration_attention(wins, QH, KVH, HD)
    sc = cand.hif4_calibration_attention(wins, QH, KVH, HD)
    ar, ac = sr["q_state"].get("a2_arm"), sc["q_state"].get("a2_arm")
    line = f"seed={seed} root_arm={ar} cand_arm={ac} root_gate={sr['q_state'].get('a2_gate_loss_rotation'):.8f} cand_gate={sc['q_state'].get('a2_gate_loss_rotation'):.8f}"
    if ar == "rotation" and ac == "rotation":
        rd = float((sr["q_state"]["learned_rotation"] - sc["q_state"]["learned_rotation"]).norm())
        cd = float((sr["k_state"]["learned_center"] - sc["k_state"]["learned_center"]).norm())
        rn = float(sr["q_state"]["learned_rotation"].norm())
        line += f" | rot_diff={rd:.4f} (root||R||={rn:.4f}) center_diff={cd:.4f}"
        line += f" | cand train {sc['q_state']['a2_first_train_loss']:.6f}->{sc['q_state']['a2_train_loss']:.6f} root train {sr['q_state']['a2_train_loss']:.6f}"
    print(line)
