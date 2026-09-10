"""CPU dispatch-cost proxy for the exact sequential descent loop.

Mirrors the per-group op sequence of the runtime variant at realistic shapes
(rows=1024) and reports seconds per activation call for one role.
"""
import time, torch

def bench(channels, rows, n_cand, label, reps=1):
    blocks = channels // 64
    x = torch.randn(rows, channels)
    g = torch.randn(rows, channels)
    delta = torch.randn(n_cand, rows, 16, 4) * 1e-3
    g44 = torch.eye(4).expand(16, 4, 4).clone()
    gram = torch.randn(channels, channels)
    t0 = time.perf_counter()
    acc = 0
    for b in range(blocks):
        for grp in range(16):
            sl = slice(b * 64 + grp * 4, b * 64 + grp * 4 + 4)
            d = delta[:, :, grp, :]
            g4 = g[:, sl]
            lin = torch.einsum("kri,ri->kr", d, g4)
            quad = torch.einsum("kri,ij,krj->kr", d, g44[grp], d)
            cost = 2.0 * lin + quad
            best_k = cost.argmin(dim=0)
            best_c = cost.gather(0, best_k.unsqueeze(0)).squeeze(0)
            take = best_c < 0.0
            chosen = d.gather(0, best_k.reshape(1, rows, 1).expand(1, rows, 4)).squeeze(0)
            chosen = torch.where(take.unsqueeze(-1), chosen, torch.zeros_like(chosen))
            x[:, sl] += chosen
            g += chosen @ gram[sl, :]
            acc += int(take.sum())
    dt = (time.perf_counter() - t0) / reps
    print(f"{label:>6} ch={channels:5d} rows={rows:5d} blocks={blocks:3d} "
          f"groups={blocks*16:4d} -> {dt*1000:7.1f} ms/call  (accepted={acc})")

for rows in (8, 128, 1024):
    bench(2560, rows, 8, f"r{rows}")
for rows in (8, 128, 1024):
    bench(4096, rows, 8, f"r{rows}")
