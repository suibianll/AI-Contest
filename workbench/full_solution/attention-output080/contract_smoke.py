"""CPU-only A-JC1 contract checks; synthetic evidence, not a 4B candidate."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('ref', ROOT / 'evaluator/reference_hif4.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
torch.set_num_threads(2)
torch.manual_seed(0)


def features(x, p):
    step = p['scale_factor'] * p['scale_lv2'] * p['scale_lv3'] / 4
    step = step.expand_as(p['mant']).double()
    m = p['mant'].double() * 4
    e = (x.double().reshape_as(m).abs() / step - m).clamp(-1, 1)
    phi = torch.stack((torch.ones_like(e), e, m / 7,
                       e.mean(-1, keepdim=True).expand_as(e)), -1)
    b = p['sign'].double().unsqueeze(-1) * step.unsqueeze(-1) * phi
    return phi, b.reshape(*x.shape, 4)


def hard(p, phi, theta):
    out = {k: v.clone() for k, v in p.items()}
    change = (phi @ theta).clamp(-1, 1).round()
    code = (4 * p['mant'].double() + change).clamp(0, 7)
    code = torch.where(p['sign'] != 0, code, torch.zeros_like(code))
    out['mant'] = (code / 4).to(p['mant'].dtype)
    out['sign'] = torch.where(code == 0, torch.zeros_like(p['sign']), p['sign'])
    return out


def attention(q, k, v):
    # Two Q heads share one K/V head; no cross-head mixing.
    q = q.reshape(8, 2, 64).transpose(0, 1)
    k = k.reshape(8, 1, 64).transpose(0, 1).expand(2, -1, -1)
    v = v.reshape(8, 1, 64).transpose(0, 1).expand(2, -1, -1)
    logits = q @ k.transpose(-1, -2) / 8
    mask = torch.ones(8, 8, dtype=torch.bool).triu(1)
    return logits.masked_fill(mask, -torch.inf).softmax(-1) @ v


def main():
    q, k, v = torch.randn(8, 128), torch.randn(8, 64), torch.randn(8, 64)
    pq, pk, pv = [ref.encode_standard_hif4(x) for x in (q, k, v)]
    q0, k0, v0 = [ref.decode_standard_hif4(p).double() for p in (pq, pk, pv)]
    fq, bq = features(q, pq)
    fk, bk = features(k, pk)
    zero = torch.zeros(8, dtype=torch.float64)
    for x, p, f in ((q, pq, fq), (k, pk, fk)):
        pz = hard(p, f, zero[:4])
        assert all(torch.equal(pz[n], p[n]) for n in p)
        forced = hard(p, f, torch.tensor([1., 0., 0., 0.], dtype=torch.float64))
        ref.validate_hif4_params(forced, x.shape)
        assert (forced['mant'] != p['mant']).any()
        assert all(torch.equal(forced[n], p[n]) for n in ('scale_factor', 'scale_lv2', 'scale_lv3'))

    def relaxed(t):
        return attention(q0 + bq @ t[:4], k0 + bk @ t[4:], v0)

    target = attention(q.double(), k.double(), v.double())
    residual = (relaxed(zero) - target).flatten()
    j = torch.autograd.functional.jacobian(relaxed, zero).reshape(-1, 8)
    h = j.T @ j / residual.numel()
    g = j.T @ residual / residual.numel()
    ridge = max(1e-4 * float(h.trace()) / 8, 1e-12)
    theta = torch.linalg.solve(h + ridge * torch.eye(8, dtype=h.dtype), -g)
    direction = torch.arange(1, 9, dtype=torch.float64)
    direction /= direction.norm()
    eps = 1e-5
    fd = ((relaxed(eps * direction) - relaxed(-eps * direction)) / (2 * eps)).flatten()
    rel = float((fd - j @ direction).norm() / fd.norm())
    assert rel < 1e-6
    assert torch.linalg.eigvalsh(h).min() > -1e-10
    before_model = float(residual.square().mean())
    after_model = float((residual + j @ theta).square().mean() + ridge * theta.square().sum())
    assert after_model <= before_model + 1e-12
    aq, ak = hard(pq, fq, theta[:4]), hard(pk, fk, theta[4:])
    for x, p in ((q, aq), (k, ak)):
        ref.validate_hif4_params(p, x.shape)
    actual = attention(ref.decode_standard_hif4(aq).double(), ref.decode_standard_hif4(ak).double(), v0)
    result = dict(scope='synthetic CPU contract only; not 4B or official evidence',
                  parent_sha256=hashlib.sha256((ROOT / 'solution.py').read_bytes()).hexdigest(),
                  smooth_jvp_fd_relative_error=rel,
                  qk_cross_hessian_norm=float(h[:4, 4:].norm()),
                  parent_mse=before_model, regularized_model_mse=after_model,
                  hard_mse=float((actual-target).square().mean()),
                  changed_q=int((aq['mant'] != pq['mant']).sum()),
                  changed_k=int((ak['mant'] != pk['mant']).sum()),
                  theta=theta.tolist(), zero_control=True, forced_reachability=True,
                  legal_state=True, frozen_scales=True)
    (Path(__file__).parent / 'contract-results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
