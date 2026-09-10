"""Where the L-EM2 calibration hook's +4.8 s actually goes.

The paired calibration measurement (``time_calibration.py``) charges the compile
hook +0.0139 s at in=2560 and +0.1289 s at in=4096, i.e. +4.8 s over the
official 144 in-scope calibrations.  With the dynamic side now down to +6.5 s
(K=1), that hook is the single largest remaining cost, so this profiler splits
it into its pieces on the real layer0/o state -- including the diagnostic
temporaries, which allocate several n^2 tensors per call and are pure overhead.

Read-only CPU/GPU probe on the real shard cache; prints only.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))

import official_eval as v2  # noqa: E402

CACHE = (
    ROOT
    / "artifacts/official_eval/cache/proxy-v3-calibration"
    / "56dc805d6e5a3aef-linear-452e6fe4aa5747590535.pt"
)
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"


def timed(label: str, fn, device, repeats: int = 5) -> float:
    fn()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize(device)
    seconds = (time.perf_counter() - started) / repeats
    print(f"  {label:<44} {seconds * 1e3:8.2f} ms")
    return seconds


def main() -> None:
    device = torch.device("cuda")
    payload = torch.load(CACHE, map_location="cpu", mmap=True, weights_only=False)
    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    for layer, role in ((0, "o"), (0, "q")):
        entry = next(
            item
            for item in payload["weight_states"]
            if int(item["layer"]) == layer and str(item["role"]) == role
        )
        state = entry["state"]
        weight_quant, weight_scale = v2._pair(pack["weights"][layer][role].to(torch.float32))
        weight_quant = weight_quant.to(device)
        weight_scale = weight_scale.to(device)
        channels = int(state["in_features"])
        h_inv = state["h_inv"].to(device=device, dtype=torch.float32)
        print(f"layer{layer}/{role} in={channels} h_inv={tuple(h_inv.shape)}")
        logical_shape = v2.dequantize_nvfp4(weight_quant, weight_scale).shape
        params = {key: value.to(device) for key, value in entry["params"].items()}
        dense = v2.dequantize_nvfp4(weight_quant, weight_scale).to(torch.float32)
        deployed = v2.dequantize_hif4(params, logical_shape).to(torch.float32)
        gram = deployed.transpose(0, 1).mm(deployed)
        cross = dense.transpose(0, 1).mm(deployed)
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(h_inv))
        ridge = float((inverse - gram).diagonal().mean())

        timed(
            "dequantize_nvfp4_float32 (ref frame)",
            lambda: v2.dequantize_nvfp4(weight_quant, weight_scale).to(torch.float32),
            device,
        )
        timed(
            "dequantize_hif4(weight_params)",
            lambda: v2.dequantize_hif4(params, logical_shape).to(torch.float32),
            device,
        )
        timed(
            "gram = deployed.T @ deployed",
            lambda: deployed.transpose(0, 1).mm(deployed),
            device,
        )
        timed(
            "cross = dense.T @ deployed",
            lambda: dense.transpose(0, 1).mm(deployed),
            device,
        )
        timed("h_matrix = gram - cross", lambda: gram - cross, device)
        timed("cholesky", lambda: torch.linalg.cholesky(h_inv), device)
        timed(
            "cholesky_inverse(cholesky(h_inv))",
            lambda: torch.cholesky_inverse(torch.linalg.cholesky(h_inv)),
            device,
        )
        eye = torch.eye(channels, device=device, dtype=torch.float32)
        timed("torch.linalg.inv(h_inv)", lambda: torch.linalg.inv(h_inv), device)
        timed(
            "torch.linalg.solve(h_inv, eye)",
            lambda: torch.linalg.solve(h_inv, eye),
            device,
        )
        factor = torch.linalg.cholesky(h_inv)
        timed(
            "cholesky_solve(eye, L)",
            lambda: torch.cholesky_solve(eye, factor),
            device,
        )
        timed(
            "lu_factor + lu_solve",
            lambda: torch.linalg.lu_solve(*torch.linalg.lu_factor(h_inv), eye),
            device,
        )
        timed("linalg.inv_ex", lambda: torch.linalg.inv_ex(h_inv)[0], device)
        timed(
            "ridge = mean(diag(inverse - gram))  [n^2 temp]",
            lambda: float((inverse - gram).diagonal().mean()),
            device,
        )
        timed(
            "ridge = mean(diag(inv)) - mean(diag(gram))  [no temp]",
            lambda: float(inverse.diagonal().mean()) - float(gram.diagonal().mean()),
            device,
        )
        timed(
            "residual = inverse - gram - ridge*eye  [2 n^2 temps]",
            lambda: inverse
            - gram
            - ridge * torch.eye(channels, device=device, dtype=torch.float32),
            device,
        )
        timed("h_matrix.norm()", lambda: float((gram - cross).norm()), device)
        timed(
            "residual.norm() + gram.norm()",
            lambda: float(
                (
                    inverse
                    - gram
                    - ridge * torch.eye(channels, device=device, dtype=torch.float32)
                ).norm()
                / gram.norm().clamp_min(1.0e-30)
            ),
            device,
        )
        timed(
            "h.contiguous() D2H copy",
            lambda: (gram - cross).to("cpu", torch.float32).contiguous(),
            device,
        )
        total = timed(
            "FULL hook (current implementation, incl. dequant)",
            lambda: _full(params, logical_shape, weight_quant, weight_scale, h_inv, device),
            device,
        )
        lean = timed(
            "LEAN hook (no residual, diag-only ridge)",
            lambda: _lean(params, logical_shape, weight_quant, weight_scale, h_inv, device),
            device,
        )
        print(
            f"  -> current {total * 1e3:.2f} ms vs lean {lean * 1e3:.2f} ms "
            f"(saves {(total - lean) * 1e3:.2f} ms/call)\n"
        )


def _prepare(params, logical_shape, weight_quant, weight_scale, h_inv):
    dense = v2.dequantize_nvfp4(weight_quant, weight_scale).to(torch.float32)
    deployed = v2.dequantize_hif4(params, logical_shape).to(torch.float32)
    gram = deployed.transpose(0, 1).mm(deployed)
    cross = dense.transpose(0, 1).mm(deployed)
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(h_inv))
    return gram, cross, inverse


def _full(params, logical_shape, weight_quant, weight_scale, h_inv, device):
    gram, cross, inverse = _prepare(params, logical_shape, weight_quant, weight_scale, h_inv)
    channels = gram.shape[0]
    h_matrix = gram - cross
    ridge = float((inverse - gram).diagonal().mean())
    residual = inverse - gram - ridge * torch.eye(
        channels, device=device, dtype=torch.float32
    )
    _ = float(residual.norm() / gram.norm().clamp_min(1.0e-30))
    _ = float(h_matrix.norm())
    return h_matrix.to("cpu", torch.float32).contiguous()


def _lean(params, logical_shape, weight_quant, weight_scale, h_inv, device):
    gram, cross, inverse = _prepare(params, logical_shape, weight_quant, weight_scale, h_inv)
    h_matrix = gram - cross
    _ = float(inverse.diagonal().mean() - gram.diagonal().mean())
    _ = float(h_matrix.norm())
    return h_matrix.to("cpu", torch.float32).contiguous()


if __name__ == "__main__":
    main()
