"""SM121 generic-route CUDA-graph gate for the B12X trellis linear.

This is the gate the SM121 scratch-planner fix exists for: on GB10 the
capture-safe K6 small-M kernel is not admitted (b12x _use_k6_small
gates on capability (12, 0)),
so `_b12x_trellis_linear` runs the generic W4A16 kernel at small M - the
exact configuration every GLM-EXL3 capture size (4..32) uses on II.
Without the planner fix, capture raises on the one-float scratch buffer.

Shapes are the GLM-5.2 TP4 shared-expert projections (our II GLM-EXL3
serving layout)
(hidden 6144, shared-expert intermediate 2048, merged column-parallel
gate/up and row-parallel down):
  - gate/up TP4 shard: 6144 -> 1024
  - down    TP4 shard:  512 -> 6144
These compile different K/N tile families; both are exercised. Steps per
shape:
  1. Encode K6/MCG online (same quantize_exl3 convention as serving).
  2. Eager `_b12x_trellis_linear` at M=4 and M=32; compare against the
     ExLlamaV3 reference (reconstruct + rotations + hgemm).
  3. Capture a CUDA graph around `_b12x_trellis_linear` at M=4 and M=32,
     replay it twice with fresh inputs, and compare replayed outputs
     against eager outputs for the same inputs.
"""

import sys

import torch

from vllm.model_executor.layers.quantization.exl3 import (
    _b12x_trellis_linear,
    _load_exl3_online_quantizer,
)

DEV = torch.device("cuda:0")
CODEBOOK = "mcg"
SHAPES = [
    (6144, 1024, "shared gate/up TP4 shard"),
    (512, 6144, "shared down TP4 shard"),
]
failures = []

cap = torch.cuda.get_device_capability(DEV)
print(f"device capability: {cap} (generic route expected on non-(12,0))")

quantize_exl3 = _load_exl3_online_quantizer()

sys.path.insert(0, "/opt/exllamav3")
import exllamav3_ext as ext  # noqa: E402


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.float() - b.float()).norm().item()
    return d / max(b.float().norm().item(), 1e-6)


for in_features, out_features, shape_label in SHAPES:
    print(f"=== {shape_label}: {in_features} -> {out_features} ===")
    torch.manual_seed(in_features + out_features)
    weight = torch.randn(out_features, in_features, device=DEV) * 0.02
    source = weight.detach().t().float().contiguous()
    quant_args = {
        "K": 6,
        "seed": 0,
        "devices": [DEV],
        "apply_out_scales": True,
        CODEBOOK: True,
    }
    _, _, tensors = quantize_exl3(
        source,
        {
            "H": torch.zeros(
                in_features, in_features, dtype=torch.float32, device="meta"
            ),
            "first_key": f"graph-gate-{in_features}x{out_features}",
            "count": 0,
            "finalized": False,
            "num_total": 0,
            "inf_nan": torch.zeros(2, dtype=torch.long, device=DEV),
            "device": DEV,
        },
        quant_args,
        return_weight_q=False,
        verbose=False,
    )
    trellis = tensors["trellis"].to(DEV).contiguous()
    suh = tensors["suh"].to(DEV).half().contiguous()
    svh = tensors["svh"].to(DEV).half().contiguous()

    w_recon = torch.empty((in_features, out_features), dtype=torch.half, device=DEV)
    ext.reconstruct(w_recon, trellis, 6, CODEBOOK == "mcg", CODEBOOK == "mul1")

    def reference(x: torch.Tensor) -> torch.Tensor:
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, suh, None, 1.0)
        y = torch.empty((x.shape[0], out_features), dtype=torch.half, device=DEV)
        ext.hgemm(xh, w_recon, y)
        ext.had_r_128(y, y, None, svh, 1.0)
        return y

    for m in (4, 32):
        torch.manual_seed(m)
        x = (torch.randn(m, in_features, device=DEV) * 0.5).half()

        y_eager = _b12x_trellis_linear(x, trellis, suh, svh)
        r_ref = rel(y_eager, reference(x))
        print(f"{shape_label} M={m}: eager vs reference rel={r_ref:.2e}")
        if not torch.isfinite(y_eager.float()).all():
            failures.append(f"{shape_label} M={m} eager output non-finite")
        if r_ref > 2e-2:
            failures.append(f"{shape_label} M={m} eager vs reference rel {r_ref:.3e}")

        static_x = x.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                _b12x_trellis_linear(static_x, trellis, suh, svh)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                static_y = _b12x_trellis_linear(static_x, trellis, suh, svh)
        except RuntimeError as exc:
            failures.append(f"{shape_label} M={m} graph capture raised: {exc}")
            print(f"{shape_label} M={m}: GRAPH CAPTURE FAILED: {exc}")
            continue

        for trial in range(2):
            torch.manual_seed(1000 + m + trial)
            x_new = (torch.randn(m, in_features, device=DEV) * 0.5).half()
            static_x.copy_(x_new)
            graph.replay()
            torch.cuda.synchronize()
            y_replay = static_y.clone()
            y_check = _b12x_trellis_linear(x_new, trellis, suh, svh)
            r_replay = rel(y_replay, y_check)
            print(f"{shape_label} M={m} replay {trial}: rel={r_replay:.2e}")
            if not torch.isfinite(y_replay.float()).all():
                failures.append(f"{shape_label} M={m} replay {trial} non-finite")
            if r_replay > 1e-3:
                failures.append(
                    f"{shape_label} M={m} replay {trial} diverges (rel {r_replay:.3e})"
                )

if failures:
    print("B12X-TRELLIS-GRAPH-FAIL:", "; ".join(failures))
    sys.exit(1)
print("B12X-TRELLIS-GRAPH-OK")
