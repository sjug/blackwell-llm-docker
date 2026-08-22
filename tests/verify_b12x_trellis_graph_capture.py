"""SM121 CUDA-graph gate for the r17 dense-Trellis path (#221 namespace).

r17 replaced the (12,0)-only cooperative kernel with the CuTe-DSL
K6/MCG small-M kernel in `b12x.gemm.trellis_linear`, admissible on
SM120 AND SM121, bound per-weight at prepare time and dispatched for
m <= 16. This gate proves the whole contract on real GB10 silicon for
both GLM-5.2 TP4 shared-expert shapes (6144 -> 1024 merged gate/up,
512 -> 6144 down):

  1. `api.is_supported(device)` is True on this device.
  2. vLLM's prepare (`_b12x_trellis_weight`) BINDS the small-M launch
     for a production-convention online-K6/MCG weight, and the launch
     admits M <= 16 while rejecting M = 17 (planner/dispatch agreement).
  3. For M in {4, 16, 17, 32}: eager `_b12x_trellis_linear` matches the
     ExLlamaV3 reference; CUDA-graph capture succeeds; two replays with
     fresh inputs match eager. M=16/17 straddle the small/generic
     dispatch boundary - the exact seam the r15-era bug lived on.
  4. B12X_DISABLE_STANDALONE_K6 must be unset (incompatible runtime
     setting: it breaks planner/dispatch agreement by design).
"""

import os
import sys

import torch

from vllm.model_executor.layers.quantization.exl3 import (
    _b12x_trellis_linear,
    _b12x_trellis_weight,
    _load_b12x_trellis_linear,
    _load_exl3_online_quantizer,
)

DEV = torch.device("cuda:0")
CODEBOOK = "mcg"
SHAPES = [
    (6144, 1024, "shared gate/up TP4 shard"),
    (512, 6144, "shared down TP4 shard"),
]
failures = []

if "B12X_DISABLE_STANDALONE_K6" in os.environ:
    failures.append("B12X_DISABLE_STANDALONE_K6 present: incompatible setting")

api = _load_b12x_trellis_linear()
cap = torch.cuda.get_device_capability(DEV)
supported = bool(api.is_supported(DEV))
print(f"device capability: {cap} api.is_supported: {supported}")
if not supported:
    failures.append(f"trellis_linear api not supported on {cap}")

quantize_exl3 = _load_exl3_online_quantizer()

# Dispatch proof: wrap the b12x small-M entry point so the gate OBSERVES
# _b12x_trellis_linear invoking it (binding + accepts_input alone do not
# prove dispatch). vLLM imports it at call time, so patching the module
# attribute intercepts every dispatch.
import b12x.gemm.trellis_linear._k6_mcg_cute as _cute  # noqa: E402

_SMALL_M_CALLS = {"n": 0}
_orig_run_small = _cute.run_k6_mcg_small_m


def _counting_run_small(*args, **kwargs):
    _SMALL_M_CALLS["n"] += 1
    return _orig_run_small(*args, **kwargs)


_cute.run_k6_mcg_small_m = _counting_run_small

sys.path.insert(0, "/opt/exllamav3")
import exllamav3_ext as ext  # noqa: E402


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative error, fail-LOUD: non-finite operands return inf, never
    a NaN that would slip through `> bound` comparisons."""
    if not torch.isfinite(a.float()).all() or not torch.isfinite(b.float()).all():
        return float("inf")
    d = (a.float() - b.float()).norm().item()
    r = d / max(b.float().norm().item(), 1e-6)
    return r if r == r else float("inf")


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

    prepared = _b12x_trellis_weight(trellis, suh, svh, torch.float16)
    launch = getattr(prepared, "k6_mcg_small_m_launch", None)
    if launch is None:
        failures.append(f"{shape_label}: small-M launch NOT bound on {cap}")
    else:
        probe16 = torch.zeros(16, in_features, dtype=torch.float16, device=DEV)
        probe17 = torch.zeros(17, in_features, dtype=torch.float16, device=DEV)
        a16, a17 = launch.accepts_input(probe16), launch.accepts_input(probe17)
        print(f"{shape_label}: launch bound; accepts M16={a16} M17={a17}")
        if not a16:
            failures.append(f"{shape_label}: launch rejects M=16")
        if a17:
            failures.append(f"{shape_label}: launch wrongly admits M=17")

    w_recon = torch.empty((in_features, out_features), dtype=torch.half, device=DEV)
    ext.reconstruct(w_recon, trellis, 6, CODEBOOK == "mcg", CODEBOOK == "mul1")

    def reference(x: torch.Tensor) -> torch.Tensor:
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, suh, None, 1.0)
        y = torch.empty((x.shape[0], out_features), dtype=torch.half, device=DEV)
        ext.hgemm(xh, w_recon, y)
        ext.had_r_128(y, y, None, svh, 1.0)
        return y

    for m in (4, 16, 17, 32):
        torch.manual_seed(m)
        x = (torch.randn(m, in_features, device=DEV) * 0.5).half()

        before = _SMALL_M_CALLS["n"]
        y_eager = _b12x_trellis_linear(x, trellis, suh, svh)
        small_fired = _SMALL_M_CALLS["n"] > before
        ref_out = reference(x)
        if not torch.isfinite(ref_out.float()).all():
            failures.append(f"{shape_label} M={m} reference output non-finite")
        r_ref = rel(y_eager, ref_out)
        route = "small-M" if small_fired else "generic"
        print(
            f"{shape_label} M={m} ({route}, dispatched={small_fired}): "
            f"eager vs reference rel={r_ref:.2e}"
        )
        if launch is not None and m <= 16 and not small_fired:
            failures.append(
                f"{shape_label} M={m}: small-M launch bound but NOT dispatched"
            )
        if m > 16 and small_fired:
            failures.append(f"{shape_label} M={m}: small-M dispatched beyond _MAX_ROWS")
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
