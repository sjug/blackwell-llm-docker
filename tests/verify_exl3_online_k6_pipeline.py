"""Online K6 encoder/ExLlama execution-contract gate (in-image, GPU).

Exercises the exact path vLLM's online EXL3 encoding uses, end to end:
`_load_exl3_online_quantizer()` -> `quantize_exl3` with K=6, MCG, the
meta-Hessian fallback (`q_fallback=True`), and output scaling, producing
packed `trellis`/`suh`/`svh` - then verifies:

  1. Every returned tensor and the proxy error are finite.
  2. The ExLlamaV3 extension's fused linear (ext.BC_LinearEXL3) with an
     identity input reproduces the encoder-returned quantized weights, and
     so does the reference path (ext.reconstruct + rotations + hgemm): the
     three independent representations of W_q agree. NOTE: this is the
     ExLlamaV3 execution contract, NOT the b12x serving route; on SM121
     the b12x capture-safe K6/MCG kernel is not admitted (it gates on
     capability (12, 0)) and serving falls back to the generic
     _b12x_trellis_linear route. That route's CUDA-graph capture behavior
     is gated at build time by verify_b12x_trellis_graph_capture.py on the
     GLM-5.2 TP4 shared-expert shapes; full-model serving qualification
     remains for the GLM-EXL3-on-II window. This gate asserts the dispatch fact so the
     fallback is documented, not discovered.
  3. Fused linear matches reconstruct+GEMM on a random prefill-shaped
     batch (M=32).
  4. A second invocation through the vLLM online cache is a HIT (the
     quantize callable is not re-entered) and returns byte-identical
     tensors.

This gate replaces the removed aarch64 fail-closed guard: online K6 is
allowed because this path is proven, not prohibited because a harness bug
was misread as an encoder defect (retraction 2026-08-15).
"""

import os
import sys
import tempfile

import torch

os.environ.setdefault("VLLM_EXL3_ONLINE_CACHE_MODE", "readwrite")
_tmp = tempfile.mkdtemp(prefix="exl3-online-gate-")
os.environ["VLLM_EXL3_ONLINE_CACHE_DIR"] = _tmp

from vllm.model_executor.layers.quantization.exl3 import (  # noqa: E402
    _load_exl3_online_quantizer,
    _load_online_encoding_with_retry,
)
from vllm.model_executor.layers.quantization.exl3_online_cache import (  # noqa: E402
    Exl3OnlineCacheKey,
)

DEV = torch.device("cuda:0")
CODEBOOK = "mcg"  # single source for quant_args, cache key, and decode flags
IN_FEATURES = 512
OUT_FEATURES = 1024
failures = []

quantize_exl3 = _load_exl3_online_quantizer()

torch.manual_seed(20260815)
weight = torch.randn(OUT_FEATURES, IN_FEATURES, device=DEV) * 0.02
source = weight.detach().t().float().contiguous()

calls = {"n": 0}


def encode():
    calls["n"] += 1
    quant_args = {
        "K": 6,
        "seed": 0,
        "devices": [DEV],
        "apply_out_scales": True,
        CODEBOOK: True,
    }
    weight_q, proxy_error, tensors = quantize_exl3(
        source,
        {
            "H": torch.zeros(
                IN_FEATURES, IN_FEATURES, dtype=torch.float32, device="meta"
            ),
            "first_key": "gate-online",
            "count": 0,
            "finalized": False,
            "num_total": 0,
            "inf_nan": torch.zeros(2, dtype=torch.long, device=DEV),
            "device": DEV,
        },
        quant_args,
        return_weight_q=True,
        verbose=False,
    )
    if not quant_args.get("q_fallback", False):
        failures.append("meta-Hessian fallback (q_fallback) did not engage")
    encode.weight_q = weight_q
    return {n: tensors[n] for n in ("trellis", "suh", "svh")}, float(proxy_error)


key = Exl3OnlineCacheKey(
    model_identity="gate-model",
    encoder_identity="gate-encoder",
    prefix="gate.layer",
    bits=6,
    seed=0,
    tp_world_size=1,
    tp_rank=0,
    input_size=IN_FEATURES,
    output_size=OUT_FEATURES,
)

result1 = _load_online_encoding_with_retry(key, device=DEV, quantize=encode)
assert calls["n"] == 1, f"first invocation ran encode {calls['n']} times"


def tensors_of(result):
    if hasattr(result, "tensors"):
        return dict(result.tensors)
    if isinstance(result, tuple):
        return dict(result[0])
    return dict(result)


t1 = tensors_of(result1)
weight_q = encode.weight_q.to(DEV).float()

# 1. Finiteness.
for name, t in t1.items():
    if not torch.isfinite(t.float()).all():
        failures.append(f"{name} contains non-finite values")
if not torch.isfinite(weight_q).all():
    failures.append("weight_q contains non-finite values")

# b12x dispatch fact: the capture-safe K6/MCG kernel must NOT be admitted
# on this device (it gates on capability (12, 0)); SM121 serving uses the
# generic route, graph-gated at build time by
# verify_b12x_trellis_graph_capture.py and fully qualified in the
# GLM-EXL3-on-II serving window.
from b12x.moe._shared.kernels.w4a16.kernel import _use_k6_small  # noqa: E402

cap = torch.cuda.get_device_capability(DEV)
admitted = _use_k6_small(
    explicit_launch_config=False,
    device=DEV,
    m=32,
    trellis_bits=6,
    trellis_codebook="mcg",
    trellis_pair_kind=None,
    compute_dtype=torch.float16,
    external_hadamard_128=None,
)
print(f"capability={cap} capture-safe K6/MCG admitted={admitted}")
if cap == (12, 1) and admitted:
    failures.append("capture-safe K6 kernel unexpectedly admitted on (12,1)")
if cap == (12, 0) and not admitted:
    failures.append("capture-safe K6 kernel not admitted on (12,0)")

# 2 + 3. Three-way W_q agreement and GEMM parity, mirroring
# verify_exl3_gemm_parity's fused/reference machinery on the REAL encoded
# tensors (rotations included). Codebook flags derived ONCE from the same
# configuration the encoder used (MCG).
sys.path.insert(0, "/opt/exllamav3")
import exllamav3_ext as ext  # noqa: E402

MCG_FLAG, MUL1_FLAG = CODEBOOK == "mcg", CODEBOOK == "mul1"

trellis = t1["trellis"].to(DEV).contiguous()
suh = t1["suh"].to(DEV).half().contiguous()
svh = t1["svh"].to(DEV).half().contiguous()

w_recon = torch.empty((IN_FEATURES, OUT_FEATURES), dtype=torch.half, device=DEV)
ext.reconstruct(w_recon, trellis, 6, MCG_FLAG, MUL1_FLAG)
if not torch.isfinite(w_recon.float()).all():
    failures.append("ext.reconstruct produced non-finite weights")

xh_temp = torch.empty((32, IN_FEATURES), dtype=torch.half, device=DEV)
lin = ext.BC_LinearEXL3(trellis, suh, svh, 6, None, MCG_FLAG, MUL1_FLAG, xh_temp)


def ref_path(x):
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    y = torch.empty((x.shape[0], OUT_FEATURES), dtype=torch.half, device=DEV)
    ext.hgemm(xh, w_recon, y)
    ext.had_r_128(y, y, None, svh, 1.0)
    return y


def rel(a, b):
    d = (a.float() - b.float()).norm().item()
    return d / max(b.float().norm().item(), 1e-6)


# Effective linear map via identity input: fused, reference, and the
# encoder-returned weight_q must agree (weight_q is the effective map the
# serving layer believes it loaded).
eye = torch.eye(IN_FEATURES, dtype=torch.half, device=DEV)
w_fused = torch.cat(
    [
        lin.run_alloc(eye[i : i + 32], OUT_FEATURES, False).view(32, OUT_FEATURES)
        for i in range(0, IN_FEATURES, 32)
    ],
    dim=0,
)
w_ref = torch.cat(
    [ref_path(eye[i : i + 32].contiguous()) for i in range(0, IN_FEATURES, 32)], dim=0
)
r_fr = rel(w_fused, w_ref)
r_fq = rel(w_fused, weight_q)
print(
    f"W_q agreement: fused-vs-reference rel={r_fr:.2e}, "
    f"fused-vs-encoder-returned rel={r_fq:.2e}"
)
if not torch.isfinite(w_fused.float()).all():
    failures.append("fused identity map non-finite")
if r_fr > 2e-2:
    failures.append(f"fused vs reconstruct+GEMM disagree (rel {r_fr:.3e})")
if r_fq > 2e-2:
    failures.append(f"fused vs encoder weight_q disagree (rel {r_fq:.3e})")

torch.manual_seed(7)
x = (torch.randn(32, IN_FEATURES, device=DEV) * 0.5).half()
r_g = rel(lin.run_alloc(x, OUT_FEATURES, False).view(32, OUT_FEATURES), ref_path(x))
print(f"GEMM parity on random M=32 batch: rel={r_g:.2e}")
if r_g > 2e-2:
    failures.append(f"fused GEMM vs reference disagree on random batch (rel {r_g:.3e})")

# 4. Second invocation must be a cache hit with byte-identical tensors.
result2 = _load_online_encoding_with_retry(key, device=DEV, quantize=encode)
if calls["n"] != 1:
    failures.append(f"second invocation re-encoded (encode ran {calls['n']}x)")
t2 = tensors_of(result2)
for name in ("trellis", "suh", "svh"):
    a = t1[name].cpu().contiguous().numpy().tobytes()
    b = t2[name].cpu().contiguous().numpy().tobytes()
    if a != b:
        failures.append(f"cache-hit tensor {name} is not byte-identical")
print(f"cache: second load hit={calls['n'] == 1}, byte-identity checked")

if failures:
    print("EXL3-ONLINE-K6-PIPELINE-FAIL:", "; ".join(failures))
    sys.exit(1)
print("EXL3-ONLINE-K6-PIPELINE-OK")
