"""EXL3 GEMM execution-parity gate (in-image).

Export presence and decode correctness do not execute GEMM. This gate runs
the fast path (BC_LinearEXL3: fused trellis-decode GEMM kernels) against the
reference path (ext.reconstruct -> ext.hgemm, the module's own
reconstruct_hgemm fallback) on identical trellis data, K3 and K4, at decode
(M=1) and prefill-shaped (M=32, 128) batch sizes. Two independent GPU decode
implementations agreeing on the same packed weights is the extension-level
serving contract. Spike-measured agreement on GB10 was ~9e-4 relative; the
gate allows 2e-2 for fp16 accumulation-order variance and hard-fails on any
non-finite output.

Encoder correctness is verified separately (corrected codebook oracle and
the online-K6 pipeline gate); here both compared paths decode the same
packed bits, so only GEMM execution is under test.
"""

import importlib
import sys
import types

import numpy as np
import torch  # noqa: F401

sys.path.insert(0, "/opt/exllamav3")
import exllamav3_ext  # noqa: E402

PKG_ROOT = "/opt/exllamav3-python/exllamav3"
for name, path in [
    ("exllamav3", PKG_ROOT),
    ("exllamav3.modules", f"{PKG_ROOT}/modules"),
    ("exllamav3.modules.quant", f"{PKG_ROOT}/modules/quant"),
    ("exllamav3.modules.quant.exl3_lib", f"{PKG_ROOT}/modules/quant/exl3_lib"),
]:
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__package__ = name
    sys.modules[name] = mod
ext_shim = types.ModuleType("exllamav3.ext")
ext_shim.exllamav3_ext = exllamav3_ext
sys.modules["exllamav3.ext"] = ext_shim

ext = exllamav3_ext
_quant = importlib.import_module("exllamav3.modules.quant.exl3_lib.quantize")
pack_trellis = _quant.pack_trellis
quantize_tiles = _quant.quantize_tiles

DEV = "cuda:0"
IN_FEATURES = 512  # 32 row tiles of 16
OUT_FEATURES = 1024  # 64 col tiles of 16
R_TILES = IN_FEATURES // 16
C_TILES = OUT_FEATURES // 16
REL_ERR_BOUND = 2e-2

failures = []

for K in (3, 4):
    torch.manual_seed(K)
    tiles = torch.randn((R_TILES * C_TILES, 256), device=DEV)
    _, out_idx = quantize_tiles(tiles, {"K": K, "mcg": False, "mul1": False})
    encoded = out_idx.view(R_TILES, C_TILES, 256).contiguous()
    trellis = pack_trellis(encoded, {"K": K}).contiguous()

    w = torch.empty((IN_FEATURES, OUT_FEATURES), dtype=torch.half, device=DEV)
    ext.reconstruct(w, trellis, K, False, False)

    suh = torch.ones(IN_FEATURES, dtype=torch.half, device=DEV)
    svh = torch.ones(OUT_FEATURES, dtype=torch.half, device=DEV)
    xh_temp = torch.empty((1, IN_FEATURES), dtype=torch.half, device=DEV)
    bc = ext.BC_LinearEXL3(trellis, suh, svh, K, None, False, False, xh_temp)

    for m in (1, 32, 128):
        x = (torch.randn((m, IN_FEATURES), device=DEV) * 0.5).half()

        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, suh, None, 1.0)
        y_ref = torch.empty((m, OUT_FEATURES), dtype=torch.half, device=DEV)
        ext.hgemm(xh, w, y_ref)
        ext.had_r_128(y_ref, y_ref, None, svh, 1.0)

        y_fast = bc.run_alloc(x, OUT_FEATURES, False).view(m, OUT_FEATURES)

        ref = y_ref.float().cpu().numpy()
        fast = y_fast.float().cpu().numpy()
        if not np.isfinite(ref).all():
            failures.append(f"K{K} M{m}: reference path non-finite")
        if not np.isfinite(fast).all():
            failures.append(f"K{K} M{m}: fast path non-finite")
        diff = float(np.linalg.norm(fast - ref))
        ref_norm = float(np.linalg.norm(ref))
        rel = diff / max(ref_norm, 1e-6)
        print(f"K{K} M{m}: fast-vs-reference rel err {rel:.6f} (bound {REL_ERR_BOUND})")
        if rel >= REL_ERR_BOUND:
            failures.append(f"K{K} M{m} rel err {rel:.6f}")

if failures:
    print("EXL3-GEMM-PARITY-FAIL:", "; ".join(failures))
    sys.exit(1)
print("EXL3-GEMM-PARITY-OK")
