"""EXL3 codebook oracle gate (in-image).

Ground truth: the Trellis codebook generators are deterministic integer ops
plus a lop3 (imm 0x6a == C^(A&B)) and IEEE fp16 arithmetic, all exactly
emulable with numpy. Gates, per codebook (3INST, MCG, MUL1):

  1. ext.decode must match the oracle for ALL 65,536 16-bit indices
     (hard fail on any mismatch): this is the entire serving-side
     correctness contract for pre-quantized EXL3 checkpoints.
  2. The encoder fast path (quantize_tiles) must match the oracle for MUL1
     (hard fail if it regresses). For 3INST/MCG the encoder is KNOWN
     inconsistent fork-wide (oracle-verified on SM120 and SM121,
     2026-08-08); its status is reported so a silent upstream fix or a
     regression is visible, but it does not fail the build. Online
     quantization is fail-closed in the composed vLLM tree.

Runs against the baked artifacts: /opt/exllamav3 (extension .so) and
/opt/exllamav3-python/exllamav3 (package source).
"""
import importlib
import sys
import types

import numpy as np
import torch  # noqa: F401  (loads libc10 before the extension)

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
quantize_tiles = importlib.import_module(
    "exllamav3.modules.quant.exl3_lib.quantize").quantize_tiles

DEV = "cuda:0"


def oracle(idx_u32: np.ndarray, mcg: bool, mul1: bool) -> np.ndarray:
    x = idx_u32.astype(np.uint32)
    if mul1:
        x = (x * np.uint32(0x83DCD12D)).astype(np.uint32)
        b = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
        s = (np.uint32(0x6400) + b).astype(np.uint32) & np.uint32(0xFFFF)
        h = s.astype(np.uint16).view(np.float16).astype(np.float32)
        k_inv = np.uint16(0x1EEE).view(np.float16).astype(np.float32)
        k_bias = np.uint16(0xC931).view(np.float16).astype(np.float32)
        return np.float16(h * k_inv + k_bias).astype(np.float32)
    if mcg:
        x = (x * np.uint32(0xCBAC1FED)).astype(np.uint32)
    else:
        x = (x * np.uint32(89226354)).astype(np.uint32)
        x = (x + np.uint32(64248484)).astype(np.uint32)
    y = np.uint32(0x3B603B60) ^ (x & np.uint32(0x8FFF8FFF))
    lo = (y & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    hi = (y >> np.uint32(16)).astype(np.uint16).view(np.float16)
    return (lo + hi).astype(np.float32)


all_idx = np.arange(65536, dtype=np.uint32)
enc_all = torch.from_numpy(all_idx.astype(np.int32).astype(np.uint16).view(np.int16)) \
    .view(256, 256).contiguous().to(DEV)

failures = []
for (mcg, mul1, label) in [(False, False, "3INST"), (True, False, "MCG"), (False, True, "MUL1")]:
    exp = np.ascontiguousarray(oracle(all_idx, mcg, mul1), dtype=np.float32)
    dec = torch.empty((256, 256), dtype=torch.float, device=DEV)
    ext.decode(enc_all, dec, mcg, mul1)
    got = np.ascontiguousarray(dec.cpu().numpy().reshape(-1), dtype=np.float32)
    # Bit-pattern comparison after explicit finiteness checks: catches NaNs
    # (abs(NaN) > tol is False) and any representation drift a tolerance
    # would forgive. "EXACT" here means identical float32 bits.
    if not np.isfinite(got).all():
        failures.append(f"{label} decode produced non-finite values")
    if not np.isfinite(exp).all():
        failures.append(f"{label} oracle produced non-finite values")
    bad = int(np.count_nonzero(got.view(np.uint32) != exp.view(np.uint32)))
    print(f"{label}: decode-vs-oracle 65536/65536: "
          f"{'BIT-EXACT' if bad == 0 else f'{bad} MISMATCHES'}")
    if bad:
        failures.append(f"{label} decode {bad} bit mismatches")

    torch.manual_seed(3)
    tiles = torch.randn((64, 256), device=DEV)
    out_tile, out_idx = quantize_tiles(tiles, {"K": 3, "mcg": mcg, "mul1": mul1})
    got_enc = np.ascontiguousarray(
        out_tile.cpu().numpy().reshape(-1), dtype=np.float32)
    idx = out_idx.cpu().numpy().astype(np.int32).astype(np.uint16).astype(np.uint32).reshape(-1)
    exp_enc = np.ascontiguousarray(oracle(idx, mcg, mul1), dtype=np.float32)
    # Non-finite encoder output always fails; the known 3INST/MCG value
    # inconsistency stays informational so an upstream fix cannot break the
    # build (a fix flips the message, not the gate).
    if not np.isfinite(got_enc).all():
        failures.append(f"{label} encoder produced non-finite values")
    bad_enc = int(np.count_nonzero(got_enc.view(np.uint32) != exp_enc.view(np.uint32)))
    state = ("BIT-EXACT" if bad_enc == 0
             else f"{bad_enc} mismatches (known fork-wide for 3INST/MCG)")
    print(f"{label}: encoder-vs-oracle: {state}")
    if mul1 and bad_enc:
        failures.append(f"MUL1 encoder regressed: {bad_enc} bit mismatches")
    if not mul1 and not mcg and bad_enc == 0:
        print("NOTE: 3INST encoder now oracle-exact; upstream may have fixed the fork bug")

if failures:
    print("EXL3-ORACLE-GATE-FAIL:", "; ".join(failures))
    sys.exit(1)
print("EXL3-ORACLE-GATE-OK")
