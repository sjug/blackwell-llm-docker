"""EXL3 codebook oracle gate (in-image).

Ground truth: the Trellis codebook generators are deterministic integer ops
plus a lop3 (imm 0x6a == C^(A&B)) and IEEE fp16 arithmetic, all exactly
emulable with numpy. Gates, per codebook (3INST, MCG, MUL1):

  1. ext.decode must match the oracle for ALL 65,536 16-bit indices
     (hard fail on any mismatch): this is the entire serving-side
     correctness contract for pre-quantized EXL3 checkpoints.
  2. The encoder (quantize_tiles) must be oracle-exact for EVERY codebook
     at K3..K8, including the production K6/MCG arm (hard fail on any
     mismatch). Each arm passes only its own codebook key, matching the
     value-based selection contract in the patched quantize.py.
  3. The encoder output must be self-consistent: reconstructed tile values
     must equal ext.decode() of the returned indices, bit for bit.
  4. Selecting both codebooks at once must raise ValueError.

HISTORY (2026-08-15): a prior revision of this gate passed BOTH codebook
keys with boolean values into quantize_tiles, whose selection semantics at
the pinned commit were key-PRESENCE. Every arm therefore encoded with one
codebook, producing "3INST/MCG encoder mismatches" that were reported as a
fork-wide encoder bug. That finding is RETRACTED: it was this harness's
bug. The combined ExLlamaV3 patch makes selection value-based and rejects
both-set; this gate now tests that contract.

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
    "exllamav3.modules.quant.exl3_lib.quantize"
).quantize_tiles

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


def bits_of(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


all_idx = np.arange(65536, dtype=np.uint32)
enc_all = (
    torch.from_numpy(all_idx.astype(np.int32).astype(np.uint16).view(np.int16))
    .view(256, 256)
    .contiguous()
    .to(DEV)
)

ARMS = [
    (False, False, "3INST", {}),
    (True, False, "MCG", {"mcg": True}),
    (False, True, "MUL1", {"mul1": True}),
]

failures = []
for mcg, mul1, label, extra_args in ARMS:
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
    print(
        f"{label}: decode-vs-oracle 65536/65536: "
        f"{'BIT-EXACT' if bad == 0 else f'{bad} MISMATCHES'}"
    )
    if bad:
        failures.append(f"{label} decode {bad} bit mismatches")

    for K in range(3, 9):
        torch.manual_seed(300 + K)
        tiles = torch.randn((64, 256), device=DEV)
        out_tile, out_idx = quantize_tiles(tiles, {"K": K, **extra_args})
        got_enc = np.ascontiguousarray(
            out_tile.cpu().numpy().reshape(-1), dtype=np.float32
        )
        idx = (
            out_idx.cpu()
            .numpy()
            .astype(np.int32)
            .astype(np.uint16)
            .astype(np.uint32)
            .reshape(-1)
        )
        if not np.isfinite(got_enc).all():
            failures.append(f"{label} K{K} encoder produced non-finite values")

        # Gate 2: encoder values vs the CPU oracle of its own indices.
        exp_enc = np.ascontiguousarray(oracle(idx, mcg, mul1), dtype=np.float32)
        bad_enc = int(np.count_nonzero(bits_of(got_enc) != bits_of(exp_enc)))
        if bad_enc:
            failures.append(f"{label} K{K} encoder {bad_enc} bit mismatches vs oracle")

        # Gate 3: encoder self-consistency vs ext.decode of its indices.
        idx_t = (
            torch.from_numpy(
                idx.astype(np.uint16).view(np.int16).reshape(64, 256).copy()
            )
            .contiguous()
            .to(DEV)
        )
        redec = torch.empty((64, 256), dtype=torch.float, device=DEV)
        ext.decode(idx_t, redec, mcg, mul1)
        got_redec = np.ascontiguousarray(
            redec.cpu().numpy().reshape(-1), dtype=np.float32
        )
        bad_rt = int(np.count_nonzero(bits_of(got_enc) != bits_of(got_redec)))
        if bad_rt:
            failures.append(
                f"{label} K{K} encoder/decode round-trip {bad_rt} mismatches"
            )
        marker = " [production arm]" if (label == "MCG" and K == 6) else ""
        print(
            f"{label} K{K}: encoder oracle-exact="
            f"{bad_enc == 0} round-trip-exact={bad_rt == 0}{marker}"
        )

# Gate 4: both codebooks selected must be rejected.
try:
    quantize_tiles(
        torch.randn((1, 256), device=DEV), {"K": 3, "mcg": True, "mul1": True}
    )
except ValueError:
    print("both-codebooks selection: correctly rejected")
else:
    failures.append("mcg+mul1 simultaneously enabled was not rejected")

if failures:
    print("EXL3-ORACLE-GATE-FAIL:", "; ".join(failures))
    sys.exit(1)
print("EXL3-ORACLE-GATE-OK")
