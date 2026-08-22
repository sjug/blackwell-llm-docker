"""Generic FlashInfer NVFP4 quantization bit-stability smoke.

Freezes exact packed e2m1 codes, e4m3 scale-factor bytes (subnormal rows
included), and decode digests for the host ``fp4_quantize`` /
``e2m1_and_ufp8sf_scale_to_float`` APIs, generated on the validated v20p1
image (GB10/SM121, 2026-07-30). Catches silent changes to the generic
quantization surface across FlashInfer pin bumps.

NOTE: this does NOT discriminate PR#3932 — the source file behind these
APIs is blob-identical at 801d57a and 7ad08da. The PR#3932 arithmetic gate
is ``verify_fi_pr3932_b12x_moe.py`` (B12X CuTe-DSL MoE vs reference).

Run inside the built image on a GPU node:
    podman run --rm --device nvidia.com/gpu=all \
      -v <repo>/tests:/build-tests:ro --entrypoint python <image> \
      /build-tests/verify_fi_nvfp4_quant_vectors.py
"""

import hashlib

import torch
from flashinfer.quantization import e2m1_and_ufp8sf_scale_to_float, fp4_quantize

EXPECTED = {
    "codes_sha256": "231c12e136bd22f6f5353db24db919fb9265ddb4a3a55f746761af7519e5b255",
    "sf_sha256": "f95e6c81d4e01123dec0da74fa3958228ae979b9157b088e801d08474bddc447",
    "codes_head": [111, 154, 227, 238, 34, 182, 63, 146,
                   214, 114, 91, 220, 175, 253, 147, 166],
    "sf_head": [88, 90, 88, 87, 103, 106, 107, 105,
                68, 69, 74, 68, 117, 126, 122, 123],
    "dequant_sha256": "2a4e3a10ac47835ed73b7badcfe5ff9ccbbdf3e88357a32b85a4e0783261f2a6",
    # NVFP4 block-truncation bound on the x16-scaled row; semantic sanity,
    # not a bit assertion.
    "max_abs_err_bound": 5.0,
}

torch.manual_seed(3932)
dev = "cuda"

# 8 rows x 64 cols; rows 4-7 scaled down hard so per-16-block scale factors
# land in (or near) the e4m3 subnormal range the decode fix addresses.
x = torch.randn(8, 64, dtype=torch.bfloat16, device=dev)
row_scale = torch.tensor(
    [1.0, 4.0, 0.25, 16.0, 2**-9, 2**-11, 2**-13, 2**-14],
    dtype=torch.bfloat16,
    device=dev,
).unsqueeze(1)
x = (x * row_scale).contiguous()

global_scale = torch.tensor([448.0 * 6.0 / x.float().abs().max()], device=dev)

codes, sf = fp4_quantize(
    x, global_scale, sf_vec_size=16, is_sf_swizzled_layout=False
)
torch.cuda.synchronize()

codes_sha = hashlib.sha256(codes.cpu().numpy().tobytes()).hexdigest()
sf_sha = hashlib.sha256(sf.cpu().numpy().tobytes()).hexdigest()
codes_head = codes.cpu().flatten()[:16].tolist()
sf_head = sf.cpu().flatten()[:16].tolist()

assert codes_head == EXPECTED["codes_head"], (
    f"packed e2m1 code bytes diverge:\n  got      {codes_head}\n"
    f"  expected {EXPECTED['codes_head']}"
)
assert sf_head == EXPECTED["sf_head"], (
    f"e4m3 scale-factor bytes diverge (subnormal rows included):\n"
    f"  got      {sf_head}\n  expected {EXPECTED['sf_head']}"
)
assert codes_sha == EXPECTED["codes_sha256"], f"codes digest {codes_sha}"
assert sf_sha == EXPECTED["sf_sha256"], f"scale digest {sf_sha}"

deq = e2m1_and_ufp8sf_scale_to_float(
    codes.cpu(),
    sf.cpu().reshape(-1),
    global_scale.reciprocal().cpu(),
    sf_vec_size=16,
    is_sf_swizzled_layout=False,
)
deq_sha = hashlib.sha256(deq.float().cpu().numpy().tobytes()).hexdigest()
assert deq_sha == EXPECTED["dequant_sha256"], f"dequant digest {deq_sha}"

max_err = (deq.float().cuda() - x.float()).abs().max().item()
assert max_err < EXPECTED["max_abs_err_bound"], (
    f"dequant reconstruction error {max_err} exceeds "
    f"{EXPECTED['max_abs_err_bound']} — decode arithmetic regressed"
)

print(f"FI NVFP4 quant vector smoke: PASS (max_abs_err={max_err:.6f})")
