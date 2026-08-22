"""Source-level matrix for the B12X trellis graph-scratch planner.

`_b12x_trellis_c_tmp_elements` may return the one-float no-scratch shortcut
ONLY when the device is architecture-capable of the capture-safe small-M
K6/MCG kernel (capability (12, 0)); capability is one admission term, the
safe static one at planner scope. On every other device the generic W4A16 kernel
serves rows <= 128 and requires real padded scratch; an undersized buffer
raises during CUDA-graph capture (SM121 finding, 2026-08-15).

Matrix: capability {(12,0), (12,1)} x rows {1, 4, 32, 128, 129, 256}. The
planner is codebook/K-agnostic by design: on (12,0) a non-K6 shape also
falls back to the generic kernel at runtime, which is upstream's
pre-existing exposure and is documented rather than fixed here; on any
non-(12,0) capability the planner is conservative for every K, so the
K6/non-K6 axis collapses into the capability axis at planner scope.
"""

import sys
from unittest import mock

import torch

from vllm.model_executor.layers.quantization import exl3 as exl3_mod

PAD_CAP = exl3_mod._B12X_TRELLIS_C_TMP_CAP
COLUMNS = 4096


def padded(rows: int) -> int:
    padded_rows = max(((rows + 47) // 48) * 48, ((rows + 63) // 64) * 64)
    return min(COLUMNS * padded_rows, PAD_CAP)


failures = []
for cap in ((12, 0), (12, 1)):
    exl3_mod._b12x_small_trellis_kernel_capable.cache_clear()
    with mock.patch.object(torch.cuda, "get_device_capability", return_value=cap):
        for rows in (1, 4, 32, 128, 129, 256):
            got = exl3_mod._b12x_trellis_c_tmp_elements(rows, COLUMNS)
            if rows <= 128 and cap == (12, 0):
                want = 1
            else:
                want = padded(rows)
            status = "ok" if got == want else "WRONG"
            print(f"cap={cap} rows={rows}: scratch={got} want={want} [{status}]")
            if got != want:
                failures.append(f"cap={cap} rows={rows}: {got} != {want}")
exl3_mod._b12x_small_trellis_kernel_capable.cache_clear()

if failures:
    print("B12X-SCRATCH-MATRIX-FAIL:", "; ".join(failures))
    sys.exit(1)
print("B12X-SCRATCH-MATRIX-OK")
