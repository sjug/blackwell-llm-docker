"""Source-level matrix for the r17 B12X dense-Trellis scratch planner.

r17 contract (differs from r15): `_b12x_trellis_c_tmp_elements(rows,
in_features, out_features)` delegates to the b12x package. When the
package exposes `k6_mcg_small_m_scratch_elements` (every r17 b12x),
rows <= 16 return the kernel's REAL split-K scratch and rows >= 17
always get padded scratch; the legacy `rows <= 128 -> 1` shortcut fires
only when the query is ABSENT (older packages, dead in this
composition). The planner is capability-INDEPENDENT by design: the #221
CuTe kernel admits SM120 and SM121 alike and launch binding is
per-weight at prepare time.

Matrix: capabilities {(12,0), (12,1)} x rows {1, 4, 16, 17, 128, 129}
x {query-present (K6/MCG package), query-absent (non-small package)}.
Asserts: identical planner output on both capabilities; the query/padded
boundary at 16/17; the legacy branch only without the query; the REAL
composed b12x exposes the query AND rejects non-%128 shapes with
ValueError (the fail-loud corner).
"""

import sys
from unittest import mock

import torch

from vllm.model_executor.layers.quantization import exl3 as exl3_mod

CAP_LIMIT = exl3_mod._B12X_TRELLIS_C_TMP_CAP
IN_F, OUT_F = 6144, 1024  # GLM-5.2 TP4 merged gate/up shard
QUERY_SENTINEL = 424242


class _QueryApi:
    @staticmethod
    def k6_mcg_small_m_scratch_elements(size_k: int, size_n: int) -> int:
        assert (size_k, size_n) == (IN_F, OUT_F)
        return QUERY_SENTINEL


class _NoQueryApi:
    pass


def padded(rows: int) -> int:
    padded_rows = max(((rows + 47) // 48) * 48, ((rows + 63) // 64) * 64)
    return min(OUT_F * padded_rows, CAP_LIMIT)


failures = []
results: dict[tuple, list] = {}
for cap in ((12, 0), (12, 1)):
    for has_query, api in (("query", _QueryApi()), ("noquery", _NoQueryApi())):
        with (
            mock.patch.object(torch.cuda, "get_device_capability", return_value=cap),
            mock.patch.object(exl3_mod, "_load_b12x_trellis_linear", return_value=api),
        ):
            row_vals = []
            for rows in (1, 4, 16, 17, 128, 129):
                got = exl3_mod._b12x_trellis_c_tmp_elements(rows, IN_F, OUT_F)
                if has_query == "query":
                    want = QUERY_SENTINEL if rows <= 16 else padded(rows)
                else:
                    want = 1 if rows <= 128 else padded(rows)
                status = "ok" if got == want else "WRONG"
                print(
                    f"cap={cap} {has_query} rows={rows}: {got} want={want} [{status}]"
                )
                if got != want:
                    failures.append(f"cap={cap} {has_query} rows={rows}: {got}!={want}")
                row_vals.append(got)
            results[(cap, has_query)] = row_vals

for key_pack in ("query", "noquery"):
    if results[((12, 0), key_pack)] != results[((12, 1), key_pack)]:
        failures.append(f"planner not capability-independent for {key_pack}")
print("capability-independence: checked (12,0) == (12,1) for both packages")

# The REAL composed b12x: query must exist (legacy branch dead here) and
# must reject non-%128 shapes loudly.
from b12x.gemm.trellis_linear import api as real_api  # noqa: E402

if not hasattr(real_api, "k6_mcg_small_m_scratch_elements"):
    failures.append("composed b12x lacks k6_mcg_small_m_scratch_elements")
else:
    real = real_api.k6_mcg_small_m_scratch_elements(IN_F, OUT_F)
    print(f"real query({IN_F},{OUT_F}) = {real} elements")
    if real <= 0:
        failures.append("real scratch query returned a non-positive size")
    try:
        real_api.k6_mcg_small_m_scratch_elements(IN_F + 8, OUT_F)
    except ValueError:
        print("non-%128 shape: correctly rejected with ValueError")
    else:
        failures.append("non-%128 shape did not raise ValueError")

if failures:
    print("B12X-SCRATCH-MATRIX-FAIL:", "; ".join(failures))
    sys.exit(1)
print("B12X-SCRATCH-MATRIX-OK")
