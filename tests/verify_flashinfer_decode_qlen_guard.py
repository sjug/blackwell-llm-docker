#!/usr/bin/env python3
"""Gate: FlashInfer persistent-decode-wrapper frozen-q_len invariant (v20p3.1).

A persistent (CUDA-graph) FlashInfer decode wrapper may be reused only when
the runtime q_len_per_req equals the value frozen during wrapper planning
(1 + num_spec_tokens). Violations kill the engine:
  "q_len_per_req is part of the frozen cudagraph shape: this wrapper was
   planned with 6, got 8|5" -> EngineDeadError   (observed 2026-08-03, Laguna
   dflash K=5: prefill-tail+spec fusion q_len 8; spec truncation near
   max_tokens q_len 5).

The decode-classification ceiling (reorder_batch_threshold = 1 + 2K under
parallel drafting) DELIBERATELY admits reduced-depth lone steps as decodes so
spec-as-decode kernels can serve them; the planned shape (1 + K) is what the
captured wrapper can execute. These are different quantities and this gate
exists partly to stop anyone from conflating them.

Runs in-container (CPU only): /opt/venv/bin/python verify_flashinfer_decode_qlen_guard.py
"""

import sys
from types import SimpleNamespace

import torch

from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.attention.backends.flashinfer import persistent_decode_wrapper_eligible
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

K = 5
PLANNED = 1 + K            # frozen wrapper shape
CEILING = 1 + 2 * K        # decode-classification ceiling (parallel drafting)
MAX_BS = 96                # capacity term, ample for all cases below

failures = []


def check(name, cond):
    if cond:
        print(f"PASS {name}")
    else:
        failures.append(name)
        print(f"FAIL {name}")


def make_threshold_stub(parallel_drafting):
    spec = SimpleNamespace(num_speculative_tokens=K, parallel_drafting=parallel_drafting)
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=spec,
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        )
    )


# T1: real threshold derivation — 1 + 2K with parallel drafting (dflash),
#     1 + K without. K=5 => 11 vs 6. The two quantities MUST differ.
stub = make_threshold_stub(parallel_drafting=True)
AttentionMetadataBuilder._init_reorder_batch_threshold(
    stub, 1, supports_spec_as_decode=True
)
check("threshold(parallel_drafting=True) == 11", stub.reorder_batch_threshold == CEILING)

stub_np = make_threshold_stub(parallel_drafting=False)
AttentionMetadataBuilder._init_reorder_batch_threshold(
    stub_np, 1, supports_spec_as_decode=True
)
check("threshold(parallel_drafting=False) == 6", stub_np.reorder_batch_threshold == PLANNED)
check("ceiling != planned shape", CEILING != PLANNED)


def eligible(q_len, *, num_reqs=1, pure_decode=True, max_bs=MAX_BS, planned=PLANNED):
    return persistent_decode_wrapper_eligible(
        pure_decode=pure_decode,
        num_decode_tokens=q_len * num_reqs,
        decode_cudagraph_max_bs=max_bs,
        decode_q_len=q_len,
        planned_decode_q_len=planned,
    )


# T2: positive FULL-graph path preserved — planned shape selects the
#     persistent wrapper, single- and multi-request.
check("q_len=6 x1 -> persistent", eligible(PLANNED))
check("q_len=6 x8 -> persistent", eligible(PLANNED, num_reqs=8))

# T3: entire exposure surface — every decode-classifiable q_len except the
#     planned shape falls back to the dynamic wrapper.
for q in range(1, CEILING + 1):
    if q == PLANNED:
        continue
    check(f"q_len={q} -> dynamic fallback", not eligible(q))

# T3b: observed crash signatures stay covered explicitly.
check("q_len=5 (spec truncation signature) -> dynamic", not eligible(5))
check("q_len=8 (tail fusion signature) -> dynamic", not eligible(8))

# T4: q_len above the ceiling never reaches this decode fallback — the real
#     splitter classifies it as prefill.
meta = SimpleNamespace(
    max_query_len=CEILING + 1,
    num_reqs=1,
    num_actual_tokens=CEILING + 1,
    query_start_loc_cpu=torch.tensor([0, CEILING + 1], dtype=torch.int32),
)
nd, npf, ndt, npt = split_decodes_and_prefills(
    meta, decode_threshold=CEILING, require_uniform=True
)
check("q_len=12 -> prefill classification", (nd, npf, ndt, npt) == (0, 1, 0, CEILING + 1))

# T5: the other predicate terms still gate.
check("mixed batch (not pure decode) -> dynamic", not eligible(PLANNED, pure_decode=False))
check("over capacity -> dynamic", not eligible(PLANNED, num_reqs=32, max_bs=MAX_BS))
check("no-spec planned=1: q_len=1 -> persistent", eligible(1, planned=1))
check("no-spec planned=1: q_len=2 -> dynamic", not eligible(2, planned=1))

if failures:
    print(f"\n{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("\nALL CHECKS PASSED")
