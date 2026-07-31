"""FlashInfer PR#3932 arithmetic gate: B12X CuTe-DSL MoE subnormal scales.

PR#3932 (fi7ad08da = 801d57a + 4 commits) fixes the b12x MoE quant path in
``flashinfer/cute_dsl/fp4_common.py`` / ``moe_w4a16_fp4_helpers.py``: the
e4m3 scale decode applied the normal-number formula to SUBNORMAL scale
bytes (up to 4.5x off; hardware ``cvt.rn.f16x2.e4m3x2`` decodes them
exactly), and the precise-path quant multiplier was reciprocal-inverted.

This gate runs the real fused kernel (which quantizes its bf16 activations
internally) with input magnitudes chosen so the per-16-block activation
scale bytes land in the subnormal e4m3 range, then compares row norms
against the upstream reference implementation. On the unfixed 801d57a
pin the kernel undershoots the reference everywhere (~0.61x at unit
scale) and collapses to ~0.24x once activation scale bytes go subnormal;
on 7ad08da both arms track the reference (~1.00). Validated empirically
on GB10 2026-07-30: PASS on v20p1 (fi7ad08da), FAIL on v20p0
(fi801d57a) in both arms. See the attribution note above the arms for
what the split does and does not prove about the PR's individual fixes.

Helpers come from the vendored upstream test module (verbatim
``tests/moe/test_b12x_fused_moe.py`` @ fi7ad08da), sha256-pinned below.

Run inside the built image on a GB10:
    podman run --rm --device nvidia.com/gpu=all \
      -v <repo>/tests:/build-tests:ro --entrypoint python <image> \
      /build-tests/verify_fi_pr3932_b12x_moe.py
"""

import hashlib
import importlib.util
import pathlib
import sys

import torch

VENDORED = pathlib.Path(__file__).parent / "upstream" / "test_b12x_fused_moe_fi7ad08da.py"
VENDORED_SHA256 = "33083f7ff8672b30d190d51e1f315c21437f7f756aea6e419da0e28bf7f32aaa"

actual = hashlib.sha256(VENDORED.read_bytes()).hexdigest()
assert actual == VENDORED_SHA256, (
    f"vendored upstream test module drifted: {actual} != {VENDORED_SHA256}"
)

# The vendored module imports pytest for decorators only; provide a shim on
# images that don't ship pytest so the helpers stay importable.
try:
    import pytest  # noqa: F401
except ModuleNotFoundError:
    import types

    def _identity_marker(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(obj):
            return obj

        return deco

    class _MarkShim:
        def __getattr__(self, _name):
            return _identity_marker

    _shim = types.ModuleType("pytest")
    _shim.mark = _MarkShim()
    _shim.fixture = _identity_marker
    sys.modules["pytest"] = _shim

spec = importlib.util.spec_from_file_location("upstream_b12x_moe_tests", VENDORED)
assert spec is not None and spec.loader is not None
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)

from flashinfer.fused_moe.cute_dsl.b12x_moe import b12x_fused_moe  # noqa: E402

NUM_TOKENS = 128
HIDDEN = 256
INTERMEDIATE = 512
NUM_EXPERTS = 256
TOP_K = 1


def run_case(label, x_scale, ratio_lo, ratio_hi):
    tensors = T.create_moe_tensors(
        num_tokens=NUM_TOKENS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        top_k=TOP_K,
        seed=3932,
    )
    x = (tensors["x_bf16"] * x_scale).contiguous()

    # Mirror upstream test_numerical_accuracy's calling convention exactly.
    out = b12x_fused_moe(
        x=x,
        w1_weight=tensors["w1_weight"],
        w1_weight_sf=tensors["w1_weight_sf"],
        w1_alpha=tensors["w1_alpha"],
        fc2_input_scale=tensors["fc2_input_scale"],
        w2_weight=tensors["w2_weight"],
        w2_weight_sf=tensors["w2_weight_sf"],
        w2_alpha=tensors["w2_alpha"],
        token_selected_experts=tensors["token_selected_experts"],
        token_final_scales=tensors["token_final_scales"],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        num_local_experts=NUM_EXPERTS,
    )
    torch.cuda.synchronize()
    assert not torch.isnan(out).any(), f"{label}: NaN in kernel output"
    assert not torch.isinf(out).any(), f"{label}: Inf in kernel output"

    ref = T.compute_reference_moe_fp4(
        hidden_states=x.float().cuda(),
        gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
        gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
        token_selected_experts=tensors["token_selected_experts"],
        token_final_scales=tensors["token_final_scales"],
        num_tokens=NUM_TOKENS,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        fc2_input_scale=tensors["fc2_input_scale"],
    )

    out_norm = out.float().norm(dim=1)
    ref_norm = ref.float().norm(dim=1)
    live = ref_norm > 1e-30
    assert live.any(), f"{label}: reference produced no live rows"
    assert (out_norm[live] > 0).all(), (
        f"{label}: kernel produced zero rows where reference is non-zero"
    )
    ratios = (out_norm[live] / ref_norm[live]).cpu()
    median = ratios.median().item()
    frac_in = ((ratios > ratio_lo) & (ratios < ratio_hi)).float().mean().item()
    print(
        f"{label}: median row-norm ratio {median:.4f}, "
        f"{frac_in * 100:.1f}% of rows within [{ratio_lo}, {ratio_hi}]"
    )
    assert ratio_lo < median < ratio_hi, (
        f"{label}: median kernel/reference ratio {median:.4f} outside "
        f"[{ratio_lo}, {ratio_hi}] — PR#3932 arithmetic regressed"
    )
    assert frac_in > 0.9, (
        f"{label}: only {frac_in * 100:.1f}% of rows within ratio bounds"
    )


# Both arms discriminate the PIN (measured on GB10, identical
# vLLM/SparkInfer, only the FlashInfer commit differing):
#   arm                fi7ad08da (fixed)   fi801d57a (unfixed)
#   normal-scales      median 1.0008       median 0.6082
#   subnormal-scales   median 0.9942       median 0.2370
#                      (27% of activation scale bytes subnormal in arm 2)
# Attribution note: the public API dispatches the fast-math kernel path, so
# the arms separate by activation-scale regime, not by the PR's two
# individual fixes — the subnormal-decode fix dominates arm 2's extra
# collapse, but arm 1's deficit is not proof of the precise-path
# (fast_math=False) multiplier branch specifically. The gate's claim is
# empirical pin discrimination, which both arms deliver.
run_case("gate/normal-scales", x_scale=1.0, ratio_lo=0.8, ratio_hi=1.25)
run_case("gate/subnormal-scales", x_scale=2.0**-1, ratio_lo=0.8, ratio_hi=1.25)

print("FI PR#3932 B12X MoE arithmetic gate: PASS")
