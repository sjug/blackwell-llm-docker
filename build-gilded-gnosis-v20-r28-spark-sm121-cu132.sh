#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# =============================================================================
# GG v20 r28 "spark" image for DGX Spark (GB10, SM121, aarch64).
#
# = the exact upstream gilded-gnosis-v20 r28 composition (2026-08-04, Docker
#   main @ d780c39; canonical bases GG 3003860 / SparkInfer 272a84bd /
#   LMCache 9cebd405 plus the r28 PR manifests, incl. vLLM #235 reasoning
#   contract, updated #217 tiered offload, #228/SI#117 shared-H mixed
#   Trellis) with the Spark overlay layered on top:
#   - FlashInfer 7ad08da (upstream still pins 801d57a, which PREDATES the
#     flashinfer-ai#3932 merge; stock r28 ships the broken SM12x NVFP4
#     arithmetic our gate measures at 0.61x. Drop when GG's pin advances.)
#   - SM121 vLLM overlay, 3 files inside the composite (locks authoritative):
#     CMakeLists arch 12.1, MTP-3D residual fallback, and the flashinfer
#     frozen-q_len guard in its PR#234-refined form (persistent decode
#     wrapper reuse requires decode_q_len == planned 1+K; decode_q_len is
#     derived zero-padding-aware via decode_q_len_from_indptr).
#   - SparkInfer and LMCache compositions are upstream r28 VERBATIM.
#   - Dockerfile.vllm-sparkinfer-cu132: the SM121 port (arch args, multiarch
#     paths, sbsa fix, per-stage SHELL, DeepGEMM hook, NCCL preloader fix,
#     tvm-ffi build pin, sha-fetch clones, SKIP_EXLLAMAV3 packaging).
#
# Composed trees (locks are authoritative):
#   vLLM       47d195012a9e5e4b42d79f9504189d04d0f7e5d1 (r28 e1e9426 + spark)
#   SparkInfer 200c1db7ef98ff8bbfd4f621555326e20f42282e (r28 verbatim)
#   LMCache    9a05c8818bae48d15b79c7e876418bb813c08cd0 (r28 verbatim)
#
# Run on the build node (dusty): rootless podman + NVIDIA CDI for the smoke.
# =============================================================================

if [[ "${CONTAINER_ENGINE:-podman}" != "podman" ]]; then
  echo "ERROR: this recipe validates with podman; CONTAINER_ENGINE=${CONTAINER_ENGINE} is unsupported" >&2
  exit 1
fi
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: SM121/GB10 image must be built natively on aarch64 (got $(uname -m))" >&2
  exit 1
fi

configure_composition() {
  local prefix="$1" composition_dir="$2"
  local composition_lock="${composition_dir}/integration.lock.json"
  local repo ref commit patch_sha256 tree
  repo="$(jq -er '.base.repository' "${composition_lock}")"
  ref="$(jq -er '.base.ref | sub("^refs/heads/"; "")' "${composition_lock}")"
  commit="$(jq -er '.base.commit' "${composition_lock}")"
  patch_sha256="$(jq -er '.result.patch_sha256' "${composition_lock}")"
  tree="$(jq -er '.result.tree' "${composition_lock}")"
  export "${prefix}_REPO=${repo}"
  export "${prefix}_REF=${ref}"
  export "${prefix}_COMMIT=${commit}"
  export "${prefix}_PATCH_FILE=${composition_dir#patches/}/integration.patch"
  export "${prefix}_PATCH_SHA256=${patch_sha256}"
  export "${prefix}_INTEGRATION_LOCK_FILE=${composition_lock}"
  export "${prefix}_INTEGRATION_TREE=${tree}"
  export "REQUIRE_CLEAN_${prefix}_COMPOSITION=1"
  export "VERIFY_${prefix}_BASE_HEAD=0"
}

release_dir="patches/releases/gilded-gnosis-v20-r28-spark"
configure_composition VLLM "${release_dir}/vllm"
configure_composition SPARKINFER "${release_dir}/sparkinfer"
configure_composition LMCACHE "${release_dir}/lmcache"
export LMCACHE_INTEGRATION_BASE_COMMIT="${LMCACHE_COMMIT}"

export DATE_TAG="${DATE_TAG:-20260804}"
export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20-r28-spark-sm121-vllm${VLLM_INTEGRATION_TREE:0:7}-si${SPARKINFER_INTEGRATION_TREE:0:7}-fi7ad08da-cu132-${DATE_TAG}}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v20.r28.spark.vllm${VLLM_INTEGRATION_TREE:0:7}.si${SPARKINFER_INTEGRATION_TREE:0:7}.fi7ad08da.sm121.cu132.${DATE_TAG}}"

export CONTAINER_ENGINE="${CONTAINER_ENGINE:-podman}"
export DOCKERFILE="${DOCKERFILE:-Dockerfile.vllm-sparkinfer-cu132}"

# SM121 / GB10 arch targets.
export TORCH_CUDA_ARCH_LIST_ARG="${TORCH_CUDA_ARCH_LIST_ARG:-12.1a}"
export CMAKE_CUDA_ARCHITECTURES_ARG="${CMAKE_CUDA_ARCHITECTURES_ARG:-121a}"
export FLASHINFER_CUDA_ARCH_LIST_ARG="${FLASHINFER_CUDA_ARCH_LIST_ARG:-12.1a}"
export PLAIN_CUDA_ARCH_LIST_ARG="${PLAIN_CUDA_ARCH_LIST_ARG:-12.1}"

# GB10: 20 cores, 121 GB unified memory shared with the GPU.
export MAX_JOBS="${MAX_JOBS:-12}"
export VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-12}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
export PIN_SOURCE_COMMITS="${PIN_SOURCE_COMMITS:-1}"

export SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-voipmonitor/vllm:spark-sm121-cu132-system-base-r13-20260730}"
export BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-voipmonitor/vllm:spark-sm121-cu132-build-base-r13-20260730}"
export BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-1}"
export PUSH_BASE_IMAGE="${PUSH_BASE_IMAGE:-0}"

export NCCL_REPO="${NCCL_REPO:-https://github.com/local-inference-lab/nccl-canonical.git}"
export NCCL_REF="${NCCL_REF:-canonical/cu132-nccl2304-amd-noxml}"
export NCCL_COMMIT="${NCCL_COMMIT:-dfab7c1ace32da250ba97757879429c341b7bcf9}"

# FlashInfer: sjug mirror of 801d57a + PR#3932 (unchanged since v20p1).
export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/sjug/flashinfer.git}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-7ad08da11eb5ba3fc92f576905dce3e2cec03313}"
export FLASHINFER_REF="${FLASHINFER_REF:-${FLASHINFER_COMMIT}}"
export FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-0}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-${DEEPGEMM_COMMIT}}"
export DEEPGEMM_PATCH_FILE="${DEEPGEMM_PATCH_FILE-deepgemm-sm121-mqa-logits-arch-number-20260708.patch}"
export DEEPGEMM_PATCH_SHA256="${DEEPGEMM_PATCH_SHA256:-c5282e8eb431d83d1b426df55c15424319ec89b20c69d72bed16534c09e71606}"
if [[ -n "${DEEPGEMM_PATCH_FILE}" ]]; then
  echo "${DEEPGEMM_PATCH_SHA256}  patches/${DEEPGEMM_PATCH_FILE}" | sha256sum -c -
fi

# EXL3/Trellis on aarch64: the 2026-08-04 spike compiled all 68 units for
# sm_121a behind pure x86 arch guards and the GEMM/reconstruct family is
# self-consistent, BUT the encoder and the standalone decode op disagree by
# whole codewords for the 3INST and MCG codebooks on SM121 (mul1 is
# bit-exact). Root cause is a cross-unit codebook divergence that the
# bounded spike scope excludes. FAIL CLOSED: skip the extension, label it,
# and reject the glm52-exl3 profile at startup (gate below).
export SKIP_EXLLAMAV3="${SKIP_EXLLAMAV3:-1}"
export EXLLAMAV3_REPO="${EXLLAMAV3_REPO:-https://github.com/brandonmmusic-max/exllamav3.git}"
export EXLLAMAV3_REF="${EXLLAMAV3_REF:-a1-retile-sm120}"
export EXLLAMAV3_COMMIT="${EXLLAMAV3_COMMIT:-704aefd743b390af4bd0fb429d1906f9b964c7d8}"

export SPARKINFER_VERSION="${SPARKINFER_VERSION:-1.0.1}"
export VLLM_PATCH_URL=

# Launchers: r28 helper suite (6a61804, includes the r26 TP4/DCP4 prefill
# policy); the composed vLLM tree provides the root-level DS4 helpers.
export LAUNCHER_REPO="${LAUNCHER_REPO:-https://github.com/local-inference-lab/blackwell-llm-docker.git}"
export LAUNCHER_REF="${LAUNCHER_REF:-6a61804c378cc8d5d666ab381c3d56b4f812b52f}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-6a61804c378cc8d5d666ab381c3d56b4f812b52f}"
export VLLM_REQUIRED_LAUNCHERS="${VLLM_REQUIRED_LAUNCHERS:-serve-ds4-flash.sh serve-ds4-flash-spark.sh serve-gilded-gnosis.sh serve-fathomless-firmament.sh serve-glm52-v16.sh serve-glm52-v18.sh serve-glm52-v19.sh serve-glm52-hybrid-v17.sh serve-glm52-hybrid-v18.sh serve-glm52-hybrid-v19.sh glm52-dcp-prefill-policy.sh glm52-pcie-runtime-env.sh glm52-pcie-calibration.py glm52-lmcache-wrapper.sh}"

export CUTLASS_REF="${CUTLASS_REF:-e6233cbac5d7c7a865c19c91cd684ceece19513c}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-e6233cbac5d7c7a865c19c91cd684ceece19513c}"
export CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.6.0}"
export TORCH_VERSION_PREFIX="${TORCH_VERSION_PREFIX:-2.12.0+cu132}"
export TOKENSPEED_MLA_VERSION="${TOKENSPEED_MLA_VERSION:-0.1.8}"
export TVM_FFI_VERSION="${TVM_FFI_VERSION:-0.1.10}"
export TRITON_KERNELS_REF=
export TRITON_KERNELS_COMMIT=

export XGRAMMAR_REPO="${XGRAMMAR_REPO:-https://github.com/mlc-ai/xgrammar.git}"
export XGRAMMAR_REF="${XGRAMMAR_REF:-v0.2.5}"
export XGRAMMAR_COMMIT="${XGRAMMAR_COMMIT:-2ea71da4ccb997a06928c9fb69b99f330da56697}"
export XGRAMMAR_VERSION="${XGRAMMAR_VERSION:-0.2.5}"
export XGRAMMAR_TRANSFORMERS5_COMPAT="${XGRAMMAR_TRANSFORMERS5_COMPAT:-1}"

# r28 InstantTensor: whole-region -> segmented -> runtime-pinned fallback.
export INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO:-https://github.com/scitix/InstantTensor.git}"
export INSTANTTENSOR_REF="${INSTANTTENSOR_REF:-49b4010afc1cae0441e71fe0b0bffc24fa05e932}"
export INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-49b4010afc1cae0441e71fe0b0bffc24fa05e932}"

export LMCACHE_BUILD_VERSION="${LMCACHE_BUILD_VERSION:-0.5.2+glm52dcp.4}"

export HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.10}"
export VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES:-nvtx==0.2.15 PyNvVideoCodec==2.0.4 nccl4py==0.3.1}"

# Launcher/test provenance preflight (baked from context, labeled with
# LAUNCHER_COMMIT; refuse to build on drift).
runtime_source_paths=(
  launchers
  tests/test-glm52-dcp-prefill-policy.sh
  tests/test-glm52-pcie-calibration-helper.sh
  tests/test-glm52-online-quant-policy.sh
  tests/test-glm52-worker-multiproc-policy.sh
  tests/test-release-manifest-fail-fast.sh
  tests/test-glm52-exl3-helper.sh
  tests/test-glm52-lmcache-helper.sh
  tests/test-glm52-pcie-calibration.py
)
if ! git diff --quiet "${LAUNCHER_COMMIT}" -- "${runtime_source_paths[@]}" || \
   [[ -n "$(git status --porcelain --untracked-files=all -- "${runtime_source_paths[@]}")" ]]; then
  printf 'Launcher/test sources do not match pinned commit %s\n' \
    "${LAUNCHER_COMMIT}" >&2
  exit 1
fi

# Helper/runtime-contract unit tests (host-side, no GPU).
helper_test_tmp="$(mktemp -d)"
trap 'rm -rf "${helper_test_tmp}"' EXIT
helper_env=(
  TMPDIR="${helper_test_tmp}"
  XDG_CACHE_HOME="${helper_test_tmp}/cache"
  LMCACHE_L2_PATH="${helper_test_tmp}/lmcache-l2"
)
env "${helper_env[@]}" ./tests/test-glm52-dcp-prefill-policy.sh
env "${helper_env[@]}" ./tests/test-glm52-pcie-calibration-helper.sh
env "${helper_env[@]}" ./tests/test-glm52-online-quant-policy.sh
env "${helper_env[@]}" ./tests/test-glm52-worker-multiproc-policy.sh
env "${helper_env[@]}" ./tests/test-release-manifest-fail-fast.sh
env "${helper_env[@]}" ./tests/test-glm52-exl3-helper.sh
env "${helper_env[@]}" ./tests/test-glm52-lmcache-helper.sh
if python3 -c 'import pytest' 2>/dev/null; then
  python3 -m pytest -q tests/test-glm52-pcie-calibration.py
else
  uv run --with pytest python -m pytest -q tests/test-glm52-pcie-calibration.py
fi

./build-vllm-sparkinfer-cu132.sh "$@"

# ---------------------------------------------------------------------------
# Post-build gates.
# ---------------------------------------------------------------------------

labels="$(podman image inspect "${IMAGE}" --format '{{json .Config.Labels}}')"
jq -e --arg value "${VLLM_COMMIT}" '."local-inference.vllm.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${SPARKINFER_COMMIT}" '."local-inference.sparkinfer.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${FLASHINFER_COMMIT}" '."local-inference.flashinfer.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${CUTLASS_DSL_VERSION}" '."local-inference.cutlass_dsl.version" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${LMCACHE_BUILD_VERSION}" '."local-inference.lmcache.version" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${VLLM_INTEGRATION_TREE}" '."local-inference.vllm.integration.tree" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${VLLM_PATCH_SHA256}" '."local-inference.vllm.patch_sha256" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${SPARKINFER_INTEGRATION_TREE}" '."local-inference.sparkinfer.integration.tree" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${SPARKINFER_PATCH_SHA256}" '."local-inference.sparkinfer.patch_sha256" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${DEEPGEMM_PATCH_SHA256}" '."local-inference.deepgemm.patch_sha256" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${SKIP_EXLLAMAV3}" '."local-inference.exllamav3.skipped" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${LAUNCHER_COMMIT}" '."local-inference.launcher.commit" == $value' <<<"${labels}" >/dev/null

# Launcher byte provenance: DS4 helpers from the composed tree (shas pinned
# at composition; unchanged from r24), everything else from the context.
image_file_sha() {
  podman run --rm --entrypoint /usr/bin/sha256sum "${IMAGE}" "$1" | awk '{print $1}'
}
declare -A ds4_helper_sha=(
  [serve-ds4-flash.sh]="c36dc018a22b3a32dafb974e8dd36b28593ba6384328ba65e0d294354576a52f"
  [serve-ds4-flash-spark.sh]="2c241d20c4092613ff8bbf06d09379c35f1df9e451984395024ae34567f21225"
)
for helper in ${VLLM_REQUIRED_LAUNCHERS}; do
  have="$(image_file_sha "/usr/local/bin/${helper}")"
  if [[ -n "${ds4_helper_sha[${helper}]:-}" ]]; then
    want="${ds4_helper_sha[${helper}]}"
    src_desc="composed vLLM tree ${VLLM_INTEGRATION_TREE:0:12}"
  else
    want="$(sha256sum "launchers/${helper}" | awk '{print $1}')"
    src_desc="context launchers/${helper}"
  fi
  [[ "${have}" == "${want}" ]] || {
    echo "ERROR: image ${helper} (${have}) differs from ${src_desc} (${want})" >&2
    exit 1
  }
done

# r26 DCP policy must be baked into the GLM helper suite.
podman run --rm --entrypoint /bin/grep "${IMAGE}" \
  -q 'owner_merge=0' /usr/local/bin/glm52-dcp-prefill-policy.sh

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then

# GPU stack smoke: SM121 identity, inherited r13-lineage fixes, our baked
# SM121/DS4 fixes, the PR#3932 marker, and the NEW r28 source contracts
# (#235 reasoning prompts, #217 tiering spec, SI#117 runtime-dynamic
# mixed-Trellis ABI).
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable -i \
  --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
import inspect

import flashinfer
import torch

import sparkinfer  # noqa: F401
import importlib.metadata as md
from sparkinfer.attention.nsa_indexer import fused_indexer, tiled_topk
from vllm import envs
from vllm.config.speculative import SpeculativeConfig  # noqa: F401

assert torch.__version__.startswith("2.12."), torch.__version__
assert flashinfer.__version__.startswith("0.6.15"), flashinfer.__version__
assert md.version("sparkinfer") == "1.0.1"
assert md.version("xgrammar") == "0.2.5"
assert md.version("lmcache").startswith("0.5.2"), md.version("lmcache")
import xgrammar  # noqa: F401
import lmcache  # noqa: F401
assert hasattr(envs, "VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH")
assert hasattr(envs, "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS")

# Upstream Int64 direct-K fix (v19 IMA class) must remain present.
src = inspect.getsource(fused_indexer._score_tokens_direct_k)
assert "k_byte_off: Int64" in src, "direct-K Int64 offset fix missing"

# SI #85 / f06881a admissions.
topk_src = inspect.getsource(tiled_topk)
assert "output_page_table_row_stride" in topk_src, \
    "SI #85 runtime page-table stride missing"
assert "Int64(row_idx) * Int64(output_page_table_row_stride)" in topk_src, \
    "SI 64-bit page-table offset arithmetic missing"

# PR#3932 presence marker (arithmetic gate runs as a separate script).
import flashinfer.fused_moe.cute_dsl.b12x_moe as _fi_b12x_moe
assert "input_global_scale" in inspect.getsource(_fi_b12x_moe), \
    "PR#3932 quantfix content missing from FlashInfer"

# Baked vLLM fixes: registered broadcast mHC op + 3D-aware MTP entry.
import vllm.model_executor.kernels.mhc.tilelang  # noqa: F401
assert hasattr(torch.ops.vllm, "mhc_pre_broadcast_tilelang"), \
    "mhc_pre_broadcast_tilelang custom op not registered"
import vllm.models.deepseek_v4.nvidia.model as _ds4_model
model_src = inspect.getsource(_ds4_model)
assert model_src.count("if x.dim() == 2:") >= 2, \
    "b12x residual-None branch lacks the 3D MTP fallback"

# vLLM #235: the official 0731 reasoning contract. The bug this fixes made
# low and high render identical prompts; require all three present AND
# pairwise distinct.
from vllm.tokenizers.deepseek_v4_encoding import REASONING_EFFORT_PROMPTS
for effort in ("low", "high", "max"):
    assert effort in REASONING_EFFORT_PROMPTS, f"missing reasoning effort {effort}"
prompts = [REASONING_EFFORT_PROMPTS[e] for e in ("low", "high", "max")]
assert len(set(prompts)) == 3, "reasoning effort prompts are not distinct (#235)"

# vLLM #217 (r27 head): tiering offload spec present.
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec  # noqa: F401

# SparkInfer #117: runtime-dynamic mixed-Trellis partitions (ABI 6, tier
# counts are launch data).
from sparkinfer.moe._shared.kernels.w4a16.mixed_trellis import (
    W4A16MixedTrellisKernel,
    make_mixed_trellis_buffers,  # noqa: F401
)
assert W4A16MixedTrellisKernel.ABI_VERSION == 6, \
    W4A16MixedTrellisKernel.ABI_VERSION

# v20p3.1 frozen-q_len guard, refined form: both functions must exist.
from vllm.v1.attention.backends.flashinfer import (
    decode_q_len_from_indptr,  # noqa: F401
    persistent_decode_wrapper_eligible,  # noqa: F401
)

props = torch.cuda.get_device_properties(torch.cuda.current_device())
assert (int(props.major), int(props.minor)) == (12, 1)
assert int(props.multi_processor_count) == 48
print("GPU stack smoke: PASS",
      f"torch={torch.__version__} flashinfer={flashinfer.__version__}")
PY

# PR#3932 arithmetic gate: B12X CuTe-DSL MoE vs reference (subnormal-scale
# discriminator; validated PASS fi7ad08da, FAIL fi801d57a).
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v "$(pwd)/tests:/build-tests:ro" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" \
  /build-tests/verify_fi_pr3932_b12x_moe.py

# Generic NVFP4 quantization bit-stability smoke.
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v "$(pwd)/tests:/build-tests:ro" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" \
  /build-tests/verify_fi_nvfp4_quant_vectors.py

# Frozen-q_len invariant gate: threshold(1+2K) vs planned shape(1+K)
# derivation, q_len=1+K -> persistent, {1..1+2K}\{1+K} -> dynamic,
# q_len>ceiling -> prefill classification.
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v "$(pwd)/tests:/build-tests:ro" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" \
  /build-tests/verify_flashinfer_decode_qlen_guard.py

fi

# DS4 helper wiring (DRY_RUN prints to stderr).
ds4_dry_run() {
  podman run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
    -e DRY_RUN=1 -e MODEL=/model "$@" "${IMAGE}" 2>&1
}

prod_cfg="$(ds4_dry_run \
  -e MODE=dspark -e BACKEND=b12x-a8 -e TP_SIZE=2 -e DSPARK_TOKENS=6)"
grep -q 'mode=dspark' <<<"${prod_cfg}"
grep -q 'backend=b12x-a8' <<<"${prod_cfg}"
grep -Eq 'method\W+dspark' <<<"${prod_cfg}"
grep -Eq 'num_speculative_tokens\W+6' <<<"${prod_cfg}"
grep -q -- '--moe-backend b12x' <<<"${prod_cfg}"

mtp2_cfg="$(ds4_dry_run -e MODE=mtp2 -e BACKEND=b12x-a8 -e TP_SIZE=2)"
grep -q 'mode=mtp2' <<<"${mtp2_cfg}"
grep -q 'backend=b12x-a8' <<<"${mtp2_cfg}"
grep -Eq 'method\W+mtp' <<<"${mtp2_cfg}"

cutlass_cfg="$(ds4_dry_run -e MODE=dspark -e BACKEND=lucifer-cutlass -e TP_SIZE=2 -e DSPARK_TOKENS=6)"
grep -q 'backend=lucifer-cutlass' <<<"${cutlass_cfg}"

a16_cfg="$(ds4_dry_run -e MODE=dspark -e BACKEND=b12x-a16 -e TP_SIZE=2 -e DSPARK_TOKENS=7)"
grep -q 'backend=b12x-a16' <<<"${a16_cfg}"

# EXL3 fail-closed gate (spike verdict 2026-08-04: DEFER). The overlay's
# _load_exl3_ext must reject with the /opt/exllamav3/SKIPPED reason (clear
# startup error, not a generic import failure), and the families we serve
# (DS4, NF3 GLM, Laguna dflash) must not import EXL3 transitively.
if [[ "${SKIP_EXLLAMAV3}" == "1" ]]; then
  podman run --rm --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
import importlib.util
import sys

# The extension must be absent and the SKIPPED marker present.
assert importlib.util.find_spec("exllamav3") is None, "exllamav3 unexpectedly present"
marker = open("/opt/exllamav3/SKIPPED").read()
assert "SM121" in marker and "deliberately not built" in marker, marker

# The overlay loader must fail closed with the recorded reason.
from vllm.model_executor.layers.quantization.exl3 import _load_exl3_ext
try:
    _load_exl3_ext()
except RuntimeError as exc:
    msg = str(exc)
    assert "unavailable in this image" in msg and "SM121" in msg, msg
else:
    raise AssertionError("_load_exl3_ext did not fail closed")

# The served families must not touch EXL3 at import time.
import vllm.models.deepseek_v4.nvidia.model  # noqa: F401  (DS4)
import vllm.model_executor.layers.quantization  # noqa: F401
loaded = [m for m in sys.modules if "exllamav3_ext" in m]
assert not loaded, f"EXL3 modules loaded transitively: {loaded}"
print("EXL3 fail-closed gate: PASS")
PY
fi

echo "r28-spark build + gates: PASS ${IMAGE}"
