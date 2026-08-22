#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# =============================================================================
# GG v20 r34 "spark" image for DGX Spark (GB10, SM121, aarch64).
# Adapted from the r33-spark build: same platform (cu132/torch 2.12), same
# FlashInfer/EXL3/DeepGEMM pins; delta = upstream r34 manifests (vLLM +#258
# +#280 +#281 -#228; b12x base 7cecbb2c pkg 1.2.1 +#144) and the spark
# overlay re-authored against the #280 exl3.py rewrite. r34-spark serves the
# EXISTING 3.42bpw pre-quantized GLM profile; online K6 is OPEN and
# gated by the corrected encoder oracle + execution-contract gate
# (the r33-era fail-closed guard rationale was retracted 2026-08-15).
#
# = the exact upstream gilded-gnosis-v20 r34 composition (2026-08-10, Docker
#   main @ 426da51; canonical bases GG e2666d9a / b12x 9bbae678 / LMCache
#   9cebd405 plus the r34 PR manifests: our vLLM #234 q_len guard rides
#   UPSTREAM at head eaf24cc15, plus #245 one-shard indexer query-split,
#   #251 B12X graph channels + DSpark context KV, #252/#254 offload
#   final-store ordering/cleanup, #235 reasoning contract, b12x QSRT ABI +
#   capture-safe K6) with the Spark overlay layered on top:
#   - FlashInfer voipmonitor 1ac6942 (0.6.18 integration branch; CONTAINS
#     the flashinfer-ai#3932 quantfix content, verified by marker + the
#     arithmetic gate below - our sjug 7ad08da mirror pin is RETIRED).
#   - SM121 vLLM overlay, 4 files (locks authoritative): CMakeLists arch
#     12.1, the capability-gated B12X trellis graph-scratch planner, the
#     MTP-3D residual fallback, and the shm_broadcast queue file:
#     busy_loop_s 1 -> 0.002 (GB10 head-node P-core spin) PLUS the full
#     upstream #45224 queue-safety fix (10c75477b07c: bounded idle reader
#     waits + finally-release of read slots; 0.002 WITHOUT #45224 hung
#     the first GLM TP4/DCP2 boot, 2026-08-18). Gated behaviorally below. The r33-era fail-closed
#     online-EXL3 guard and VLLM_EXL3_ONLINE_UNSAFE_OVERRIDE are REMOVED
#     (2026-08-15): the "fork-wide encoder bug" rationale was a harness
#     key-presence artifact, retracted; the encoder-contract fix ships in
#     the combined ExLlamaV3 patch and online K6 is gated by the
#     corrected oracle + the execution-contract and graph-capture gates.
#   - EXL3 extension now BUILDS for sm_121a behind the pure x86 arch-guard
#     patch (EXLLAMAV3_PATCH_FILE); serving-side correctness is gated by
#     the exhaustive decode-vs-oracle test below.
#   - b12x and LMCache compositions are upstream r34 VERBATIM.
#
# Composed trees (locks are authoritative):
#   vLLM    17b78ef411db8ec26fc516a0dafca0b8b8868f2f (r34 4d006a43 + spark + #45224)
#   b12x    cd3ce190f0f1917402cdfd5773724267cc9a63f8 (r34 verbatim)
#   LMCache 9a05c8818bae48d15b79c7e876418bb813c08cd0 (r34 verbatim; same tree as r33)
#
# DATE_TAG is OUR build date (house convention since v20p3p1); the canonical
# r34 registry image is dated 20260810. Canonical linkage lives in the lock
# (spark_overlay.derived_from), not the tag date.
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

release_dir="patches/releases/gilded-gnosis-v20-r34-spark"
configure_composition VLLM "${release_dir}/vllm"
configure_composition B12X "${release_dir}/b12x"
configure_composition LMCACHE "${release_dir}/lmcache"
export LMCACHE_INTEGRATION_BASE_COMMIT="${LMCACHE_COMMIT}"

export DATE_TAG="${DATE_TAG:-20260818}"
export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20-r34-spark-sm121-vllm${VLLM_INTEGRATION_TREE:0:7}-b12x${B12X_INTEGRATION_TREE:0:7}-fi1ac6942-cu132-${DATE_TAG}}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v20.r34.spark.vllm${VLLM_INTEGRATION_TREE:0:7}.b12x${B12X_INTEGRATION_TREE:0:7}.fi1ac6942.sm121.cu132.${DATE_TAG}}"

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

# FlashInfer: upstream r33/r34 integration pin, unchanged (0.6.18 + PCIe IPC,
# #3932 NVFP4 quantfix). The sjug 7ad08da mirror is retired; the arithmetic
# gate below verifies the mainline lineage stays correct on SM12x.
export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/voipmonitor/flashinfer.git}"
export FLASHINFER_REF="${FLASHINFER_REF:-integration/main-pr4393-pcie-ipc-qualified-20260807}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-1ac6942776b383c6b03c7a5805a22e72a3e3349f}"
export FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-0}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-${DEEPGEMM_COMMIT}}"
export DEEPGEMM_PATCH_FILE="${DEEPGEMM_PATCH_FILE-deepgemm-sm121-mqa-logits-arch-number-20260708.patch}"
export DEEPGEMM_PATCH_SHA256="${DEEPGEMM_PATCH_SHA256:-c5282e8eb431d83d1b426df55c15424319ec89b20c69d72bed16534c09e71606}"
if [[ -n "${DEEPGEMM_PATCH_FILE}" ]]; then
  echo "${DEEPGEMM_PATCH_SHA256}  patches/${DEEPGEMM_PATCH_FILE}" | sha256sum -c -
fi

# EXL3/Trellis: BUILDS for sm_121a behind the arch-guard patch (x86 guards
# around AVX target macros/units; no kernel changes). Decode family is
# codebook-oracle-exact on GB10; the exhaustive oracle below is a gate.
export SKIP_EXLLAMAV3="${SKIP_EXLLAMAV3:-0}"
if [[ "${SKIP_EXLLAMAV3}" != "0" ]]; then
  echo "ERROR: EXL3 is a product requirement of the r34-spark release; SKIP_EXLLAMAV3 must be 0" >&2
  exit 1
fi
export EXLLAMAV3_REPO="${EXLLAMAV3_REPO:-https://github.com/brandonmmusic-max/exllamav3.git}"
export EXLLAMAV3_REF="${EXLLAMAV3_REF:-a1-retile-sm120}"
export EXLLAMAV3_COMMIT="${EXLLAMAV3_COMMIT:-704aefd743b390af4bd0fb429d1906f9b964c7d8}"
export EXLLAMAV3_PATCH_FILE="${EXLLAMAV3_PATCH_FILE-exllamav3-sm121-guards-and-encoder-contract-20260815.patch}"
export EXLLAMAV3_PATCH_SHA256="${EXLLAMAV3_PATCH_SHA256:-fd8d8c5e0a89cbac108282bb84bfab5650e108bf5f9960cdcb1ae31ce51a1c7b}"
echo "1ea33e89310b8eaf618aed4f310538d6b7f3efbc46e7b5d73b5902aa00678ad4  tests/upstream/test_b12x_w4a16_mixed_trellis_r34.py" | sha256sum -c -
echo "f2c2edda497f076d9349fa560213d141f0646a19e359423c3a3dabb70574ac9d  tests/upstream/test_shm_broadcast_45224.py" | sha256sum -c -
if [[ -n "${EXLLAMAV3_PATCH_FILE}" ]]; then
  echo "${EXLLAMAV3_PATCH_SHA256}  patches/${EXLLAMAV3_PATCH_FILE}" | sha256sum -c -
fi

export B12X_VERSION="${B12X_VERSION:-1.2.1}"
export VLLM_PATCH_URL=

# Launchers: r34 helper suite (7302862b); the composed vLLM tree provides the
# root-level DS4 helpers (serve-ds4-flash.sh changed in r33; shas below).
export LAUNCHER_REPO="${LAUNCHER_REPO:-https://github.com/local-inference-lab/blackwell-llm-docker.git}"
export LAUNCHER_REF="${LAUNCHER_REF:-7302862b8fcfdc7c06a411a61e1f0fb072258880}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-7302862b8fcfdc7c06a411a61e1f0fb072258880}"
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

export INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO:-https://github.com/voipmonitor/InstantTensor.git}"
export INSTANTTENSOR_REF="${INSTANTTENSOR_REF:-49b4010afc1cae0441e71fe0b0bffc24fa05e932}"
export INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-49b4010afc1cae0441e71fe0b0bffc24fa05e932}"

export LMCACHE_BUILD_VERSION="${LMCACHE_BUILD_VERSION:-0.5.2+glm52dcp.4}"

export HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.10}"
export VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES:-nvtx==0.2.15 PyNvVideoCodec==2.0.4 nccl4py==0.3.1}"

# Launcher/test provenance preflight.
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

./build-vllm-b12x-cu132.sh "$@"

# ---------------------------------------------------------------------------
# Post-build gates.
# ---------------------------------------------------------------------------

labels="$(podman image inspect "${IMAGE}" --format '{{json .Config.Labels}}')"
jq -e --arg value "${VLLM_COMMIT}" '."local-inference.vllm.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${B12X_COMMIT}" '."local-inference.sparkinfer.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${FLASHINFER_COMMIT}" '."local-inference.flashinfer.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${CUTLASS_DSL_VERSION}" '."local-inference.cutlass_dsl.version" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${LMCACHE_BUILD_VERSION}" '."local-inference.lmcache.version" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${VLLM_INTEGRATION_TREE}" '."local-inference.vllm.integration.tree" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${VLLM_PATCH_SHA256}" '."local-inference.vllm.patch_sha256" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${B12X_INTEGRATION_TREE}" '."local-inference.sparkinfer.integration.tree" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${B12X_PATCH_SHA256}" '."local-inference.sparkinfer.patch_sha256" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${DEEPGEMM_PATCH_SHA256}" '."local-inference.deepgemm.patch_sha256" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${EXLLAMAV3_PATCH_FILE}" '."local-inference.exllamav3.patch_file" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${EXLLAMAV3_PATCH_SHA256}" '."local-inference.exllamav3.patch_sha256" == $value' <<<"${labels}" >/dev/null
image_encoder_rev="$(podman run --rm --entrypoint /bin/sh "${IMAGE}" -c 'printf %s "$VLLM_EXL3_ENCODER_REVISION"')"
test "${image_encoder_rev}" = "${EXLLAMAV3_COMMIT}+p.${EXLLAMAV3_PATCH_SHA256}" || {
  echo "ERROR: VLLM_EXL3_ENCODER_REVISION is '${image_encoder_rev}', want commit+p.sha" >&2
  exit 1
}
jq -e '."local-inference.exllamav3.skipped" == "0"' <<<"${labels}" >/dev/null || {
  echo "ERROR: image label exllamav3.skipped is not 0" >&2
  exit 1
}
# EXL3 artifacts must exist regardless of GPU-check availability.
podman run --rm --entrypoint /bin/bash "${IMAGE}" -c \
  'test ! -f /opt/exllamav3/SKIPPED && ls /opt/exllamav3/exllamav3_ext*.so >/dev/null' || {
  echo "ERROR: EXL3 extension artifacts missing or SKIPPED marker present" >&2
  exit 1
}
jq -e --arg value "${LAUNCHER_COMMIT}" '."local-inference.launcher.commit" == $value' <<<"${labels}" >/dev/null

# Launcher byte provenance: DS4 helpers from the composed r33 tree
# (serve-ds4-flash.sh changed in r33), GLM suite from the pinned context.
image_file_sha() {
  podman run --rm --entrypoint /usr/bin/sha256sum "${IMAGE}" "$1" | awk '{print $1}'
}
declare -A ds4_helper_sha=(
  [serve-ds4-flash.sh]="1a2e112bdd8e5d72731c467196c1a5298473f572c2fadb451f29ceaf37476d67"
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

# r26 DCP policy must remain baked into the GLM helper suite.
podman run --rm --entrypoint /bin/grep "${IMAGE}" \
  -q 'owner_merge=0' /usr/local/bin/glm52-dcp-prefill-policy.sh

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then

# GPU stack smoke: SM121 identity, inherited fixes, r33 contracts.
# Queue-safety gates (#45224, behavioral - run BEFORE the GPU ladder).
# The vendored upstream regression tests exercise the composed source in
# the image: bounded indefinite reader waits, shm reread after idle-wait
# timeout without notify, and slot release when the reader raises.
podman run --rm \
  -v "$(pwd)/tests/upstream/test_shm_broadcast_45224.py:/build-tests/test_shm_broadcast_45224.py:ro" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" \
  -m pytest -q -p no:cacheprovider /build-tests/test_shm_broadcast_45224.py \
  -k "test_reader_timeout_caps_indefinite_waits or test_reader_rechecks_shm_after_idle_wait_timeout_without_notify or test_acquire_read_releases_slot_when_reader_raises"

# The effective SpinCondition constructor default must be exactly 0.002.
podman run --rm --entrypoint /opt/venv/bin/python "${IMAGE}" -c '
import inspect
from vllm.distributed.device_communicators.shm_broadcast import SpinCondition
d = inspect.signature(SpinCondition.__init__).parameters["busy_loop_s"].default
assert d == 0.002, f"busy_loop_s default {d} != 0.002"
print("SpinCondition busy_loop_s default:", d)'
printf 'shm_broadcast #45224 + 0.002 gates: PASS\n'

podman run --rm --device nvidia.com/gpu=all --security-opt label=disable -i \
  --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
import inspect

import flashinfer
import torch

import b12x  # noqa: F401
import importlib.metadata as md
from b12x.attention.nsa_indexer import fused_indexer, tiled_topk
from vllm import envs
from vllm.config.speculative import SpeculativeConfig  # noqa: F401

assert torch.__version__.startswith("2.12."), torch.__version__
assert flashinfer.__version__.startswith("0.6.18"), flashinfer.__version__
assert md.version("b12x") == "1.2.1"
assert md.version("xgrammar") == "0.2.5"
assert md.version("lmcache").startswith("0.5.2"), md.version("lmcache")
import xgrammar  # noqa: F401
import lmcache  # noqa: F401
assert hasattr(envs, "VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH")
assert hasattr(envs, "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS")

# Inherited kernel-safety admissions.
src = inspect.getsource(fused_indexer._score_tokens_direct_k)
assert "k_byte_off: Int64" in src, "direct-K Int64 offset fix missing"
topk_src = inspect.getsource(tiled_topk)
assert "output_page_table_row_stride" in topk_src, \
    "runtime page-table stride missing"

# FlashInfer #3932 content marker (arithmetic gate runs separately).
import flashinfer.fused_moe.cute_dsl.b12x_moe as _fi_b12x_moe
assert "input_global_scale" in inspect.getsource(_fi_b12x_moe), \
    "#3932 quantfix content missing from FlashInfer"

# Spark overlay: registered broadcast mHC op + 3D-aware MTP entry.
import vllm.model_executor.kernels.mhc.tilelang  # noqa: F401
assert hasattr(torch.ops.vllm, "mhc_pre_broadcast_tilelang"), \
    "mhc_pre_broadcast_tilelang custom op not registered"
import vllm.models.deepseek_v4.nvidia.model as _ds4_model
assert inspect.getsource(_ds4_model).count("if x.dim() == 2:") >= 2, \
    "b12x residual-None branch lacks the 3D MTP fallback"

# Upstream-composed #234 (our guard) and #235 (reasoning contract).
from vllm.v1.attention.backends.flashinfer import (
    decode_q_len_from_indptr,  # noqa: F401
    persistent_decode_wrapper_eligible,  # noqa: F401
)
from vllm.tokenizers.deepseek_v4_encoding import REASONING_EFFORT_PROMPTS
prompts = [REASONING_EFFORT_PROMPTS[e] for e in ("low", "high", "max")]
assert len(set(prompts)) == 3, "reasoning effort prompts are not distinct"

# #217 head + mixed-Trellis ABI.
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec  # noqa: F401
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    W4A16MixedTrellis3Kernel,
    W4A16MixedTrellisKernel,
)
assert W4A16MixedTrellisKernel.ABI_VERSION == 12, W4A16MixedTrellisKernel.ABI_VERSION
assert W4A16MixedTrellis3Kernel.ABI_VERSION == 4, W4A16MixedTrellis3Kernel.ABI_VERSION

# EXL3 extension present and loadable; online quant OPEN on aarch64 (the
# r33-era fail-closed guard was removed 2026-08-15: its "fork-wide encoder
# bug" rationale was a harness key-presence artifact, retracted). Online K6
# is gated by the corrected codebook oracle and the production-path
# pipeline gate; here we assert the mode selection works with no override
# machinery present.
import importlib.util
import sys
sys.path.insert(0, "/opt/exllamav3")
import exllamav3_ext  # noqa: F401
assert hasattr(exllamav3_ext, "exl3_gemm")
import os
assert not os.path.exists("/opt/exllamav3/SKIPPED")
import vllm.envs as _envs
assert "VLLM_EXL3_ONLINE_UNSAFE_OVERRIDE" not in dir(_envs), \
    "stale unsafe-override env survived the guard removal"
from vllm.model_executor.layers.quantization.exl3 import _online_trellis_bits
os.environ["VLLM_EXL3_ONLINE_TRELLIS_BITS"] = "6"
assert _online_trellis_bits() == 6, "online K6 mode selection must be open"
os.environ.pop("VLLM_EXL3_ONLINE_TRELLIS_BITS")
assert _online_trellis_bits() is None

props = torch.cuda.get_device_properties(torch.cuda.current_device())
assert (int(props.major), int(props.minor)) == (12, 1)
assert int(props.multi_processor_count) == 48
print("GPU stack smoke: PASS",
      f"torch={torch.__version__} flashinfer={flashinfer.__version__}")
PY

# FlashInfer #3932 arithmetic gate (now verifying the MAINLINE lineage).
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v "$(pwd)/tests:/build-tests:ro" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" \
  /build-tests/verify_fi_pr3932_b12x_moe.py

# NVFP4 quantization bit-stability smoke.
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v "$(pwd)/tests:/build-tests:ro" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" \
  /build-tests/verify_fi_nvfp4_quant_vectors.py

# Frozen-q_len invariant gate (guard now composed upstream via our #234).
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v "$(pwd)/tests:/build-tests:ro" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" \
  /build-tests/verify_flashinfer_decode_qlen_guard.py

# EXL3 codebook oracle gate: exhaustive decode-vs-oracle (all 65,536
# indices, all three codebooks, bit-pattern + finiteness) + MUL1 encoder
# regression guard.
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v "$(pwd)/tests:/build-tests:ro" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" \
  /build-tests/verify_exl3_codebook_oracle.py

# EXL3 GEMM execution parity: fused trellis-decode GEMM vs reconstruct+hgemm
# on identical trellis data (decode correctness does not execute GEMM).
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v "$(pwd)/tests:/build-tests:ro" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" \
  /build-tests/verify_exl3_gemm_parity.py

# Online K6 encoder/ExLlama execution-contract gate: quantize_exl3 K6/MCG
# with the meta-Hessian fallback and output scaling; finiteness, three-way
# W_q agreement (fused / reference / encoder-returned), GEMM parity, a
# byte-identical cache hit, and the b12x small-kernel dispatch fact. The
# b12x SERVING route is covered by the scratch-planner matrix and the
# graph-capture gate below.
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v "$(pwd)/tests:/build-tests:ro" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" \
  /build-tests/verify_exl3_online_k6_pipeline.py

# B12X graph-scratch planner matrix: the one-float shortcut only on (12,0)
# small-kernel admissibility; SM121 gets real padded scratch (source-level,
# capability mocked for both arms).
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v "$(pwd)/tests:/build-tests:ro" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" \
  /build-tests/verify_b12x_scratch_planner_matrix.py

# SM121 generic-route CUDA-graph gate: real vLLM _b12x_trellis_linear on
# both production R7 TP4 shared-expert shapes (6144->1024 gate/up,
# 512->6144 down), eager vs ExLlama reference, then graph capture + replay
# at M=4 and M=32 (the GLM capture-ladder band). This is the gate that
# detects the undersized-scratch capture failure.
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v "$(pwd)/tests:/build-tests:ro" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" \
  /build-tests/verify_b12x_trellis_graph_capture.py

# b12x r34 mixed K3/K4/K5 shared-H numeric and graph tests (vendored from the
# composed b12x tree): exercises the actual 3.42bpw MoE route on SM12x.
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  -v "$(pwd)/tests:/build-tests:ro" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" \
  -m pytest -q /build-tests/upstream/test_b12x_w4a16_mixed_trellis_r34.py

fi

# DS4 helper wiring (DRY_RUN prints to stderr).
ds4_dry_run() {
  podman run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
    -e DRY_RUN=1 -e MODEL=/model "$@" "${IMAGE}" 2>&1
}

prod_cfg="$(ds4_dry_run \
  -e MODE=dspark -e BACKEND=b12x-a8 -e TP_SIZE=2 -e DSPARK_TOKENS=5)"
grep -q 'mode=dspark' <<<"${prod_cfg}"
grep -q 'backend=b12x-a8' <<<"${prod_cfg}"
grep -Eq 'method\W+dspark' <<<"${prod_cfg}"
grep -Eq 'num_speculative_tokens\W+5' <<<"${prod_cfg}"
grep -q -- '--moe-backend b12x' <<<"${prod_cfg}"

cutlass_cfg="$(ds4_dry_run -e MODE=dspark -e BACKEND=lucifer-cutlass -e TP_SIZE=2 -e DSPARK_TOKENS=5)"
grep -q 'backend=lucifer-cutlass' <<<"${cutlass_cfg}"

# GLM EXL3 family wiring: the family must be selectable (extension ships),
# and the online-quant preset must remain a separate opt-in.
exl3_cfg="$(podman run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e DRY_RUN=1 -e MODEL=/model -e MODEL_FAMILY=glm52-exl3 "${IMAGE}" 2>&1)"
grep -Eq 'QUANTIZATION.*exl3|quantization.*exl3' <<<"${exl3_cfg}"

echo "r34-spark build + gates: PASS ${IMAGE}"
