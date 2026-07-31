#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:vllm-b12x-cu132}"
# Container engine: docker (default) or podman. Rootless podman works; all
# build stages are compile-only and do not need GPU access.
CONTAINER_ENGINE="${CONTAINER_ENGINE:-docker}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.vllm-b12x-cu132}"
# CUDA target archs. Defaults are the historical SM120 x86_64 values; for
# GB10 / DGX Spark (SM121) use 12.1a / 121a / 12.1a / 12.1 (FlashInfer
# refuses to run SM120 cubins on SM121 and only enables its sm121 AOT
# modules when 12.1a is in the arch list).
TORCH_CUDA_ARCH_LIST_ARG="${TORCH_CUDA_ARCH_LIST_ARG:-12.0a}"
CMAKE_CUDA_ARCHITECTURES_ARG="${CMAKE_CUDA_ARCHITECTURES_ARG:-120a}"
FLASHINFER_CUDA_ARCH_LIST_ARG="${FLASHINFER_CUDA_ARCH_LIST_ARG:-12.0f}"
PLAIN_CUDA_ARCH_LIST_ARG="${PLAIN_CUDA_ARCH_LIST_ARG:-12.0}"
SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-voipmonitor/vllm:vllm-b12x-cu132-system-base}"
BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-voipmonitor/vllm:vllm-b12x-cu132-build-base}"
BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-1}"
PUSH_BASE_IMAGE="${PUSH_BASE_IMAGE:-0}"
MAX_JOBS="${MAX_JOBS:-64}"
VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-64}"
NVCC_THREADS="${NVCC_THREADS:-1}"
VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
PIN_SOURCE_COMMITS="${PIN_SOURCE_COMMITS:-1}"

NCCL_REPO="${NCCL_REPO:-https://github.com/local-inference-lab/nccl-canonical.git}"
NCCL_REF="${NCCL_REF:-canonical/cu132-nccl2304-amd-noxml}"
FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
FLASHINFER_REF="${FLASHINFER_REF:-refs/pull/3395/head}"
FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-1}"
DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
DEEPGEMM_REF="${DEEPGEMM_REF:-refs/pull/324/head}"
DEEPGEMM_PATCH_FILE="${DEEPGEMM_PATCH_FILE:-}"
DEEPGEMM_PATCH_SHA256="${DEEPGEMM_PATCH_SHA256:-}"
SKIP_EXLLAMAV3="${SKIP_EXLLAMAV3:-0}"
EXLLAMAV3_REPO="${EXLLAMAV3_REPO:-https://github.com/brandonmmusic-max/exllamav3.git}"
EXLLAMAV3_REF="${EXLLAMAV3_REF:-a1-retile-sm120}"
B12X_REPO="${B12X_REPO:-https://github.com/lukealonso/b12x.git}"
B12X_REF="${B12X_REF:-refs/pull/11/head}"
B12X_PATCH_SHA256="${B12X_PATCH_SHA256:-}"
B12X_PATCH_FILE="${B12X_PATCH_FILE:-}"
REQUIRE_CLEAN_B12X_COMPOSITION="${REQUIRE_CLEAN_B12X_COMPOSITION:-0}"
VERIFY_B12X_BASE_HEAD="${VERIFY_B12X_BASE_HEAD:-1}"
B12X_INTEGRATION_LOCK_FILE="${B12X_INTEGRATION_LOCK_FILE:-}"
B12X_INTEGRATION_BASE_COMMIT="${B12X_INTEGRATION_BASE_COMMIT:-}"
B12X_INTEGRATION_TREE="${B12X_INTEGRATION_TREE:-}"
B12X_INTEGRATION_PRS="${B12X_INTEGRATION_PRS:-}"
B12X_INTEGRATION_LOCK_SHA256="${B12X_INTEGRATION_LOCK_SHA256:-}"
VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
VLLM_REF="${VLLM_REF:-dev/black-benediction}"
VLLM_PATCH_URL="${VLLM_PATCH_URL:-}"
VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-}"
VLLM_PATCH_FILE="${VLLM_PATCH_FILE:-}"
REQUIRE_CLEAN_VLLM_COMPOSITION="${REQUIRE_CLEAN_VLLM_COMPOSITION:-0}"
VERIFY_VLLM_BASE_HEAD="${VERIFY_VLLM_BASE_HEAD:-1}"
VLLM_INTEGRATION_LOCK_FILE="${VLLM_INTEGRATION_LOCK_FILE:-}"
VLLM_INTEGRATION_BASE_COMMIT="${VLLM_INTEGRATION_BASE_COMMIT:-}"
VLLM_INTEGRATION_TREE="${VLLM_INTEGRATION_TREE:-}"
VLLM_INTEGRATION_PRS="${VLLM_INTEGRATION_PRS:-}"
VLLM_INTEGRATION_LOCK_SHA256="${VLLM_INTEGRATION_LOCK_SHA256:-}"
LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
VLLM_REQUIRED_LAUNCHERS="${VLLM_REQUIRED_LAUNCHERS:-}"
CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
CUTLASS_REF="${CUTLASS_REF:-main}"
CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.5.2}"
TOKENSPEED_MLA_VERSION="${TOKENSPEED_MLA_VERSION:-0.1.2}"
TVM_FFI_VERSION="${TVM_FFI_VERSION:-0.1.9}"
VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev279+black.benediction.b12x.cu132}"
TRITON_KERNELS_REPO="${TRITON_KERNELS_REPO:-https://github.com/triton-lang/triton.git}"
TRITON_KERNELS_REF="${TRITON_KERNELS_REF:-}"
INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO:-https://github.com/scitix/InstantTensor.git}"
INSTANTTENSOR_REF="${INSTANTTENSOR_REF:-main}"
LMCACHE_REPO="${LMCACHE_REPO:-https://github.com/local-inference-lab/LMCache.git}"
LMCACHE_REF="${LMCACHE_REF:-release/v0.5.2-glm52-dcp-base}"
LMCACHE_PATCH_FILE="${LMCACHE_PATCH_FILE:-}"
LMCACHE_PATCH_SHA256="${LMCACHE_PATCH_SHA256:-}"
LMCACHE_BUILD_VERSION="${LMCACHE_BUILD_VERSION:-0.5.2+glm52dcp.3}"
REQUIRE_CLEAN_LMCACHE_COMPOSITION="${REQUIRE_CLEAN_LMCACHE_COMPOSITION:-0}"
VERIFY_LMCACHE_BASE_HEAD="${VERIFY_LMCACHE_BASE_HEAD:-1}"
LMCACHE_INTEGRATION_LOCK_FILE="${LMCACHE_INTEGRATION_LOCK_FILE:-}"
LMCACHE_INTEGRATION_BASE_COMMIT="${LMCACHE_INTEGRATION_BASE_COMMIT:-}"
LMCACHE_INTEGRATION_TREE="${LMCACHE_INTEGRATION_TREE:-}"
LMCACHE_INTEGRATION_PRS="${LMCACHE_INTEGRATION_PRS:-}"
LMCACHE_INTEGRATION_LOCK_SHA256="${LMCACHE_INTEGRATION_LOCK_SHA256:-}"
XGRAMMAR_REPO="${XGRAMMAR_REPO:-https://github.com/mlc-ai/xgrammar.git}"
XGRAMMAR_REF="${XGRAMMAR_REF-v0.2.5}"
XGRAMMAR_VERSION="${XGRAMMAR_VERSION-0.2.5}"
XGRAMMAR_TRANSFORMERS5_COMPAT="${XGRAMMAR_TRANSFORMERS5_COMPAT:-1}"
HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.4}"
VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES:-}"

resolve_ref() {
  local repo="$1"
  local ref="$2"
  local sha=""

  if [[ "${ref}" =~ ^[0-9a-f]{40}$ ]]; then
    printf '%s\n' "${ref}"
    return
  fi

  sha="$(git ls-remote "${repo}" "refs/heads/${ref}" | awk 'NR == 1 {print $1}')"
  if [[ -z "${sha}" ]]; then
    sha="$(git ls-remote "${repo}" "refs/tags/${ref}^{}" | awk 'NR == 1 {print $1}')"
  fi
  if [[ -z "${sha}" ]]; then
    sha="$(git ls-remote "${repo}" "${ref}" | awk 'NR == 1 {print $1}')"
  fi
  if [[ -z "${sha}" ]]; then
    echo "Unable to resolve ${repo} ${ref}" >&2
    exit 1
  fi
  printf '%s\n' "${sha}"
}

if [[ "${PIN_SOURCE_COMMITS}" == "1" ]]; then
  NCCL_COMMIT="${NCCL_COMMIT:-$(resolve_ref "${NCCL_REPO}" "${NCCL_REF}")}"
  FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-$(resolve_ref "${FLASHINFER_REPO}" "${FLASHINFER_REF}")}"
  DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-$(resolve_ref "${DEEPGEMM_REPO}" "${DEEPGEMM_REF}")}"
  EXLLAMAV3_COMMIT="${EXLLAMAV3_COMMIT:-$(resolve_ref "${EXLLAMAV3_REPO}" "${EXLLAMAV3_REF}")}"
  B12X_COMMIT="${B12X_COMMIT:-$(resolve_ref "${B12X_REPO}" "${B12X_REF}")}"
  VLLM_COMMIT="${VLLM_COMMIT:-$(resolve_ref "${VLLM_REPO}" "${VLLM_REF}")}"
  LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-$(resolve_ref "${LAUNCHER_REPO}" "${LAUNCHER_REF}")}"
  CUTLASS_COMMIT="${CUTLASS_COMMIT:-$(resolve_ref "${CUTLASS_REPO}" "${CUTLASS_REF}")}"
  if [[ -n "${TRITON_KERNELS_REF}" ]]; then
    TRITON_KERNELS_COMMIT="${TRITON_KERNELS_COMMIT:-$(resolve_ref "${TRITON_KERNELS_REPO}" "${TRITON_KERNELS_REF}")}"
  else
    TRITON_KERNELS_COMMIT="${TRITON_KERNELS_COMMIT:-}"
  fi
  if [[ -n "${INSTANTTENSOR_REF}" ]]; then
    INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-$(resolve_ref "${INSTANTTENSOR_REPO}" "${INSTANTTENSOR_REF}")}"
  else
    INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-}"
  fi
  LMCACHE_COMMIT="${LMCACHE_COMMIT:-$(resolve_ref "${LMCACHE_REPO}" "${LMCACHE_REF}")}"
  if [[ -n "${XGRAMMAR_REF}" ]]; then
    XGRAMMAR_COMMIT="${XGRAMMAR_COMMIT:-$(resolve_ref "${XGRAMMAR_REPO}" "${XGRAMMAR_REF}")}"
  else
    XGRAMMAR_COMMIT="${XGRAMMAR_COMMIT:-}"
  fi
else
  NCCL_COMMIT="${NCCL_COMMIT:-}"
  FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-}"
  DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-}"
  EXLLAMAV3_COMMIT="${EXLLAMAV3_COMMIT:-}"
  B12X_COMMIT="${B12X_COMMIT:-}"
  VLLM_COMMIT="${VLLM_COMMIT:-}"
  LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-}"
  CUTLASS_COMMIT="${CUTLASS_COMMIT:-}"
  TRITON_KERNELS_COMMIT="${TRITON_KERNELS_COMMIT:-}"
  INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-}"
  LMCACHE_COMMIT="${LMCACHE_COMMIT:-}"
  XGRAMMAR_COMMIT="${XGRAMMAR_COMMIT:-}"
fi

if [[ -n "${XGRAMMAR_REF}" && ( -z "${XGRAMMAR_COMMIT}" || -z "${XGRAMMAR_VERSION}" ) ]]; then
  echo "Pinned xgrammar requires XGRAMMAR_COMMIT and XGRAMMAR_VERSION" >&2
  exit 1
fi
if [[ "${XGRAMMAR_TRANSFORMERS5_COMPAT}" != "0" && "${XGRAMMAR_TRANSFORMERS5_COMPAT}" != "1" ]]; then
  echo "XGRAMMAR_TRANSFORMERS5_COMPAT must be 0 or 1" >&2
  exit 1
fi

if [[ "${REQUIRE_CLEAN_VLLM_COMPOSITION}" == "1" ]]; then
  command -v jq >/dev/null || {
    echo "jq is required for clean vLLM release composition" >&2
    exit 1
  }
  [[ "${VLLM_REF}" == "dev/gilded-gnosis" ]] || {
    echo "Clean GG composition requires VLLM_REF=dev/gilded-gnosis, got ${VLLM_REF}" >&2
    exit 1
  }
  [[ -n "${VLLM_COMMIT}" && -n "${VLLM_PATCH_FILE}" && -n "${VLLM_INTEGRATION_LOCK_FILE}" ]] || {
    echo "Clean GG composition requires a base commit, generated patch, and lockfile" >&2
    exit 1
  }
  [[ -f "${VLLM_INTEGRATION_LOCK_FILE}" ]] || {
    echo "VLLM integration lockfile does not exist: ${VLLM_INTEGRATION_LOCK_FILE}" >&2
    exit 1
  }

  lock_repo="$(jq -er '.base.repository' "${VLLM_INTEGRATION_LOCK_FILE}")"
  lock_ref="$(jq -er '.base.ref' "${VLLM_INTEGRATION_LOCK_FILE}")"
  lock_commit="$(jq -er '.base.commit' "${VLLM_INTEGRATION_LOCK_FILE}")"
  VLLM_INTEGRATION_TREE="$(jq -er '.result.tree' "${VLLM_INTEGRATION_LOCK_FILE}")"
  lock_patch_sha="$(jq -er '.result.patch_sha256' "${VLLM_INTEGRATION_LOCK_FILE}")"
  VLLM_INTEGRATION_PRS="$(jq -r '[.pull_requests[] | "\(.number)@\(.head)"] | join(",")' "${VLLM_INTEGRATION_LOCK_FILE}")"
  VLLM_INTEGRATION_LOCK_SHA256="$(sha256sum "${VLLM_INTEGRATION_LOCK_FILE}" | awk '{print $1}')"
  VLLM_INTEGRATION_BASE_COMMIT="${lock_commit}"

  [[ "${lock_repo}" == "${VLLM_REPO}" ]] || {
    echo "Integration lock repository mismatch: ${lock_repo} != ${VLLM_REPO}" >&2
    exit 1
  }
  [[ "${lock_ref}" == "refs/heads/${VLLM_REF}" ]] || {
    echo "Integration lock base ref mismatch: ${lock_ref} != refs/heads/${VLLM_REF}" >&2
    exit 1
  }
  [[ "${lock_commit}" == "${VLLM_COMMIT}" ]] || {
    echo "Integration lock base commit mismatch: ${lock_commit} != ${VLLM_COMMIT}" >&2
    exit 1
  }
  [[ "${VERIFY_VLLM_BASE_HEAD}" =~ ^[01]$ ]] || {
    echo "VERIFY_VLLM_BASE_HEAD must be 0 or 1" >&2
    exit 1
  }
  if [[ "${VERIFY_VLLM_BASE_HEAD}" == "1" ]]; then
    current_base_commit="$(resolve_ref "${VLLM_REPO}" "${VLLM_REF}")"
    [[ "${current_base_commit}" == "${VLLM_COMMIT}" ]] || {
      echo "GG advanced from ${VLLM_COMMIT} to ${current_base_commit}; rerun the release composer" >&2
      exit 1
    }
  fi
  if [[ -n "${VLLM_PATCH_SHA256}" && "${VLLM_PATCH_SHA256}" != "${lock_patch_sha}" ]]; then
    echo "Integration lock patch SHA mismatch: ${lock_patch_sha} != ${VLLM_PATCH_SHA256}" >&2
    exit 1
  fi
  VLLM_PATCH_SHA256="${lock_patch_sha}"
fi

if [[ "${REQUIRE_CLEAN_B12X_COMPOSITION}" == "1" ]]; then
  command -v jq >/dev/null || {
    echo "jq is required for clean B12X/SparkInfer release composition" >&2
    exit 1
  }
  [[ -n "${B12X_COMMIT}" && -n "${B12X_PATCH_FILE}" && -n "${B12X_INTEGRATION_LOCK_FILE}" ]] || {
    echo "Clean B12X/SparkInfer composition requires a base commit, generated patch, and lockfile" >&2
    exit 1
  }
  [[ -f "${B12X_INTEGRATION_LOCK_FILE}" ]] || {
    echo "B12X/SparkInfer integration lockfile does not exist: ${B12X_INTEGRATION_LOCK_FILE}" >&2
    exit 1
  }

  b12x_lock_repo="$(jq -er '.base.repository' "${B12X_INTEGRATION_LOCK_FILE}")"
  b12x_lock_ref="$(jq -er '.base.ref' "${B12X_INTEGRATION_LOCK_FILE}")"
  b12x_lock_commit="$(jq -er '.base.commit' "${B12X_INTEGRATION_LOCK_FILE}")"
  B12X_INTEGRATION_TREE="$(jq -er '.result.tree' "${B12X_INTEGRATION_LOCK_FILE}")"
  b12x_lock_patch_sha="$(jq -er '.result.patch_sha256' "${B12X_INTEGRATION_LOCK_FILE}")"
  B12X_INTEGRATION_PRS="$(jq -r '[.pull_requests[] | "\(.number)@\(.head)"] | join(",")' "${B12X_INTEGRATION_LOCK_FILE}")"
  B12X_INTEGRATION_LOCK_SHA256="$(sha256sum "${B12X_INTEGRATION_LOCK_FILE}" | awk '{print $1}')"
  B12X_INTEGRATION_BASE_COMMIT="${b12x_lock_commit}"

  [[ "${b12x_lock_repo}" == "${B12X_REPO}" ]] || {
    echo "B12X/SparkInfer integration lock repository mismatch: ${b12x_lock_repo} != ${B12X_REPO}" >&2
    exit 1
  }
  [[ "${b12x_lock_ref}" == "refs/heads/${B12X_REF}" ]] || {
    echo "B12X/SparkInfer integration lock base ref mismatch: ${b12x_lock_ref} != refs/heads/${B12X_REF}" >&2
    exit 1
  }
  [[ "${b12x_lock_commit}" == "${B12X_COMMIT}" ]] || {
    echo "B12X/SparkInfer integration lock base commit mismatch: ${b12x_lock_commit} != ${B12X_COMMIT}" >&2
    exit 1
  }
  [[ "${VERIFY_B12X_BASE_HEAD}" =~ ^[01]$ ]] || {
    echo "VERIFY_B12X_BASE_HEAD must be 0 or 1" >&2
    exit 1
  }
  if [[ "${VERIFY_B12X_BASE_HEAD}" == "1" ]]; then
    current_b12x_base_commit="$(resolve_ref "${B12X_REPO}" "${B12X_REF}")"
    [[ "${current_b12x_base_commit}" == "${B12X_COMMIT}" ]] || {
      echo "B12X/SparkInfer base advanced from ${B12X_COMMIT} to ${current_b12x_base_commit}; rerun the release composer" >&2
      exit 1
    }
  fi
  if [[ -n "${B12X_PATCH_SHA256}" && "${B12X_PATCH_SHA256}" != "${b12x_lock_patch_sha}" ]]; then
    echo "B12X/SparkInfer integration lock patch SHA mismatch: ${b12x_lock_patch_sha} != ${B12X_PATCH_SHA256}" >&2
    exit 1
  fi
  B12X_PATCH_SHA256="${b12x_lock_patch_sha}"
fi

if [[ "${REQUIRE_CLEAN_LMCACHE_COMPOSITION}" == "1" ]]; then
  command -v jq >/dev/null || {
    echo "jq is required for clean LMCache release composition" >&2
    exit 1
  }
  [[ -n "${LMCACHE_COMMIT}" && -n "${LMCACHE_PATCH_FILE}" && \
     -n "${LMCACHE_INTEGRATION_LOCK_FILE}" ]] || {
    echo "Clean LMCache composition requires a base commit, generated patch, and lockfile" >&2
    exit 1
  }
  [[ -f "${LMCACHE_INTEGRATION_LOCK_FILE}" ]] || {
    echo "LMCache integration lockfile does not exist: ${LMCACHE_INTEGRATION_LOCK_FILE}" >&2
    exit 1
  }

  lmcache_lock_repo="$(jq -er '.base.repository' "${LMCACHE_INTEGRATION_LOCK_FILE}")"
  lmcache_lock_ref="$(jq -er '.base.ref' "${LMCACHE_INTEGRATION_LOCK_FILE}")"
  lmcache_lock_commit="$(jq -er '.base.commit' "${LMCACHE_INTEGRATION_LOCK_FILE}")"
  LMCACHE_INTEGRATION_TREE="$(jq -er '.result.tree' "${LMCACHE_INTEGRATION_LOCK_FILE}")"
  lmcache_lock_patch_sha="$(jq -er '.result.patch_sha256' "${LMCACHE_INTEGRATION_LOCK_FILE}")"
  LMCACHE_INTEGRATION_PRS="$(jq -r '[.pull_requests[] | "\(.number)@\(.head)"] | join(",")' "${LMCACHE_INTEGRATION_LOCK_FILE}")"
  LMCACHE_INTEGRATION_LOCK_SHA256="$(sha256sum "${LMCACHE_INTEGRATION_LOCK_FILE}" | awk '{print $1}')"
  LMCACHE_INTEGRATION_BASE_COMMIT="${lmcache_lock_commit}"

  [[ "${lmcache_lock_repo}" == "${LMCACHE_REPO}" ]] || {
    echo "LMCache integration lock repository mismatch: ${lmcache_lock_repo} != ${LMCACHE_REPO}" >&2
    exit 1
  }
  [[ "${lmcache_lock_ref}" == "refs/heads/${LMCACHE_REF}" ]] || {
    echo "LMCache integration lock base ref mismatch: ${lmcache_lock_ref} != refs/heads/${LMCACHE_REF}" >&2
    exit 1
  }
  [[ "${lmcache_lock_commit}" == "${LMCACHE_COMMIT}" ]] || {
    echo "LMCache integration lock base commit mismatch: ${lmcache_lock_commit} != ${LMCACHE_COMMIT}" >&2
    exit 1
  }
  [[ "${VERIFY_LMCACHE_BASE_HEAD}" =~ ^[01]$ ]] || {
    echo "VERIFY_LMCACHE_BASE_HEAD must be 0 or 1" >&2
    exit 1
  }
  if [[ "${VERIFY_LMCACHE_BASE_HEAD}" == "1" ]]; then
    current_lmcache_base="$(resolve_ref "${LMCACHE_REPO}" "${LMCACHE_REF}")"
    [[ "${current_lmcache_base}" == "${LMCACHE_COMMIT}" ]] || {
      echo "LMCache base advanced from ${LMCACHE_COMMIT} to ${current_lmcache_base}; rerun the release composer" >&2
      exit 1
    }
  fi
  if [[ -n "${LMCACHE_PATCH_SHA256}" && \
        "${LMCACHE_PATCH_SHA256}" != "${lmcache_lock_patch_sha}" ]]; then
    echo "LMCache integration lock patch SHA mismatch: ${lmcache_lock_patch_sha} != ${LMCACHE_PATCH_SHA256}" >&2
    exit 1
  fi
  LMCACHE_PATCH_SHA256="${lmcache_lock_patch_sha}"
fi

b12x_local_patch_sha=""
b12x_local_patch_path=""
if [[ -n "${B12X_PATCH_FILE}" ]]; then
  if [[ -f "${B12X_PATCH_FILE}" ]]; then
    b12x_local_patch_path="${B12X_PATCH_FILE}"
  elif [[ -f "patches/${B12X_PATCH_FILE}" ]]; then
    b12x_local_patch_path="patches/${B12X_PATCH_FILE}"
  fi
  [[ -n "${b12x_local_patch_path}" ]] || {
    echo "B12X_PATCH_FILE does not exist: ${B12X_PATCH_FILE}" >&2
    exit 1
  }
  b12x_local_patch_sha="$(sha256sum "${b12x_local_patch_path}" | awk '{print $1}')"
  if [[ -n "${B12X_PATCH_SHA256}" && "${b12x_local_patch_sha}" != "${B12X_PATCH_SHA256}" ]]; then
    echo "B12X_PATCH_FILE SHA256 mismatch: got ${b12x_local_patch_sha}, expected ${B12X_PATCH_SHA256}" >&2
    exit 1
  fi
fi

local_patch_sha=""
local_patch_path=""
if [[ -n "${VLLM_PATCH_FILE}" ]]; then
  if [[ -f "${VLLM_PATCH_FILE}" ]]; then
    local_patch_path="${VLLM_PATCH_FILE}"
  elif [[ -f "patches/${VLLM_PATCH_FILE}" ]]; then
    local_patch_path="patches/${VLLM_PATCH_FILE}"
  fi
  [[ -n "${local_patch_path}" ]] || {
    echo "VLLM_PATCH_FILE does not exist: ${VLLM_PATCH_FILE}" >&2
    exit 1
  }
  local_patch_sha="$(sha256sum "${local_patch_path}" | awk '{print $1}')"
  if [[ -n "${VLLM_PATCH_SHA256}" && "${local_patch_sha}" != "${VLLM_PATCH_SHA256}" ]]; then
    echo "VLLM_PATCH_FILE SHA256 mismatch: got ${local_patch_sha}, expected ${VLLM_PATCH_SHA256}" >&2
    exit 1
  fi
fi

lmcache_local_patch_sha=""
lmcache_local_patch_path=""
if [[ -n "${LMCACHE_PATCH_FILE}" ]]; then
  if [[ -f "${LMCACHE_PATCH_FILE}" ]]; then
    lmcache_local_patch_path="${LMCACHE_PATCH_FILE}"
  elif [[ -f "patches/${LMCACHE_PATCH_FILE}" ]]; then
    lmcache_local_patch_path="patches/${LMCACHE_PATCH_FILE}"
  fi
  [[ -n "${lmcache_local_patch_path}" ]] || {
    echo "LMCACHE_PATCH_FILE does not exist: ${LMCACHE_PATCH_FILE}" >&2
    exit 1
  }
  lmcache_local_patch_sha="$(sha256sum "${lmcache_local_patch_path}" | awk '{print $1}')"
  if [[ -n "${LMCACHE_PATCH_SHA256}" && "${lmcache_local_patch_sha}" != "${LMCACHE_PATCH_SHA256}" ]]; then
    echo "LMCACHE_PATCH_FILE SHA256 mismatch: got ${lmcache_local_patch_sha}, expected ${LMCACHE_PATCH_SHA256}" >&2
    exit 1
  fi
  LMCACHE_PATCH_SHA256="${lmcache_local_patch_sha}"
fi

if [[ -n "${DEEPGEMM_PATCH_FILE}" ]]; then
  deepgemm_local_patch_path="patches/${DEEPGEMM_PATCH_FILE}"
  if [[ ! -f "${deepgemm_local_patch_path}" ]]; then
    echo "DEEPGEMM_PATCH_FILE not found: ${deepgemm_local_patch_path}" >&2
    exit 1
  fi
  deepgemm_local_patch_sha="$(sha256sum "${deepgemm_local_patch_path}" | awk '{print $1}')"
  if [[ -n "${DEEPGEMM_PATCH_SHA256}" && "${deepgemm_local_patch_sha}" != "${DEEPGEMM_PATCH_SHA256}" ]]; then
    echo "DEEPGEMM_PATCH_FILE SHA256 mismatch: got ${deepgemm_local_patch_sha}, expected ${DEEPGEMM_PATCH_SHA256}" >&2
    exit 1
  fi
  DEEPGEMM_PATCH_SHA256="${deepgemm_local_patch_sha}"
fi

runtime_files_sha="$({
  sha256sum "${DOCKERFILE}"
  sha256sum tests/verify_xgrammar_required_tools.py
  find launchers -type f -print0 | sort -z | xargs -0 sha256sum
} | sha256sum | awk '{print $1}')"

cache_hash="$(printf '%s\n' \
  "SYSTEM_BASE_IMAGE=${SYSTEM_BASE_IMAGE}" \
  "BUILD_BASE_IMAGE_TAG=${BUILD_BASE_IMAGE_TAG}" \
  "NCCL_REPO=${NCCL_REPO}" \
  "NCCL_REF=${NCCL_REF}" \
  "NCCL_COMMIT=${NCCL_COMMIT}" \
  "FLASHINFER_REPO=${FLASHINFER_REPO}" \
  "FLASHINFER_REF=${FLASHINFER_REF}" \
  "FLASHINFER_COMMIT=${FLASHINFER_COMMIT}" \
  "FLASHINFER_BUILD_CUBIN=${FLASHINFER_BUILD_CUBIN}" \
  "DEEPGEMM_REPO=${DEEPGEMM_REPO}" \
  "DEEPGEMM_REF=${DEEPGEMM_REF}" \
  "DEEPGEMM_COMMIT=${DEEPGEMM_COMMIT}" \
  "DEEPGEMM_PATCH_FILE=${DEEPGEMM_PATCH_FILE}" \
  "DEEPGEMM_PATCH_SHA256=${DEEPGEMM_PATCH_SHA256}" \
  "SKIP_EXLLAMAV3=${SKIP_EXLLAMAV3}" \
  "EXLLAMAV3_REPO=${EXLLAMAV3_REPO}" \
  "EXLLAMAV3_REF=${EXLLAMAV3_REF}" \
  "EXLLAMAV3_COMMIT=${EXLLAMAV3_COMMIT}" \
  "B12X_REPO=${B12X_REPO}" \
  "B12X_REF=${B12X_REF}" \
  "B12X_COMMIT=${B12X_COMMIT}" \
  "B12X_PATCH_SHA256=${B12X_PATCH_SHA256}" \
  "B12X_PATCH_FILE_SHA256=${b12x_local_patch_sha}" \
  "B12X_INTEGRATION_BASE_COMMIT=${B12X_INTEGRATION_BASE_COMMIT}" \
  "B12X_INTEGRATION_TREE=${B12X_INTEGRATION_TREE}" \
  "B12X_INTEGRATION_PRS=${B12X_INTEGRATION_PRS}" \
  "B12X_INTEGRATION_LOCK_SHA256=${B12X_INTEGRATION_LOCK_SHA256}" \
  "VLLM_REPO=${VLLM_REPO}" \
  "VLLM_REF=${VLLM_REF}" \
  "VLLM_COMMIT=${VLLM_COMMIT}" \
  "VLLM_PATCH_URL=${VLLM_PATCH_URL}" \
  "VLLM_PATCH_SHA256=${VLLM_PATCH_SHA256}" \
  "VLLM_PATCH_FILE_SHA256=${local_patch_sha}" \
  "VLLM_INTEGRATION_BASE_COMMIT=${VLLM_INTEGRATION_BASE_COMMIT}" \
  "VLLM_INTEGRATION_TREE=${VLLM_INTEGRATION_TREE}" \
  "VLLM_INTEGRATION_PRS=${VLLM_INTEGRATION_PRS}" \
  "VLLM_INTEGRATION_LOCK_SHA256=${VLLM_INTEGRATION_LOCK_SHA256}" \
  "VLLM_BUILD_VERSION=${VLLM_BUILD_VERSION}" \
  "LAUNCHER_REPO=${LAUNCHER_REPO}" \
  "LAUNCHER_REF=${LAUNCHER_REF}" \
  "LAUNCHER_COMMIT=${LAUNCHER_COMMIT}" \
  "CUTLASS_REPO=${CUTLASS_REPO}" \
  "CUTLASS_REF=${CUTLASS_REF}" \
  "CUTLASS_COMMIT=${CUTLASS_COMMIT}" \
  "CUTLASS_DSL_VERSION=${CUTLASS_DSL_VERSION}" \
  "TOKENSPEED_MLA_VERSION=${TOKENSPEED_MLA_VERSION}" \
  "TVM_FFI_VERSION=${TVM_FFI_VERSION}" \
  "TRITON_KERNELS_REPO=${TRITON_KERNELS_REPO}" \
  "TRITON_KERNELS_REF=${TRITON_KERNELS_REF}" \
  "TRITON_KERNELS_COMMIT=${TRITON_KERNELS_COMMIT}" \
  "INSTANTTENSOR_REPO=${INSTANTTENSOR_REPO}" \
  "INSTANTTENSOR_REF=${INSTANTTENSOR_REF}" \
  "INSTANTTENSOR_COMMIT=${INSTANTTENSOR_COMMIT}" \
  "LMCACHE_REPO=${LMCACHE_REPO}" \
  "LMCACHE_REF=${LMCACHE_REF}" \
  "LMCACHE_COMMIT=${LMCACHE_COMMIT}" \
  "LMCACHE_PATCH_FILE_SHA256=${lmcache_local_patch_sha}" \
  "LMCACHE_INTEGRATION_BASE_COMMIT=${LMCACHE_INTEGRATION_BASE_COMMIT}" \
  "LMCACHE_INTEGRATION_TREE=${LMCACHE_INTEGRATION_TREE}" \
  "LMCACHE_INTEGRATION_PRS=${LMCACHE_INTEGRATION_PRS}" \
  "LMCACHE_INTEGRATION_LOCK_SHA256=${LMCACHE_INTEGRATION_LOCK_SHA256}" \
  "LMCACHE_BUILD_VERSION=${LMCACHE_BUILD_VERSION}" \
  "XGRAMMAR_REPO=${XGRAMMAR_REPO}" \
  "XGRAMMAR_REF=${XGRAMMAR_REF}" \
  "XGRAMMAR_COMMIT=${XGRAMMAR_COMMIT}" \
  "XGRAMMAR_VERSION=${XGRAMMAR_VERSION}" \
  "XGRAMMAR_TRANSFORMERS5_COMPAT=${XGRAMMAR_TRANSFORMERS5_COMPAT}" \
  "HUMMING_KERNELS_SPEC=${HUMMING_KERNELS_SPEC}" \
  "VLLM_RUNTIME_EXTRA_PACKAGES=${VLLM_RUNTIME_EXTRA_PACKAGES}" \
  "RUNTIME_FILES_SHA256=${runtime_files_sha}" \
  | sha256sum | awk '{print substr($1, 1, 16)}')"
vllm_cache_id="${VLLM_COMMIT:0:10}"
b12x_cache_id="${B12X_COMMIT:0:10}"
vllm_cache_id="${vllm_cache_id:-unpinned}"
b12x_cache_id="${b12x_cache_id:-unpinned}"
CACHE_FINGERPRINT="${CACHE_FINGERPRINT:-vllm${vllm_cache_id}-b12x${b12x_cache_id}-${cache_hash}}"
if [[ ! "${CACHE_FINGERPRINT}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ || "${CACHE_FINGERPRINT}" == *..* ]]; then
  echo "Invalid CACHE_FINGERPRINT: ${CACHE_FINGERPRINT}" >&2
  exit 1
fi

echo "Building ${IMAGE}"
echo "  SYSTEM_BASE_IMAGE=${SYSTEM_BASE_IMAGE}"
echo "  BUILD_BASE_IMAGE_TAG=${BUILD_BASE_IMAGE_TAG}"
echo "  BUILD_BASE_IMAGE=${BUILD_BASE_IMAGE}"
echo "  PUSH_BASE_IMAGE=${PUSH_BASE_IMAGE}"
echo "  MAX_JOBS=${MAX_JOBS}"
echo "  VLLM_MAX_JOBS=${VLLM_MAX_JOBS}"
echo "  NVCC_THREADS=${NVCC_THREADS}"
echo "  VLLM_NVCC_THREADS=${VLLM_NVCC_THREADS}"
echo "  FLASHINFER_REF=${FLASHINFER_REF} ${FLASHINFER_COMMIT}"
echo "  FLASHINFER_BUILD_CUBIN=${FLASHINFER_BUILD_CUBIN}"
echo "  DEEPGEMM_REF=${DEEPGEMM_REF} ${DEEPGEMM_COMMIT}"
echo "  DEEPGEMM_PATCH_FILE=${DEEPGEMM_PATCH_FILE} sha256=${DEEPGEMM_PATCH_SHA256}"
echo "  DOCKERFILE=${DOCKERFILE} engine=${CONTAINER_ENGINE}"
echo "  ARCHS torch=${TORCH_CUDA_ARCH_LIST_ARG} cmake=${CMAKE_CUDA_ARCHITECTURES_ARG} flashinfer=${FLASHINFER_CUDA_ARCH_LIST_ARG} plain=${PLAIN_CUDA_ARCH_LIST_ARG}"
echo "  EXLLAMAV3_REF=${EXLLAMAV3_REF} ${EXLLAMAV3_COMMIT} skip=${SKIP_EXLLAMAV3}"
echo "  B12X_REF=${B12X_REF} ${B12X_COMMIT}"
echo "  B12X_PATCH_SHA256=${B12X_PATCH_SHA256}"
echo "  B12X_PATCH_FILE=${B12X_PATCH_FILE}"
echo "  B12X_INTEGRATION_BASE_COMMIT=${B12X_INTEGRATION_BASE_COMMIT}"
echo "  B12X_INTEGRATION_TREE=${B12X_INTEGRATION_TREE}"
echo "  B12X_INTEGRATION_PRS=${B12X_INTEGRATION_PRS}"
echo "  B12X_INTEGRATION_LOCK_SHA256=${B12X_INTEGRATION_LOCK_SHA256}"
echo "  VLLM_REF=${VLLM_REF} ${VLLM_COMMIT}"
echo "  VLLM_PATCH_URL=${VLLM_PATCH_URL}"
echo "  VLLM_PATCH_SHA256=${VLLM_PATCH_SHA256}"
echo "  VLLM_PATCH_FILE=${VLLM_PATCH_FILE}"
echo "  VLLM_INTEGRATION_BASE_COMMIT=${VLLM_INTEGRATION_BASE_COMMIT}"
echo "  VLLM_INTEGRATION_TREE=${VLLM_INTEGRATION_TREE}"
echo "  VLLM_INTEGRATION_PRS=${VLLM_INTEGRATION_PRS}"
echo "  VLLM_INTEGRATION_LOCK_SHA256=${VLLM_INTEGRATION_LOCK_SHA256}"
echo "  LAUNCHER_REF=${LAUNCHER_REF} ${LAUNCHER_COMMIT}"
echo "  VLLM_REQUIRED_LAUNCHERS=${VLLM_REQUIRED_LAUNCHERS}"
echo "  CUTLASS_REF=${CUTLASS_REF} ${CUTLASS_COMMIT}"
echo "  CUTLASS_DSL_VERSION=${CUTLASS_DSL_VERSION}"
echo "  TOKENSPEED_MLA_VERSION=${TOKENSPEED_MLA_VERSION}"
echo "  TVM_FFI_VERSION=${TVM_FFI_VERSION}"
echo "  TRITON_KERNELS_REF=${TRITON_KERNELS_REF} ${TRITON_KERNELS_COMMIT}"
echo "  INSTANTTENSOR_REF=${INSTANTTENSOR_REF} ${INSTANTTENSOR_COMMIT}"
echo "  NCCL_REF=${NCCL_REF} ${NCCL_COMMIT}"
echo "  HUMMING_KERNELS_SPEC=${HUMMING_KERNELS_SPEC}"
echo "  VLLM_RUNTIME_EXTRA_PACKAGES=${VLLM_RUNTIME_EXTRA_PACKAGES}"
echo "  LMCACHE=${LMCACHE_REPO} ${LMCACHE_REF} ${LMCACHE_COMMIT}"
echo "  LMCACHE_PATCH_FILE=${LMCACHE_PATCH_FILE} sha256=${LMCACHE_PATCH_SHA256}"
echo "  LMCACHE_INTEGRATION_TREE=${LMCACHE_INTEGRATION_TREE}"
echo "  LMCACHE_INTEGRATION_PRS=${LMCACHE_INTEGRATION_PRS}"
echo "  XGRAMMAR=${XGRAMMAR_REPO} ${XGRAMMAR_REF} ${XGRAMMAR_COMMIT} version=${XGRAMMAR_VERSION} transformers5_compat=${XGRAMMAR_TRANSFORMERS5_COMPAT}"
echo "  CACHE_FINGERPRINT=${CACHE_FINGERPRINT}"

# podman/buildah does not know --progress; docker BuildKit wants plain output.
# For podman, force docker image format: the OCI format does not persist the
# SHELL directive, so RUN steps in stages built FROM a tagged OCI base image
# would silently fall back to /bin/sh.
PROGRESS_ARGS=()
if [[ "${CONTAINER_ENGINE}" == "docker" ]]; then
  PROGRESS_ARGS=(--progress=plain)
else
  PROGRESS_ARGS=(--format docker)
fi

if [[ "${BUILD_BASE_IMAGE}" == "1" ]]; then
  DOCKER_BUILDKIT=1 "${CONTAINER_ENGINE}" build \
    --target vllm-b12x-cu132-system-base-build \
    --build-arg NCCL_REPO="${NCCL_REPO}" \
    --build-arg NCCL_REF="${NCCL_REF}" \
    --build-arg NCCL_COMMIT="${NCCL_COMMIT}" \
    "${PROGRESS_ARGS[@]}" \
    -f "${DOCKERFILE}" \
    -t "${SYSTEM_BASE_IMAGE}" \
    "$@" \
    .

  DOCKER_BUILDKIT=1 "${CONTAINER_ENGINE}" build \
    --target vllm-b12x-cu132-build-base-build \
    --build-arg VLLM_B12X_CU132_SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE}" \
    --build-arg CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION}" \
    --build-arg TORCH_CUDA_ARCH_LIST_ARG="${TORCH_CUDA_ARCH_LIST_ARG}" \
    --build-arg CMAKE_CUDA_ARCHITECTURES_ARG="${CMAKE_CUDA_ARCHITECTURES_ARG}" \
    --build-arg FLASHINFER_CUDA_ARCH_LIST_ARG="${FLASHINFER_CUDA_ARCH_LIST_ARG}" \
    "${PROGRESS_ARGS[@]}" \
    -f "${DOCKERFILE}" \
    -t "${BUILD_BASE_IMAGE_TAG}" \
    "$@" \
    .

  if [[ "${PUSH_BASE_IMAGE}" == "1" ]]; then
    "${CONTAINER_ENGINE}" push "${SYSTEM_BASE_IMAGE}"
    "${CONTAINER_ENGINE}" push "${BUILD_BASE_IMAGE_TAG}"
  fi
fi

DOCKER_BUILDKIT=1 "${CONTAINER_ENGINE}" build \
  --build-arg VLLM_B12X_CU132_SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE}" \
  --build-arg VLLM_B12X_CU132_BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE_TAG}" \
  --build-arg TORCH_CUDA_ARCH_LIST_ARG="${TORCH_CUDA_ARCH_LIST_ARG}" \
  --build-arg CMAKE_CUDA_ARCHITECTURES_ARG="${CMAKE_CUDA_ARCHITECTURES_ARG}" \
  --build-arg FLASHINFER_CUDA_ARCH_LIST_ARG="${FLASHINFER_CUDA_ARCH_LIST_ARG}" \
  --build-arg PLAIN_CUDA_ARCH_LIST_ARG="${PLAIN_CUDA_ARCH_LIST_ARG}" \
  --build-arg DEEPGEMM_PATCH_FILE="${DEEPGEMM_PATCH_FILE}" \
  --build-arg DEEPGEMM_PATCH_SHA256="${DEEPGEMM_PATCH_SHA256}" \
  --build-arg SKIP_EXLLAMAV3="${SKIP_EXLLAMAV3}" \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  --build-arg VLLM_MAX_JOBS="${VLLM_MAX_JOBS}" \
  --build-arg NVCC_THREADS="${NVCC_THREADS}" \
  --build-arg VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS}" \
  --build-arg NCCL_REPO="${NCCL_REPO}" \
  --build-arg NCCL_REF="${NCCL_REF}" \
  --build-arg NCCL_COMMIT="${NCCL_COMMIT}" \
  --build-arg FLASHINFER_REPO="${FLASHINFER_REPO}" \
  --build-arg FLASHINFER_REF="${FLASHINFER_REF}" \
  --build-arg FLASHINFER_COMMIT="${FLASHINFER_COMMIT}" \
  --build-arg FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN}" \
  --build-arg DEEPGEMM_REPO="${DEEPGEMM_REPO}" \
  --build-arg DEEPGEMM_REF="${DEEPGEMM_REF}" \
  --build-arg DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT}" \
  --build-arg EXLLAMAV3_REPO="${EXLLAMAV3_REPO}" \
  --build-arg EXLLAMAV3_REF="${EXLLAMAV3_REF}" \
  --build-arg EXLLAMAV3_COMMIT="${EXLLAMAV3_COMMIT}" \
  --build-arg B12X_REPO="${B12X_REPO}" \
  --build-arg B12X_REF="${B12X_REF}" \
  --build-arg B12X_COMMIT="${B12X_COMMIT}" \
  --build-arg B12X_PATCH_SHA256="${B12X_PATCH_SHA256}" \
  --build-arg B12X_PATCH_FILE="${B12X_PATCH_FILE}" \
  --build-arg B12X_INTEGRATION_BASE_COMMIT="${B12X_INTEGRATION_BASE_COMMIT}" \
  --build-arg B12X_INTEGRATION_TREE="${B12X_INTEGRATION_TREE}" \
  --build-arg B12X_INTEGRATION_PRS="${B12X_INTEGRATION_PRS}" \
  --build-arg B12X_INTEGRATION_LOCK_SHA256="${B12X_INTEGRATION_LOCK_SHA256}" \
  --build-arg VLLM_REPO="${VLLM_REPO}" \
  --build-arg VLLM_REF="${VLLM_REF}" \
  --build-arg VLLM_COMMIT="${VLLM_COMMIT}" \
  --build-arg VLLM_PATCH_URL="${VLLM_PATCH_URL}" \
  --build-arg VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256}" \
  --build-arg VLLM_PATCH_FILE="${VLLM_PATCH_FILE}" \
  --build-arg VLLM_INTEGRATION_BASE_COMMIT="${VLLM_INTEGRATION_BASE_COMMIT}" \
  --build-arg VLLM_INTEGRATION_TREE="${VLLM_INTEGRATION_TREE}" \
  --build-arg VLLM_INTEGRATION_PRS="${VLLM_INTEGRATION_PRS}" \
  --build-arg VLLM_INTEGRATION_LOCK_SHA256="${VLLM_INTEGRATION_LOCK_SHA256}" \
  --build-arg VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION}" \
  --build-arg LAUNCHER_REPO="${LAUNCHER_REPO}" \
  --build-arg LAUNCHER_REF="${LAUNCHER_REF}" \
  --build-arg LAUNCHER_COMMIT="${LAUNCHER_COMMIT}" \
  --build-arg VLLM_REQUIRED_LAUNCHERS="${VLLM_REQUIRED_LAUNCHERS}" \
  --build-arg CUTLASS_REPO="${CUTLASS_REPO}" \
  --build-arg CUTLASS_REF="${CUTLASS_REF}" \
  --build-arg CUTLASS_COMMIT="${CUTLASS_COMMIT}" \
  --build-arg CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION}" \
  --build-arg TOKENSPEED_MLA_VERSION="${TOKENSPEED_MLA_VERSION}" \
  --build-arg TVM_FFI_VERSION="${TVM_FFI_VERSION}" \
  --build-arg TRITON_KERNELS_REPO="${TRITON_KERNELS_REPO}" \
  --build-arg TRITON_KERNELS_REF="${TRITON_KERNELS_REF}" \
  --build-arg TRITON_KERNELS_COMMIT="${TRITON_KERNELS_COMMIT}" \
  --build-arg INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO}" \
  --build-arg INSTANTTENSOR_REF="${INSTANTTENSOR_REF}" \
  --build-arg INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT}" \
  --build-arg LMCACHE_REPO="${LMCACHE_REPO}" \
  --build-arg LMCACHE_REF="${LMCACHE_REF}" \
  --build-arg LMCACHE_COMMIT="${LMCACHE_COMMIT}" \
  --build-arg LMCACHE_PATCH_FILE="${LMCACHE_PATCH_FILE}" \
  --build-arg LMCACHE_PATCH_SHA256="${LMCACHE_PATCH_SHA256}" \
  --build-arg LMCACHE_BUILD_VERSION="${LMCACHE_BUILD_VERSION}" \
  --build-arg LMCACHE_INTEGRATION_BASE_COMMIT="${LMCACHE_INTEGRATION_BASE_COMMIT}" \
  --build-arg LMCACHE_INTEGRATION_TREE="${LMCACHE_INTEGRATION_TREE}" \
  --build-arg LMCACHE_INTEGRATION_PRS="${LMCACHE_INTEGRATION_PRS}" \
  --build-arg LMCACHE_INTEGRATION_LOCK_SHA256="${LMCACHE_INTEGRATION_LOCK_SHA256}" \
  --build-arg XGRAMMAR_REPO="${XGRAMMAR_REPO}" \
  --build-arg XGRAMMAR_REF="${XGRAMMAR_REF}" \
  --build-arg XGRAMMAR_COMMIT="${XGRAMMAR_COMMIT}" \
  --build-arg XGRAMMAR_VERSION="${XGRAMMAR_VERSION}" \
  --build-arg XGRAMMAR_TRANSFORMERS5_COMPAT="${XGRAMMAR_TRANSFORMERS5_COMPAT}" \
  --build-arg HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC}" \
  --build-arg VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES}" \
  --build-arg CACHE_FINGERPRINT="${CACHE_FINGERPRINT}" \
  "${PROGRESS_ARGS[@]}" \
  -f "${DOCKERFILE}" \
  -t "${IMAGE}" \
  "$@" \
  .

image_cache_fingerprint="$("${CONTAINER_ENGINE}" image inspect "${IMAGE}" --format '{{index .Config.Labels "local-inference.cache.fingerprint"}}')"
[[ "${image_cache_fingerprint}" == "${CACHE_FINGERPRINT}" ]] || {
  echo "Image cache fingerprint mismatch: got ${image_cache_fingerprint}, expected ${CACHE_FINGERPRINT}" >&2
  exit 1
}

image_exllamav3_commit="$("${CONTAINER_ENGINE}" image inspect "${IMAGE}" --format '{{index .Config.Labels "local-inference.exllamav3.commit"}}')"
[[ "${image_exllamav3_commit}" == "${EXLLAMAV3_COMMIT}" ]] || {
  echo "Image EXL3 source mismatch: got ${image_exllamav3_commit}, expected ${EXLLAMAV3_COMMIT}" >&2
  exit 1
}

image_lmcache_commit="$("${CONTAINER_ENGINE}" image inspect "${IMAGE}" --format '{{index .Config.Labels "local-inference.lmcache.commit"}}')"
[[ "${image_lmcache_commit}" == "${LMCACHE_COMMIT}" ]] || {
  echo "Image LMCache source mismatch: got ${image_lmcache_commit}, expected ${LMCACHE_COMMIT}" >&2
  exit 1
}
if [[ "${REQUIRE_CLEAN_LMCACHE_COMPOSITION}" == "1" ]]; then
  image_lmcache_tree="$("${CONTAINER_ENGINE}" image inspect "${IMAGE}" --format '{{index .Config.Labels "local-inference.lmcache.integration.tree"}}')"
  image_lmcache_prs="$("${CONTAINER_ENGINE}" image inspect "${IMAGE}" --format '{{index .Config.Labels "local-inference.lmcache.integration.prs"}}')"
  image_lmcache_lock_sha="$("${CONTAINER_ENGINE}" image inspect "${IMAGE}" --format '{{index .Config.Labels "local-inference.lmcache.integration.lock_sha256"}}')"
  [[ "${image_lmcache_tree}" == "${LMCACHE_INTEGRATION_TREE}" ]] || {
    echo "Image LMCache integration tree mismatch: got ${image_lmcache_tree}, expected ${LMCACHE_INTEGRATION_TREE}" >&2
    exit 1
  }
  [[ "${image_lmcache_prs}" == "${LMCACHE_INTEGRATION_PRS}" ]] || {
    echo "Image LMCache integration PR list mismatch" >&2
    exit 1
  }
  [[ "${image_lmcache_lock_sha}" == "${LMCACHE_INTEGRATION_LOCK_SHA256}" ]] || {
    echo "Image LMCache integration lock mismatch" >&2
    exit 1
  }
fi

if [[ -n "${XGRAMMAR_REF}" ]]; then
  image_xgrammar_commit="$("${CONTAINER_ENGINE}" image inspect "${IMAGE}" --format '{{index .Config.Labels "local-inference.xgrammar.commit"}}')"
  image_xgrammar_version="$("${CONTAINER_ENGINE}" image inspect "${IMAGE}" --format '{{index .Config.Labels "local-inference.xgrammar.version"}}')"
  image_xgrammar_transformers5_compat="$("${CONTAINER_ENGINE}" image inspect "${IMAGE}" --format '{{index .Config.Labels "local-inference.xgrammar.transformers5_compat"}}')"
  [[ "${image_xgrammar_commit}" == "${XGRAMMAR_COMMIT}" ]] || {
    echo "Image xgrammar source mismatch: got ${image_xgrammar_commit}, expected ${XGRAMMAR_COMMIT}" >&2
    exit 1
  }
  [[ "${image_xgrammar_version}" == "${XGRAMMAR_VERSION}" ]] || {
    echo "Image xgrammar version mismatch: got ${image_xgrammar_version}, expected ${XGRAMMAR_VERSION}" >&2
    exit 1
  }
  [[ "${image_xgrammar_transformers5_compat}" == "${XGRAMMAR_TRANSFORMERS5_COMPAT}" ]] || {
    echo "Image xgrammar Transformers compatibility label mismatch: got ${image_xgrammar_transformers5_compat}, expected ${XGRAMMAR_TRANSFORMERS5_COMPAT}" >&2
    exit 1
  }
fi

image_env="$("${CONTAINER_ENGINE}" image inspect "${IMAGE}" --format '{{range .Config.Env}}{{println .}}{{end}}')"
cache_root="/cache/jit/${CACHE_FINGERPRINT}"
for expected in \
  "LOCAL_INFERENCE_CACHE_FINGERPRINT=${CACHE_FINGERPRINT}" \
  "XDG_CACHE_HOME=${cache_root}" \
  "VLLM_CACHE_ROOT=${cache_root}/vllm" \
  "TRITON_CACHE_DIR=${cache_root}/triton" \
  "TORCHINDUCTOR_CACHE_DIR=${cache_root}/torchinductor" \
  "B12X_CUTE_COMPILE_CACHE_DIR=${cache_root}/b12x-cute" \
  "VLLM_EXL3_EXT_PATH=/opt/exllamav3" \
  "MM_SPARSE_ATTN_AOT_CACHE=${cache_root}/minfer/mm_sparse_attn"; do
  grep -Fxq "${expected}" <<<"${image_env}" || {
    echo "Image is missing cache environment: ${expected}" >&2
    exit 1
  }
done
