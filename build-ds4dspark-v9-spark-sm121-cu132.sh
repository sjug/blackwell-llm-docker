#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# DS4 / DSpark v9 image rebuilt for DGX Spark (GB10, SM121, aarch64).
#
# Same source pins as the x86_64 SM120 v9 image
# voipmonitor/vllm:eldritch-enlightenment-v45c1582-b12xf3686b5-pc1441b5-cu132-20260704
# (models/ds4dspark-v9.md), retargeted at SM121:
#
# vLLM:
# - local-inference-lab/vllm dev/eldritch-enlightenment
#   @ 45c1582e9b80ba83e71c3a6458e71da4736fbdc4
# - plus the local B12X indexer decode-warmup fallback patch. The patch file
#   here was regenerated from /opt/vllm of the pushed v9 image (digest
#   sha256:7703639ae953...); it applies cleanly to 45c1582 and the patched
#   file is byte-identical to the shipped image, but the patch file hash
#   differs from the original c1441b5... because it was re-created.
#
# B12X:
# - lukealonso/b12x master @ f3686b555d639823b276c2080f173145eed7f007
#   (pure-Python CuTe DSL package; kernels JIT per-device -> sm_121a on GB10)
#
# FlashInfer 25dd814, DeepGEMM nv_dev 2073ddb, CUTLASS d80a4e5: same as v9.
# DeepGEMM gets a one-line SM121 fix so the fp8 (paged) MQA logits generators
# emit the existing sm120 impl headers on a (12,1) device even when they run
# before the JIT compiler singleton initializes.
#
# SM121 arch targets: TORCH_CUDA_ARCH_LIST=12.1a, CMAKE_CUDA_ARCHITECTURES=121a,
# FLASHINFER_CUDA_ARCH_LIST=12.1a. FlashInfer intentionally refuses to run
# SM120 cubins on SM121 and only enables its sm121 AOT modules when 12.1a is
# listed, so 12.0f is NOT carried over from the x86 build.
#
# This is an aarch64-native build: run it on a DGX Spark node (rusty/toby).
# The base images must be (re)built there too, hence BUILD_BASE_IMAGE=1.
# All build stages are compile-only, so rootless podman works.

export CONTAINER_ENGINE="${CONTAINER_ENGINE:-podman}"

export IMAGE="${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-dspark-spark-sm121-v45c1582-b12xf3686b5-cu132-20260708}"
export SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-voipmonitor/vllm:spark-sm121-cu132-system-base-20260708}"
export BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-voipmonitor/vllm:spark-sm121-cu132-build-base-20260708}"
export BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-1}"
export PUSH_BASE_IMAGE="${PUSH_BASE_IMAGE:-0}"

# SM121 / GB10 arch targets.
export TORCH_CUDA_ARCH_LIST_ARG="${TORCH_CUDA_ARCH_LIST_ARG:-12.1a}"
export CMAKE_CUDA_ARCHITECTURES_ARG="${CMAKE_CUDA_ARCHITECTURES_ARG:-121a}"
export FLASHINFER_CUDA_ARCH_LIST_ARG="${FLASHINFER_CUDA_ARCH_LIST_ARG:-12.1a}"

# GB10: 20 cores, 121 GB unified memory shared with the GPU. Keep compile
# parallelism well below the 16-GPU host defaults (64).
export MAX_JOBS="${MAX_JOBS:-12}"
export VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-12}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
export PIN_SOURCE_COMMITS="${PIN_SOURCE_COMMITS:-1}"

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-25dd814e03791e370f96c3148242f0dc8de504ac}"
export FLASHINFER_REF="${FLASHINFER_REF:-${FLASHINFER_COMMIT}}"
export FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-0}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
# nv_dev has moved past the v9 pin; fetch the pinned commit directly.
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-2073ddb2814892014c33ef4cd1c7d4c148baf1fe}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-${DEEPGEMM_COMMIT}}"
export DEEPGEMM_PATCH_FILE="${DEEPGEMM_PATCH_FILE:-deepgemm-sm121-mqa-logits-arch-number-20260708.patch}"

export B12X_REPO="${B12X_REPO:-https://github.com/lukealonso/b12x.git}"
export B12X_REF="${B12X_REF:-master}"
export B12X_COMMIT="${B12X_COMMIT:-f3686b555d639823b276c2080f173145eed7f007}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-dev/eldritch-enlightenment}"
export VLLM_COMMIT="${VLLM_COMMIT:-45c1582e9b80ba83e71c3a6458e71da4736fbdc4}"
export VLLM_PATCH_URL="${VLLM_PATCH_URL:-}"
export VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-}"
export VLLM_PATCH_FILE="${VLLM_PATCH_FILE:-vllm-b12x-indexer-warmup-fallback-20260704.patch}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev279+eldritch.enlightenment.v45c1582.b12xf3686b5.fi25dd814.sm121.cu132.20260708}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"

export CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
export CUTLASS_REF="${CUTLASS_REF:-d80a4e53b52b42550659a8696dab32705265e324}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-d80a4e53b52b42550659a8696dab32705265e324}"
export HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.6}"

sha256sum -c <<'EOF'
8d8c794e4a6d66c2b59b47c00818ab3440dc74f08b1845bc160968a63264602c  patches/vllm-b12x-indexer-warmup-fallback-20260704.patch
EOF

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: this is an aarch64-native (DGX Spark) build; run on rusty/toby." >&2
  exit 1
fi

FLASHINFER_WHEEL_DIR=".tmp-flashinfer-wheels"
FLASHINFER_WHEEL_STASH=".tmp-flashinfer-wheels.disabled-ds4dspark-v9-spark-20260708"
if [[ "${FORCE_FLASHINFER_SOURCE:-1}" == "1" ]] \
  && compgen -G "${FLASHINFER_WHEEL_DIR}/flashinfer_*.whl" >/dev/null; then
  rm -rf "${FLASHINFER_WHEEL_STASH}"
  mkdir -p "${FLASHINFER_WHEEL_STASH}"
  mv "${FLASHINFER_WHEEL_DIR}"/flashinfer_*.whl "${FLASHINFER_WHEEL_STASH}"/
  restore_flashinfer_wheels() {
    if compgen -G "${FLASHINFER_WHEEL_STASH}/flashinfer_*.whl" >/dev/null; then
      mv "${FLASHINFER_WHEEL_STASH}"/flashinfer_*.whl "${FLASHINFER_WHEEL_DIR}"/
    fi
    rmdir "${FLASHINFER_WHEEL_STASH}" 2>/dev/null || true
  }
  trap restore_flashinfer_wheels EXIT
fi

./build-vllm-b12x-cu132.sh "$@"
