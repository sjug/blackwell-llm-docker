#!/usr/bin/env bash
set -euo pipefail

# SPARK/SM121 variant of build-deepseek-infernal-invocation-cu133-torch213.sh.
# Builds the Infernal Invocation r18 runtime for DGX Spark (GB10, aarch64).
# Overlay is THREE items (2026-08-18 review): CMake 12.1 arch acceptance,
# shm_broadcast busy_loop_s 1 -> 0.002 (the r17 base already carries the
# upstream #45224 queue-safety fix, verified behaviorally below), and the
# combined ExLlamaV3 patch (pin unmoved at 704aefd). The r15-era
# scratch-planner overlay is DROPPED: r17's #221 CuTe K6/MCG small-M path
# is SM121-admissible with per-weight launch binding and a coherent
# query/dispatch scratch contract, proven by the rebuilt matrix + graph
# gates. #305 ownership overlay is unnecessary: the r17 base clones every
# deferred accelerator tensor (_own_deferred_accelerator_tensors).
# B12X_DISABLE_STANDALONE_K6 is an INCOMPATIBLE runtime setting and is
# asserted absent. Online EXL3 K6 remains OPEN and gate-proven.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

release_name=${RELEASE_NAME:-infernal-invocation-r18-spark-sm121}
release_date=${RELEASE_DATE:-20260818}
revision=${REVISION:-r18}
composition_root=patches/releases/infernal-invocation-r18
base_image=${BASE_IMAGE:-localhost/voipmonitor/vllm:ii-spark-sm121-cu133-torch213-nccl2312-20260814-r1}
instanttensor_repo=${INSTANTTENSOR_REPO:-https://github.com/voipmonitor/InstantTensor.git}
instanttensor_commit=${INSTANTTENSOR_COMMIT:-49b4010afc1cae0441e71fe0b0bffc24fa05e932}
instanttensor_libaio_repo=${INSTANTTENSOR_LIBAIO_REPO:-https://pagure.io/libaio.git}
instanttensor_libaio_commit=${INSTANTTENSOR_LIBAIO_COMMIT:-1b18bfafc6a2f7b9fa2c6be77a95afed8b7be448}
instanttensor_libaio_tree=${INSTANTTENSOR_LIBAIO_TREE:-c9442e111b747e9329ea782c6edb9d13a827cc08}
exllamav3_repo=${EXLLAMAV3_REPO:-https://github.com/brandonmmusic-max/exllamav3.git}
exllamav3_commit=${EXLLAMAV3_COMMIT:-704aefd743b390af4bd0fb429d1906f9b964c7d8}
exllamav3_patch_file=${EXLLAMAV3_PATCH_FILE:-exllamav3-sm121-guards-and-encoder-contract-20260815.patch}
spark_overlay_file=${VLLM_SPARK_OVERLAY_FILE:-releases/infernal-invocation-r18-spark/vllm-spark-overlay.patch}
spark_overlay_tree=${VLLM_SPARK_OVERLAY_TREE:-f560085b72537567a2d2c5f3032b0bae61422cf4}
exllamav3_patch_sha256="$(sha256sum "patches/${exllamav3_patch_file}" | cut -d' ' -f1)"
spark_overlay_sha256="$(sha256sum "patches/${spark_overlay_file}" | cut -d' ' -f1)"

case "${revision}" in
  r[1-9]|r[1-9][0-9]*) ;;
  *) printf 'REVISION must use the rN form; got %s\n' "${revision}" >&2; exit 2 ;;
esac
[[ "${release_date}" =~ ^[0-9]{8}$ ]] || {
  printf 'RELEASE_DATE must use YYYYMMDD; got %s\n' "${release_date}" >&2
  exit 2
}

read_lock() {
  local component=$1 prefix=$2
  local lock="${composition_root}/${component}/integration.lock.json"
  local patch="${composition_root}/${component}/integration.patch"

  test -f "${lock}" || { printf 'Missing composition lock: %s\n' "${lock}" >&2; exit 1; }
  test -f "${patch}" || { printf 'Missing integration patch: %s\n' "${patch}" >&2; exit 1; }
  echo "$(jq -er '.result.patch_sha256' "${lock}")  ${patch}" | sha256sum -c - >/dev/null

  export "${prefix}_REPO=$(jq -er '.base.repository' "${lock}")"
  export "${prefix}_REF=$(jq -er '.base.ref | sub("^refs/heads/"; "")' "${lock}")"
  export "${prefix}_COMMIT=$(jq -er '.base.commit' "${lock}")"
  export "${prefix}_PATCH_FILE=${composition_root#patches/}/${component}/integration.patch"
  export "${prefix}_PATCH_SHA256=$(jq -er '.result.patch_sha256' "${lock}")"
  export "${prefix}_INTEGRATION_TREE=$(jq -er '.result.tree' "${lock}")"
  export "${prefix}_INTEGRATION_LOCK_SHA256=$(sha256sum "${lock}" | cut -d' ' -f1)"
  export "${prefix}_PRS=$(jq -er '[.pull_requests[] | "\(.number)@\(.head)"] | join(",")' "${lock}")"
}

read_lock vllm VLLM
read_lock b12x B12X
read_lock lmcache LMCACHE

test "${VLLM_REF}" = dev/infernal-invocation
test "${B12X_REF}" = master

vllm_package_version=${VLLM_PACKAGE_VERSION:-0.26.1rc0+infernal.invocation.cu133.${revision}.vllm${VLLM_INTEGRATION_TREE:0:7}.b12x${B12X_INTEGRATION_TREE:0:7}}
flashinfer_version=${FLASHINFER_VERSION:-0.6.18+cu133}
lmcache_build_version=${LMCACHE_BUILD_VERSION:-0.5.2+glm52dcp.5}
cache_fingerprint="cu133-torch213-spark-sm121-vllm${spark_overlay_tree:0:10}-b12x${B12X_INTEGRATION_TREE:0:10}-lmcache${LMCACHE_INTEGRATION_TREE:0:10}"
image=${IMAGE:-localhost/voipmonitor/vllm:infernal-invocation-r18-spark-sm121-vllm${spark_overlay_tree:0:7}-b12x${B12X_INTEGRATION_TREE:0:7}-fi1ac6942-cu133-torch213-${release_date}}

if [[ "${PRINT_RELEASE_CONFIG:-0}" == 1 ]]; then
  printf 'release=%s\nrevision=%s\nbase=%s\nimage=%s\n' \
    "${release_name}" "${revision}" "${base_image}" "${image}"
  printf 'vllm_ref=%s\nvllm_commit=%s\nvllm_tree=%s\nvllm_patch=%s\n' \
    "${VLLM_REF}" "${VLLM_COMMIT}" "${VLLM_INTEGRATION_TREE}" "${VLLM_PATCH_FILE}"
  printf 'b12x_ref=%s\nb12x_commit=%s\nb12x_tree=%s\nb12x_patch=%s\n' \
    "${B12X_REF}" "${B12X_COMMIT}" "${B12X_INTEGRATION_TREE}" "${B12X_PATCH_FILE}"
  printf 'lmcache_ref=%s\nlmcache_commit=%s\nlmcache_tree=%s\nlmcache_patch=%s\n' \
    "${LMCACHE_REF}" "${LMCACHE_COMMIT}" "${LMCACHE_INTEGRATION_TREE}" "${LMCACHE_PATCH_FILE}"
  printf 'torch=2.13.0\ncuda=13.3\nnccl=2.31.2\nflashinfer=%s\n' "${flashinfer_version}"
  exit 0
fi

if ! podman image inspect "${base_image}" >/dev/null 2>&1; then
  podman pull "${base_image}" || {
    printf 'CUDA 13.3 base image is unavailable. Build it with ./build-kimi-k3-cu133-torch213-base.sh.\n' >&2
    exit 1
  }
fi
base_image_id="$(podman image inspect "${base_image}" --format '{{.Id}}')"
docker_commit="$(git rev-parse HEAD)"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]] \
    && [[ "${ALLOW_DIRTY_BUILD:-0}" != 1 ]]; then
  printf 'Set ALLOW_DIRTY_BUILD=1 for an image whose recipe is not committed.\n' >&2
  git status --short >&2
  exit 1
fi

printf 'release=%s\nbase=%s (%s)\nimage=%s\n' \
  "${release_name}" "${base_image}" "${base_image_id}" "${image}"
printf 'vllm=%s + %s -> %s\n' "${VLLM_COMMIT}" "${VLLM_PRS}" "${VLLM_INTEGRATION_TREE}"
printf 'b12x=%s + %s -> %s\n' "${B12X_COMMIT}" "${B12X_PRS}" "${B12X_INTEGRATION_TREE}"
printf 'lmcache=%s + %s -> %s\n' "${LMCACHE_COMMIT}" "${LMCACHE_PRS}" "${LMCACHE_INTEGRATION_TREE}"

podman build \
  --pull=false \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "BASE_IMAGE_ID=${base_image_id}" \
  --build-arg "VLLM_REPO=${VLLM_REPO}" \
  --build-arg "VLLM_REF=${VLLM_REF}" \
  --build-arg "VLLM_COMMIT=${VLLM_COMMIT}" \
  --build-arg "VLLM_PATCH_FILE=${VLLM_PATCH_FILE}" \
  --build-arg "VLLM_PATCH_SHA256=${VLLM_PATCH_SHA256}" \
  --build-arg "VLLM_INTEGRATION_TREE=${VLLM_INTEGRATION_TREE}" \
  --build-arg "VLLM_INTEGRATION_LOCK_SHA256=${VLLM_INTEGRATION_LOCK_SHA256}" \
  --build-arg "VLLM_PRS=${VLLM_PRS}" \
  --build-arg "B12X_REPO=${B12X_REPO}" \
  --build-arg "B12X_REF=${B12X_REF}" \
  --build-arg "B12X_COMMIT=${B12X_COMMIT}" \
  --build-arg "B12X_PATCH_FILE=${B12X_PATCH_FILE}" \
  --build-arg "B12X_PATCH_SHA256=${B12X_PATCH_SHA256}" \
  --build-arg "B12X_INTEGRATION_TREE=${B12X_INTEGRATION_TREE}" \
  --build-arg "B12X_INTEGRATION_LOCK_SHA256=${B12X_INTEGRATION_LOCK_SHA256}" \
  --build-arg "B12X_PRS=${B12X_PRS}" \
  --build-arg "LMCACHE_REPO=${LMCACHE_REPO}" \
  --build-arg "LMCACHE_REF=${LMCACHE_REF}" \
  --build-arg "LMCACHE_COMMIT=${LMCACHE_COMMIT}" \
  --build-arg "LMCACHE_PATCH_FILE=${LMCACHE_PATCH_FILE}" \
  --build-arg "LMCACHE_PATCH_SHA256=${LMCACHE_PATCH_SHA256}" \
  --build-arg "LMCACHE_INTEGRATION_TREE=${LMCACHE_INTEGRATION_TREE}" \
  --build-arg "LMCACHE_INTEGRATION_LOCK_SHA256=${LMCACHE_INTEGRATION_LOCK_SHA256}" \
  --build-arg "LMCACHE_PRS=${LMCACHE_PRS}" \
  --build-arg "LMCACHE_BUILD_VERSION=${lmcache_build_version}" \
  --build-arg "INSTANTTENSOR_REPO=${instanttensor_repo}" \
  --build-arg "INSTANTTENSOR_COMMIT=${instanttensor_commit}" \
  --build-arg "INSTANTTENSOR_LIBAIO_REPO=${instanttensor_libaio_repo}" \
  --build-arg "INSTANTTENSOR_LIBAIO_COMMIT=${instanttensor_libaio_commit}" \
  --build-arg "INSTANTTENSOR_LIBAIO_TREE=${instanttensor_libaio_tree}" \
  --build-arg "EXLLAMAV3_REPO=${exllamav3_repo}" \
  --build-arg "EXLLAMAV3_COMMIT=${exllamav3_commit}" \
  --build-arg "EXLLAMAV3_PATCH_FILE=${exllamav3_patch_file}" \
  --build-arg "EXLLAMAV3_PATCH_SHA256=${exllamav3_patch_sha256}" \
  --build-arg "VLLM_SPARK_OVERLAY_FILE=${spark_overlay_file}" \
  --build-arg "VLLM_SPARK_OVERLAY_SHA256=${spark_overlay_sha256}" \
  --build-arg "VLLM_SPARK_OVERLAY_TREE=${spark_overlay_tree}" \
  --build-arg "VLLM_PACKAGE_VERSION=${vllm_package_version}" \
  --build-arg "FLASHINFER_VERSION=${flashinfer_version}" \
  --build-arg "RELEASE_NAME=${release_name}" \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --build-arg "CACHE_FINGERPRINT=${cache_fingerprint}" \
  --file Dockerfile.deepseek-ii-r18-spark-sm121 \
  --tag "${image}" \
  .

labels="$(podman image inspect "${image}" --format '{{json .Config.Labels}}')"
assert_label() {
  local key=$1 expected=$2
  jq -e --arg key "${key}" --arg expected "${expected}" \
    '.[$key] == $expected' <<<"${labels}" >/dev/null || {
      printf 'Image label %s does not match %s\n' "${key}" "${expected}" >&2
      exit 1
    }
}
assert_label local-inference.runtime.base-id "${base_image_id}"
assert_label local-inference.vllm.integration.tree "${VLLM_INTEGRATION_TREE}"
assert_label local-inference.b12x.integration.tree "${B12X_INTEGRATION_TREE}"
assert_label local-inference.lmcache.integration.tree "${LMCACHE_INTEGRATION_TREE}"
assert_label local-inference.instanttensor.commit "${instanttensor_commit}"
assert_label local-inference.instanttensor.libaio.repo "${instanttensor_libaio_repo}"
assert_label local-inference.instanttensor.libaio.commit "${instanttensor_libaio_commit}"
assert_label local-inference.instanttensor.libaio.tree "${instanttensor_libaio_tree}"
assert_label local-inference.nccl.version 2.31.2
assert_label local-inference.exllamav3.repo "${exllamav3_repo}"
assert_label local-inference.exllamav3.commit "${exllamav3_commit}"
assert_label local-inference.exllamav3.patch_sha256 "${exllamav3_patch_sha256}"
assert_label local-inference.vllm.spark-overlay.sha256 "${spark_overlay_sha256}"
assert_label local-inference.vllm.spark-overlay.tree "${spark_overlay_tree}"

podman run --rm --entrypoint /opt/venv/bin/python "${image}" \
  /opt/local-inference/verify_deepseek_infernal_cu133_runtime.py \
  --vllm-version "${vllm_package_version}" \
  --flashinfer-version "${flashinfer_version}" \
  --lmcache-version "${lmcache_build_version}" \
  --instanttensor-version 0.1.9

launcher_output="$(
  podman run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
    -e DRY_RUN=1 -e MODE=dspark -e DSPARK_TOKENS=5 -e MAX_NUM_SEQS=16 \
    -e GRAPH=auto -e LOAD_FORMAT=instanttensor "${image}" 2>&1
)"
grep -Fq \
  'Process-group interfaces: GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo' \
  <<<"${launcher_output}"
! grep -q "B12X_DISABLE_STANDALONE_K6" <<<"${launcher_output}" \
  || { echo "forbidden B12X_DISABLE_STANDALONE_K6 in DS4 launcher env" >&2; exit 1; }
printf '%s\n' "${launcher_output}"

glm_launcher_output="$(
  podman run --rm --entrypoint /usr/local/bin/serve-infernal-invocation.sh \
    -e DRY_RUN=1 -e MODEL_FAMILY=glm52 -e TP=8 -e DCP=1 -e MTP=0 \
    "${image}" 2>&1
)"
grep -Fq -- '--attention-backend B12X_MLA_SPARSE' <<<"${glm_launcher_output}"
grep -Fq -- '--tensor-parallel-size 8' <<<"${glm_launcher_output}"
grep -Fq -- '--decode-context-parallel-size 1' <<<"${glm_launcher_output}"
! grep -q "B12X_DISABLE_STANDALONE_K6" <<<"${glm_launcher_output}" \
  || { echo "forbidden B12X_DISABLE_STANDALONE_K6 in GLM launcher env" >&2; exit 1; }

podman run --rm --entrypoint /opt/venv/bin/python "${image}" -c \
  'import importlib, os, pathlib, torch; ext = importlib.import_module("exllamav3_ext"); assert hasattr(ext, "exl3_gemm"); assert pathlib.Path(os.environ["VLLM_EXL3_ENCODER_SOURCE"], "modules/quant/exl3_lib/quantize.py").is_file()'

podman run --rm --device nvidia.com/gpu=all \
  -v "$(pwd)/tests/verify_exl3_codebook_oracle.py:/tmp/gate-oracle.py:ro" \
  --entrypoint /opt/venv/bin/python "${image}" /tmp/gate-oracle.py

podman run --rm --device nvidia.com/gpu=all \
  -v "$(pwd)/tests/verify_exl3_gemm_parity.py:/tmp/gate-gemm.py:ro" \
  --entrypoint /opt/venv/bin/python "${image}" /tmp/gate-gemm.py

# Queue contract (#45224 native in the r17 base + our 0.002 overlay),
# verified behaviorally in-image, never by source markers alone.
echo "f2c2edda497f076d9349fa560213d141f0646a19e359423c3a3dabb70574ac9d  tests/upstream/test_shm_broadcast_45224.py" | sha256sum -c -
podman run --rm \
  -v "$(pwd)/tests/upstream/test_shm_broadcast_45224.py:/tmp/test_shm_45224.py:ro" \
  --entrypoint /opt/venv/bin/python "${image}" \
  -m pytest -q -p no:cacheprovider /tmp/test_shm_45224.py \
  -k "test_reader_timeout_caps_indefinite_waits or test_reader_rechecks_shm_after_idle_wait_timeout_without_notify or test_acquire_read_releases_slot_when_reader_raises"
podman run --rm --entrypoint /opt/venv/bin/python "${image}" -c '
import inspect
from vllm.distributed.device_communicators.shm_broadcast import SpinCondition
d = inspect.signature(SpinCondition.__init__).parameters["busy_loop_s"].default
assert d == 0.002, f"busy_loop_s default {d} != 0.002"
print("SpinCondition busy_loop_s default:", d)'

# Incompatible runtime setting: must not be baked into the image env.
podman run --rm --entrypoint /opt/venv/bin/python "${image}" -c '
import os
assert "B12X_DISABLE_STANDALONE_K6" not in os.environ, "forbidden env baked into image"
print("B12X_DISABLE_STANDALONE_K6: absent from image env")'
printf 'r18 queue + env hardening gates: PASS\n'

# GLM EXL3 production-workload dry run (the profile the window qualifies):
# MODEL_FAMILY=glm52-exl3 TP4/DCP2/MTP3 through the in-image launcher.
glm_exl3_output="$(
  podman run --rm --entrypoint /usr/local/bin/serve-infernal-invocation.sh \
    -e DRY_RUN=1 -e MODEL_FAMILY=glm52-exl3 -e TP=4 -e DCP=2 -e MTP=3 \
    "${image}" 2>&1
)"
grep -Fq -- '--tensor-parallel-size 4' <<<"${glm_exl3_output}"
grep -Fq -- '--decode-context-parallel-size 2' <<<"${glm_exl3_output}"
grep -Fq -- 'num_speculative_tokens' <<<"${glm_exl3_output}"
grep -Eq -- '--kv-cache-dtype [a-z0-9_]+' <<<"${glm_exl3_output}"
! grep -q "B12X_DISABLE_STANDALONE_K6" <<<"${glm_exl3_output}" \
  || { echo "forbidden env in glm52-exl3 launcher" >&2; exit 1; }

# Spark GLM runner contract (FAIL-CLOSED): the II r18 runner file is
# REQUIRED; its render must be nonempty, carry the r18 arguments and exact
# four-channel plus implicit-ordering RoCE contract, and contain neither image-owned NCCL-library /
# cache-identity overrides nor any of the four GG EXL3 tuning variables.
SPARK_GLM_RUNNER="${SPARK_GLM_RUNNER:-run-glm52-r18-exl3-tp4-node.sh}"
[[ -f "${SPARK_GLM_RUNNER}" ]] || { echo "SPARK_GLM_RUNNER missing: ${SPARK_GLM_RUNNER}" >&2; exit 1; }
runner_render="$(DRY_RUN=1 NODE_RANK=0 bash "${SPARK_GLM_RUNNER}" 2>/dev/null)"
[[ -n "${runner_render}" ]] || { echo "Spark GLM runner rendered nothing" >&2; exit 1; }
grep -Fq -- 'DRY-RUN: podman run' <<<"${runner_render}" || { echo "runner render missing DRY-RUN marker" >&2; exit 1; }
grep -Fq -- '--tensor-parallel-size 4' <<<"${runner_render}" || { echo "runner render missing TP4" >&2; exit 1; }
grep -Fq -- '-e NCCL_MIN_NCHANNELS=4' <<<"${runner_render}" || { echo "runner render missing NCCL_MIN_NCHANNELS=4" >&2; exit 1; }
grep -Fq -- '-e NCCL_MAX_NCHANNELS=4' <<<"${runner_render}" || { echo "runner render missing NCCL_MAX_NCHANNELS=4" >&2; exit 1; }
# The failed window-5 arm combined implicit ordering with automatic 64-channel
# selection and therefore did not isolate ordering. Native stacks from the
# exact reproducer place the hang in NCCL's concurrent-communicator ordering /
# resource class. The source-correct candidate keeps the qualified four-channel
# bound and adds ordering; that exact combination passes patterned validation
# on both ranks. Require the literal value so empty/zero cannot ship silently.
grep -Fq -- '-e NCCL_LAUNCH_ORDER_IMPLICIT=1' <<<"${runner_render}" || {
  echo "runner render missing NCCL_LAUNCH_ORDER_IMPLICIT=1" >&2; exit 1
}
if grep -qE -- '-e (LD_PRELOAD|VLLM_NCCL_SO_PATH|XDG_CACHE_HOME|VLLM_CACHE_DIR|TRITON_CACHE_DIR|TORCHINDUCTOR_CACHE_DIR|TORCH_EXTENSIONS_DIR|FLASHINFER_WORKSPACE_BASE|TVM_CACHE_DIR|TILELANG_CACHE_DIR)=' <<<"${runner_render}"; then
  echo "Spark GLM runner renders forbidden NCCL-library/cache-identity overrides" >&2; exit 1
fi
if grep -qE -- '-e VLLM_EXL3_(TRELLIS_MAX_M|TRELLIS_BLOCK_M|PREFILL_TRELLIS|PREFILL_CHUNK)=' <<<"${runner_render}"; then
  echo "Spark GLM runner renders GG EXL3 tuning variables (forbidden on II)" >&2; exit 1
fi
printf 'Spark GLM runner render: fail-closed checks PASS\n'
# P1 (2026-08-18): the unoverridden runner render must reference EXACTLY
# this build's image - a stale default silently boots the wrong release.
grep -Fq -- "${image}" <<<"${runner_render}" || {
  echo "Spark GLM runner default image is not this build's image" >&2; exit 1
}


# In-image launcher: the four GG tuning variables must be ABSENT from the
# glm52-exl3 render (requires the respun serve-gilded-gnosis.sh).
if grep -qE 'VLLM_EXL3_(TRELLIS_MAX_M|TRELLIS_BLOCK_M|PREFILL_TRELLIS|PREFILL_CHUNK)' <<<"${glm_exl3_output}"; then
  echo "in-image launcher still exports GG EXL3 tuning variables" >&2; exit 1
fi

# Clean image-series break: the GG unified dispatcher must NOT exist in
# II images, and the II dispatcher itself must carry no GG tuning vars.
podman run --rm --entrypoint /opt/venv/bin/python "${image}" -c '
import os
assert not os.path.exists("/usr/local/bin/serve-gilded-gnosis.sh"), "GG launcher present in II image"
s = open("/usr/local/bin/serve-infernal-invocation.sh").read()
for v in ("VLLM_EXL3_TRELLIS_MAX_M", "VLLM_EXL3_TRELLIS_BLOCK_M", "VLLM_EXL3_PREFILL_TRELLIS", "VLLM_EXL3_PREFILL_CHUNK"):
    assert v not in s, f"GG tuning var {v} in II dispatcher"
print("II image-series break: GG launcher absent, II dispatcher clean")'
printf 'image-series break gates: PASS\n'
printf 'glm52-exl3 dry-run gates: PASS\n'


# Encoder provenance: the runtime revision must carry the patched identity.
image_encoder_rev="$(podman run --rm --entrypoint /opt/venv/bin/python "${image}" \
  -c 'import os; print(os.environ["VLLM_EXL3_ENCODER_REVISION"])')"
test "${image_encoder_rev}" = "${exllamav3_commit}+p.${exllamav3_patch_sha256}" || {
  printf 'encoder revision mismatch: %s\n' "${image_encoder_rev}" >&2
  exit 1
}

# Online-K6 OPEN gate: no unsafe override exists anywhere, and online
# trellis bits resolve without one (replaces the removed r10 override
# matrix; retraction 2026-08-15).
podman run --rm -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 \
  --entrypoint /opt/venv/bin/python "${image}" -c '
import vllm.envs as _envs
assert "VLLM_EXL3_ONLINE_UNSAFE_OVERRIDE" not in _envs.environment_variables
assert "VLLM_EXL3_ONLINE_UNSAFE_OVERRIDE" not in dir(_envs)
from vllm.model_executor.layers.quantization.exl3 import _online_trellis_bits
assert _online_trellis_bits() == 6, "online K6 must be open with no override"'

# Online K6 encoder/ExLlama execution-contract gate (production call
# convention, meta-Hessian fallback, cache-hit byte identity, b12x
# r17 stack-support assert + incompatible-env hardening).
podman run --rm --device nvidia.com/gpu=all \
  -v "$(pwd)/tests/verify_exl3_online_k6_pipeline.py:/tmp/gate-pipeline.py:ro" \
  --entrypoint /opt/venv/bin/python "${image}" /tmp/gate-pipeline.py

# B12X scratch planner matrix (r17): query-based rows<=16, padded rows>=17,
# capability-independent; legacy <=128->1 only without the query; real query
# must exist and reject non-%128 (source-level, capability mocked).
podman run --rm --device nvidia.com/gpu=all \
  -v "$(pwd)/tests/verify_b12x_scratch_planner_matrix.py:/tmp/gate-scratch.py:ro" \
  --entrypoint /opt/venv/bin/python "${image}" /tmp/gate-scratch.py

# SM121 dense-Trellis CUDA-graph gate (r17 #221 namespace): proves the
# CuTe K6/MCG small-M launch BINDS and DISPATCHES on SM121 (observed via
# an instrumented run_k6_mcg_small_m), and captures + replays at
# M={4,16,17,32} on both GLM TP4 shared-expert shapes - 16/17 straddle
# the small/generic dispatch boundary.
podman run --rm --device nvidia.com/gpu=all \
  -v "$(pwd)/tests/verify_b12x_trellis_graph_capture.py:/tmp/gate-graph.py:ro" \
  --entrypoint /opt/venv/bin/python "${image}" /tmp/gate-graph.py

printf 'SM121 EXL3 gate battery: PASS\n'

if [[ "${RUN_NCCL_SMOKE:-0}" == 1 ]]; then
  smoke_ranks=${NCCL_SMOKE_RANKS:-4}
  podman run --rm --device nvidia.com/gpu=all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --entrypoint torchrun "${image}" \
    --standalone --nproc-per-node="${smoke_ranks}" \
    /opt/local-inference/torch_nccl_smoke.py
fi

if [[ "${PUSH_IMAGE:-0}" == 1 ]]; then podman push "${image}"; fi

podman image inspect "${image}" --format \
  'image={{.Id}} size={{.Size}} entrypoint={{json .Config.Entrypoint}}'
printf '%s\n' "${image}"
