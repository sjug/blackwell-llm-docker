#!/usr/bin/env bash
# II r18 GLM runner: derived from run-glm52-exl3-tp4-node.sh but
# UNCONDITIONALLY omits the four GG EXL3 tuning variables
# (VLLM_EXL3_TRELLIS_MAX_M / TRELLIS_BLOCK_M / PREFILL_TRELLIS /
# PREFILL_CHUNK) - that exact set collapsed II prefill to 169 tok/s at
# r10. No profile switch by design: GG images use the GG runner, II
# images use THIS file, and the r18 build gates assert the four are
# absent from both the in-image launcher and this runner's render.
set -euo pipefail

# GLM-5.2 v18.4 Grid48 hybrid four-node TP4 launcher for the DGX Spark
# cluster (sparky + buddy + rocky + lucky, one 48-SM GB10 per node).
#
# Serves madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid per models/glm5.2_v17.md.
# Unlike the DS4 v16 launcher, this REPLICATES the vllm serve command
# (v10-launcher style) instead of delegating to the in-image helper:
# serve-glm52-hybrid-v18.sh -> serve-glm52-v18.sh -> serve-glm52-v16.sh have no EXTRA_VLLM_ARGS
# escape hatch and unconditionally export single-node x86 values
# (CUTE_DSL_ARCH=sm_120a, NCCL_IB_DISABLE=1, VLLM_ENABLE_PCIE_ALLREDUCE=1)
# that would override anything passed with -e. Env block and argv below are
# a faithful copy of the v18 launcher chain (blackwell-llm-docker @ 7f3cbc6) with the hybrid
# preset applied; keep them in sync if the pin moves. Upstream candidate:
# add EXTRA_VLLM_ARGS + default-if-unset env to serve-glm52-v16.sh.
#
# Deviations from the x86 v17 reference, limited to SM121 / GB10 / 4-node:
#   1. podman rootless + CDI (--device nvidia.com/gpu=all), /dev/infiniband
#      for RoCE; no --shm-size with --ipc host; nofile capped at 500000.
#   2. TP4 across four nodes via --nnodes/--node-rank/--master-addr/
#      --headless (torch-distributed, no Ray): one GB10 (121 GB) cannot
#      hold the ~TP4 checkpoint shard plus peers.
#   3. DCP2 default (production since 2026-07-28): v20's ag_rs transport
#      carries DCP cross-node over PYNCCL/RoCE. DCP=1 remains selectable.
#   4. NCCL over the two SWITCHED 200 GbE CX7 rails (helper sets
#      NCCL_IB_DISABLE=1): IB on, HCA list, RoCE-v2 GID auto-detect,
#      TC 106, socket ifnames, per-node VLLM_HOST_IP. NCCL_PROTO=LL,Simple
#      (LL128 uncertified off NVLink-class links). NCCL_P2P_LEVEL dropped
#      (no intra-node GPU pairs). The f1 back-to-back rails are excluded —
#      only pairwise-reachable (see the networking block below).
#   5. CUTE_DSL_ARCH=sm_121a, TORCH_CUDA_ARCH_LIST=12.1a (helper hardcodes
#      sm_120a/12.0a; SM121 refuses SM120 cubins).
#   6. VLLM_ENABLE_PCIE_ALLREDUCE=0 and VLLM_USE_B12X_PCIE_DMA=0 (helper:
#      1/1) — no PCIe peers exist across nodes.
#   7. GPU_MEMORY_UTILIZATION=0.86 (helper 0.90, hybrid preset 0.96 —
#      dedicated-VRAM assumptions; on 121 GB UMA, 0.90 starves the host).
#   8. MAX_MODEL_LEN=262144 (matches the hybrid preset; validated on this
#      cluster since v17). MAX_NUM_SEQS=8 and GRAPH=seqs*4 follow the
#      helper's own derivation. NOTE: the bench's 128k decode cells 400
#      whenever max_model_len < ~135k — never lower this for bring-up
#      without expecting that row to fail.
#   9. MTP default 3 — the production mode since v17-era validation
#      (2026-07-15) and the v18p4/v20 deployments. Defaults here ARE the
#      production profile (was: bring-up values requiring a magic env
#      string from the deploy script — config drift hazard).
#  10. The mapped-grid gate is hardware-neutral: exact SM121/48-SM devices
#      select Grid48. The legacy Grid188 variable is exported as a launcher
#      compatibility alias; it does not select an SM120 kernel on GB10.
#  11. NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=4. The II NCCL 2.31.2 build
#      otherwise selects 64 channels for each DCP2 communicator. GLM overlaps
#      its full-CKV prefetch all-gather with the sparse-indexer candidate
#      all-gather; the two 64-channel communicators hang on the Spark RoCE
#      topology. The exact production-shaped overlap passes at four channels,
#      which was also the fastest bounded channel count tested.
#  12. NCCL_LAUNCH_ORDER_IMPLICIT=1. The two collectives above use different
#      communicators and streams. Native stacks exclude proxy-pool exhaustion
#      and place the hang in NCCL's documented concurrent-communicator launch-
#      ordering/resource class. Implicit ordering plus the existing exact-four
#      cap passes the patterned-output reproducer with both ranks exiting zero.
#      Exact-four remains in place because ordering at automatic 64 channels
#      does not bound the separate GB10 UMA/resource-pressure exposure.
#      This combined setting is a staging candidate, not yet a production-
#      qualified replacement for exact-four with implicit ordering unset.
#
# The checkpoint (revision 68babde) must be staged in the HF cache on ALL
# four nodes before launch (HF_HUB_OFFLINE=1), or pass MODEL=/path.
#
# Usage (workers first, head last; same command everywhere — the rank is
# derived from the hostname, MASTER_ADDR defaults to sparky's switch IP):
#   buddy : ./run-glm52-v18.4-grid48-hybrid-tp4-node.sh
#   rocky : ./run-glm52-v18.4-grid48-hybrid-tp4-node.sh
#   lucky : ./run-glm52-v18.4-grid48-hybrid-tp4-node.sh
#   sparky: ./run-glm52-v18.4-grid48-hybrid-tp4-node.sh
#   endpoint: sparky:8000, served model GLM-5.2-MXFP8-NVFP4-NF3-Hybrid

# v20p1 port note (2026-07-23): image = upstream v20 release pins + our three
# fixes baked (see spark/v20/). The fork-only VLLM_NF3_MAPPED_GRID_DECODE env
# below is retained for compatibility but upstream 2167295 does not read it;
# upstream's hybrid decode was reworked (constexpr tiers) — decode perf vs the
# v18p4 grid48 fork kernel is TBD by benchmark. DCP2 cross-node validated
# 2026-07-23/24 on this cluster (ag_rs over PYNCCL/RoCE) and default since
# 2026-07-28; DCP4 works too (KV 5.48x) but costs -11% decode.
# r28-spark (2026-08-04): upstream r28 composition + SM121 overlay. Qualified
# on this cluster 2026-08-04: TP4/DCP2/MTP3 boot with PYNCCL/RoCE ag_rs,
# decode 24.9 tok/s cc1, uncached prefill flat 766-779 tok/s at 8k/64k and
# 771 tok/s at 128k (154,631 tokens), zero errors on all four nodes. The
# upstream r26 TP4/DCP4 auto prefill policy ships in the image's baked GLM
# helpers, but this runner bypasses the helper suite entirely (direct vllm
# serve, --entrypoint /bin/bash): every DCP mechanism is set explicitly
# below and the upstream PCIe calibrator NEVER runs on this cluster; its
# same-host PCIe conclusions do not transfer to our cross-node RoCE DCP.
# CKV/query-split/workspace/prefetch get their own RoCE A/B (queued).
# Rollback: gilded-gnosis-v20p1-...-20260722 (previous production).
IMAGE=${IMAGE:-localhost/voipmonitor/vllm:infernal-invocation-r18-spark-sm121-vllmf560085-b12x75787c7-fi1ac6942-cu133-torch213-20260818}
NAME=${NAME:-glm52-exl3-tp4}
PORT=${PORT:-8000}
NNODES=${NNODES:-4}

if [[ -z "${NODE_RANK:-}" ]]; then
  case "$(hostname -s)" in
    sparky) NODE_RANK=0 ;;
    buddy)  NODE_RANK=1 ;;
    rocky)  NODE_RANK=2 ;;
    lucky)  NODE_RANK=3 ;;
    *) echo "Unknown host $(hostname -s): set NODE_RANK=0..$((NNODES - 1)) (0 = head)" >&2; exit 2 ;;
  esac
fi
HEADLESS_ARGS=()
if [[ "$NODE_RANK" != "0" ]]; then HEADLESS_ARGS=(--headless); fi

MASTER_ADDR=${MASTER_ADDR:-10.11.11.1}
MASTER_PORT=${MASTER_PORT:-25000}

# FP8 KV + DCP2 is the quality-first serving default (adopted upstream KLD:
# 3.42bpw checkpoint 0.074 @fp8 KV vs 0.108 @nvfp4 KV; FP4 cache costs ~46%
# relative). Capacity is recovered via cross-node DCP2 instead of KV
# quantization. KV_CACHE_DTYPE=nvfp4_ds_mla + DCP=1 remain selectable.
DCP=${DCP:-2}
# v18-era hard block removed: the IPC-only DCP pool could not cross nodes,
# but v20's a2a transport has an ag_rs (NCCL-collective) large-message
# backend that carries cross-node over PYNCCL/RoCE. DCP2 is the production
# default since 2026-07-28: KV 739,200 (2.82x, five 128k sessions vs two),
# decode within noise of DCP1, prefill -4-5% (full decision table in the
# v20 rollout notes). DCP4 validated but decode -11%. The upstream
# TP4/DCP>=2 dispatch enables query-split + transient full-CKV gather,
# mirrored below. DCP_A2A_MAX_TOKENS=0 forces every message onto ag_rs if
# the small-message pool cannot initialize without IPC peers.
DCP_QUERY_SPLIT_V=0
DCP_CKV_GATHER_V=0
if [[ "$DCP" != "1" ]]; then
  DCP_QUERY_SPLIT_V=1
  DCP_CKV_GATHER_V=1
fi

# --- Cluster networking (deviations 2 & 4) ---
# Surveyed 2026-07-14: each node has two SWITCHED 200G rails on the f0
# ports — 10.11.11.{1,2,3,4} on enp1s0f0np0 (rocep1s0f0) and
# 10.11.12.{1,2,3,4} on enP2p1s0f0np0 (roceP2p1s0f0); node order
# sparky/buddy/lucky/rocky. The f1 ports are BACK-TO-BACK pairs
# (sparky<->buddy, lucky<->rocky on 10.11.1.x/10.11.2.x) and are NOT
# all-to-all — never put them in NCCL_IB_HCA for 4-node TP.
NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f0,roceP2p1s0f0}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f0np0,enP2p1s0f0np0}
CONTROL_IF=${CONTROL_IF:-enp1s0f0np0}
NCCL_IB_TC=${NCCL_IB_TC:-106}

VLLM_HOST_IP=${VLLM_HOST_IP:-$(ip -4 -o addr show dev "$CONTROL_IF" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)}
if [[ -z "$VLLM_HOST_IP" ]]; then
  echo "Could not derive VLLM_HOST_IP from $CONTROL_IF; set VLLM_HOST_IP or CONTROL_IF" >&2
  exit 2
fi

NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-}
if [[ -z "$NCCL_IB_GID_INDEX" ]]; then
  first_hca=${NCCL_IB_HCA%%,*}
  for i in $(seq 0 15); do
    t=$(cat "/sys/class/infiniband/$first_hca/ports/1/gid_attrs/types/$i" 2>/dev/null || true)
    g=$(cat "/sys/class/infiniband/$first_hca/ports/1/gids/$i" 2>/dev/null || true)
    if [[ "$t" == *"RoCE v2"* && "$g" == *0000:0000:0000:0000:0000:ffff:* ]]; then
      NCCL_IB_GID_INDEX=$i
      break
    fi
  done
fi

# --- Model (v17 pinned revision), resolved from the local HF cache ---
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
HYBRID_REPO_DIR=models--willfalco--GLM-5.2-EXL3-TR3-3.42bpw
SNAPSHOT=${SNAPSHOT:-ae68c65947efa90bea37308e15421872f124c46d}
MODEL=${MODEL:-/root/.cache/huggingface/hub/$HYBRID_REPO_DIR/snapshots/$SNAPSHOT}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-GLM-5.2-EXL3-TR3-3.42bpw}
if [[ "${DRY_RUN:-0}" != "1" ]] && [[ "$MODEL" == /root/.cache/huggingface/* ]] \
  && ! test -f "$HF_CACHE/hub/$HYBRID_REPO_DIR/snapshots/$SNAPSHOT/config.json"; then
  echo "Checkpoint not staged: $HF_CACHE/hub/$HYBRID_REPO_DIR/snapshots/$SNAPSHOT" >&2
  echo "Stage willfalco/GLM-5.2-EXL3-TR3-3.42bpw @ $SNAPSHOT on all $NNODES nodes (or pass MODEL=/path)." >&2
  exit 2
fi

CACHE=${CACHE:-$HOME/.cache/vllm-glm52-exl3}
CONTAINER_TMP=${CONTAINER_TMP:-$CACHE/tmp}

# --- Serving profile (helper defaults + hybrid preset + deviations 7-9) ---
TP=${TP:-4}
MTP=${MTP:-3}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
GRAPH=${GRAPH:-$((MAX_NUM_SEQS * 4))}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-3072}
GPU_MEM=${GPU_MEM:-0.86}
# 78-char pattern, verbatim from the helper; the serving contract depends on
# the exact string.
INDEX_TOPK_PATTERN=${INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}
[[ "${#INDEX_TOPK_PATTERN}" -eq 78 ]] || { echo "INDEX_TOPK_PATTERN must be 78 chars" >&2; exit 2; }

# TP4:DCP1 resolves the helper's DCP_PREFILL_WORKSPACE=auto gate to 0.
DCP_PROJECT_MIN_PREFILL_TOKENS=1024
if ((GRAPH > DCP_PROJECT_MIN_PREFILL_TOKENS)); then
  DCP_PROJECT_MIN_PREFILL_TOKENS=$GRAPH
fi

SPEC_ARGS=()
if [[ "$MTP" != "0" ]]; then
  SPEC_JSON=$(printf '{"model":"%s","method":"mtp","num_speculative_tokens":%s,"moe_backend":"triton","draft_sample_method":"greedy"}' "$MODEL" "$MTP")
  SPEC_ARGS=(--speculative-config "$SPEC_JSON")
fi

HF_OVERRIDES=$(printf '{"use_index_cache":true,"index_topk_pattern":"%s"}' "$INDEX_TOPK_PATTERN")
# EXL3 capture ladder: multiples of (MTP+1) up to GRAPH, FULL_AND_PIECEWISE
# (verbatim helper derivation for QUANTIZATION=exl3).
graph_step=$((MTP + 1))
graph_sizes=""
for ((capture_size = graph_step; capture_size <= GRAPH; capture_size += graph_step)); do
  [[ -n "$graph_sizes" ]] && graph_sizes+=,
  graph_sizes+="$capture_size"
done
[[ -n "$graph_sizes" ]] || graph_sizes="$GRAPH"
COMPILATION_CONFIG_JSON=$(printf '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[%s],"custom_ops":["all"],"pass_config":{"fuse_allreduce_rms":true}}' "$graph_sizes")

# DRY_RUN=1 is SIDE-EFFECT-FREE (2026-08-18): no directory creation and
# no teardown of a live container - render only.
if [ "${DRY_RUN:-0}" != "1" ]; then
  mkdir -p \
    "$CACHE/vllm" "$CACHE/tilelang/tmp" "$CACHE/tvm" "$CACHE/triton" \
    "$CACHE/torchinductor" "$CACHE/torch_extensions" "$CACHE/flashinfer" \
    "$CONTAINER_TMP"

  # Graceful teardown of a stale instance: never rm -f a running server.
  if podman container exists "$NAME" 2>/dev/null; then
    podman stop -t 60 "$NAME" >/dev/null 2>&1 || true
    podman rm "$NAME" >/dev/null 2>&1 || true
  fi
fi

# DRY_RUN=1 (2026-08-18): render the full podman invocation without
# executing - used by build gates to assert the runner does not replace the
# image-owned NCCL library or cache identity. NCCL channel count is a
# topology/workload runtime contract and is intentionally set by this runner.
run_cmd=(podman run -d)
if [ "${DRY_RUN:-0}" = "1" ]; then run_cmd=(echo DRY-RUN: podman run -d); fi
"${run_cmd[@]}" \
  --name "$NAME" \
  --device nvidia.com/gpu=all \
  --device /dev/infiniband \
  --security-opt label=disable \
  --ipc host \
  --network host \
  --init \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --ulimit nofile=500000:500000 \
  -v "$HF_CACHE:/root/.cache/huggingface:ro" \
  -v "$CACHE:/cache:rw" \
  -v "$CONTAINER_TMP:/container-tmp:rw" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e CUTE_DSL_ARCH=sm_121a \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e SAFETENSORS_FAST_GPU=1 \
  -e INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-BUFFERED}" \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_WO_PROJECTION=1 \
  -e VLLM_USE_B12X_MHC=1 \
  -e VLLM_EXL3_EXT_PATH=/opt/exllamav3 \
  -e VLLM_DCP_QUERY_SPLIT="$DCP_QUERY_SPLIT_V" \
  -e VLLM_B12X_MLA_CKV_GATHER="$DCP_CKV_GATHER_V" \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_USE_B12X_DCP_A2A=1 \
  -e VLLM_DCP_A2A_MAX_TOKENS="${DCP_A2A_MAX_TOKENS:-16}" \
  -e VLLM_DCP_A2A_LARGE_BACKEND=ag_rs \
  -e VLLM_DCP_PROJECT_BEFORE_MERGE=0 \
  -e VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS="$DCP_PROJECT_MIN_PREFILL_TOKENS" \
  -e VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e VLLM_USE_B12X_PCIE_DMA=0 \
  -e VLLM_PCIE_DMA_FP8=0 \
  -e B12X_PCIE_DMA_FP8=0 \
  -e VLLM_DCP_GLOBAL_TOPK=1 \
  -e VLLM_DCP_SHARD_DRAFT=1 \
  -e B12X_MLA_SM120_UNIFIED=1 \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_W4A8_TINY_DECODE=1 \
  -e B12X_MOE_FORCE_A8=0 \
  -e B12X_MOE_FORCE_A16=1 \
  -e NCCL_PROTO=LL,Simple \
  -e NCCL_LAUNCH_ORDER_IMPLICIT=1 \
  -e NCCL_MIN_NCHANNELS=4 \
  -e NCCL_MAX_NCHANNELS=4 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$NCCL_IB_HCA" \
  ${NCCL_IB_GID_INDEX:+-e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX"} \
  -e NCCL_IB_TC="$NCCL_IB_TC" \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  -e GLOO_SOCKET_IFNAME="$CONTROL_IF" \
  -e VLLM_HOST_IP="$VLLM_HOST_IP" \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  -e TMPDIR=/container-tmp \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
    # NCCL library and cache identity are IMAGE-owned (2026-08-18): the runner
    # must not override LD_PRELOAD/VLLM_NCCL_SO_PATH or the fingerprinted cache
    # dirs. Channel count remains topology/workload-owned outside the image.
    if [ -z "${VLLM_NCCL_SO_PATH:-}" ] || [ ! -f "$VLLM_NCCL_SO_PATH" ]; then
      echo "preflight: VLLM_NCCL_SO_PATH invalid: ${VLLM_NCCL_SO_PATH:-unset}" >&2
      exit 78
    fi
    exec vllm serve "$@"' \
  -- "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size "$TP" \
  --decode-context-parallel-size "$DCP" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" \
  --attention-backend B12X_MLA_SPARSE \
  --moe-backend b12x \
  --quantization exl3 \
  --compilation-config "$COMPILATION_CONFIG_JSON" \
  --load-format instanttensor \
  -cc.pass_config.fuse_allreduce_rms=True \
  --gpu-memory-utilization "$GPU_MEM" \
  ${KV_CACHE_MEMORY_BYTES:+--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"} \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --max-cudagraph-capture-size "$GRAPH" \
  --async-scheduling \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
  --enable-prompt-tokens-details \
  --enable-force-include-usage \
  --enable-request-id-headers \
  --hf-overrides "$HF_OVERRIDES" \
  --nnodes "$NNODES" \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  "${HEADLESS_ARGS[@]}" \
  "${SPEC_ARGS[@]}"

echo "$NAME rank=$NODE_RANK/$NNODES master=$MASTER_ADDR host_ip=$VLLM_HOST_IP tp=$TP dcp=$DCP mtp=$MTP exl3=3.42bpw port=$PORT seqs=$MAX_NUM_SEQS graph=$GRAPH len=$MAX_MODEL_LEN gpu_mem=$GPU_MEM gid=${NCCL_IB_GID_INDEX:-auto} nccl_channels=4 nccl_order=implicit"
