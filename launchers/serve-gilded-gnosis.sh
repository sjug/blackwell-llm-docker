#!/usr/bin/env bash
set -euo pipefail

# The unified image targets a single PCIe host. Multi-node deployments can
# override both interfaces explicitly.
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
printf 'Process-group interfaces: GLOO_SOCKET_IFNAME=%s NCCL_SOCKET_IFNAME=%s\n' \
  "${GLOO_SOCKET_IFNAME}" "${NCCL_SOCKET_IFNAME}"

model_command=()
glm52_server="${GLM52_SERVER:-/usr/local/bin/serve-glm52-v19.sh}"
case "${MODEL_FAMILY:-}" in
  glm52|glm5.2|glm)
    model_command=("${glm52_server}" "$@")
    ;;
  glm52-hybrid|nf3)
    model_command=(/usr/local/bin/serve-glm52-hybrid-v19.sh "$@")
    ;;
  glm52-exl3|exl3)
    export MODEL="${MODEL:-brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw}"
    export MODEL_REVISION="${MODEL_REVISION:-9297b9f1d53af5c67cffa01e30cc071a1ff7144b}"
    export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2-EXL3-TR3-3.0bpw}"
    export TP="${TP:-4}"
    export DCP="${DCP:-4}"
    export MTP="${MTP:-3}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
    export GRAPH="${GRAPH:-32}"
    export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-3072}"
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
    export MOE_MODE="${MOE_MODE:-a16}"
    export MOE_BACKEND="${MOE_BACKEND:-b12x}"
    export MTP_MOE_BACKEND="${MTP_MOE_BACKEND:-triton}"
    export MTP_DRAFT_SAMPLE_METHOD="${MTP_DRAFT_SAMPLE_METHOD:-greedy}"
    export QUANTIZATION="${QUANTIZATION:-exl3}"
    export ONLINE_QUANT="${ONLINE_QUANT:-none}"
    export LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
    export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4_ds_mla}"
    export ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
    export DCP_KV_CACHE_INTERLEAVE_SIZE="${DCP_KV_CACHE_INTERLEAVE_SIZE:-1}"
    export F8_DMA="${F8_DMA:-0}"
    # II r17-spark (2026-08-18): the four GG EXL3 tuning exports are
    # REMOVED unconditionally - that set collapsed II mixed-Trellis
    # prefill to 169 tok/s at r10. GG images keep them in their own
    # launcher; II must never apply GG tuning (no profile switch).
    model_command=("${glm52_server}" "$@")
    ;;
  ds4|ds4-flash|dspark)
    model_command=(/usr/local/bin/serve-ds4-flash.sh "$@")
    ;;
  *)
    echo "ERROR: MODEL_FAMILY must be glm52, glm52-hybrid, glm52-exl3, or ds4" >&2
    exit 2
    ;;
esac

lmcache_mode="${LMCACHE_MODE:-off}"
lmcache_mode="${lmcache_mode,,}"
if [[ "${lmcache_mode}" == "off" || "${lmcache_mode}" == "0" ]]; then
  exec "${model_command[@]}"
fi

case "${MODEL_FAMILY:-}" in
  glm52|glm5.2|glm|glm52-hybrid|nf3|glm52-exl3|exl3)
    ;;
  *)
    echo "ERROR: LMCache is validated only for GLM-5.2 MLA model families" >&2
    exit 2
    ;;
esac

lmcache_wrapper="${LMCACHE_MP_WRAPPER:-${GLM52_LMCACHE_WRAPPER:-/usr/local/bin/lmcache-mp-wrapper.sh}}"
exec "${lmcache_wrapper}" "${model_command[@]}"
