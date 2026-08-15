#!/usr/bin/env bash
# Spark/SM121 variant of build-kimi-k3-cu133-torch213-base.sh: same pinned
# sources, arm64 NGC base digest, sm_121 codegen, podman flow, and GB10
# single-GPU smokes (the 16-GPU x86 smoke does not apply; cross-node NCCL
# validation happens in the dedicated RoCE gate before serving quals).
set -euo pipefail

cd "$(dirname "$0")"

nvidia_pytorch_image="${NVIDIA_PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3}"
pytorch_commit="${PYTORCH_COMMIT:-cf30153c4c131c8164ee7798e5022d810682e2cb}"
torchvision_commit="${TORCHVISION_COMMIT:-8fb87713a24951e639c494b0f2a8a81b5f8e33a6}"
nccl_commit="${NCCL_COMMIT:-fb6f40999a2a9e63104d4ae4a84118bce61528f8}"
xgrammar_commit="${XGRAMMAR_COMMIT:-2ea71da4ccb997a06928c9fb69b99f330da56697}"
release_date="${RELEASE_DATE:-$(date -u +%Y%m%d)}"
revision="${REVISION:-r1}"
docker_commit="${DOCKER_COMMIT:-$(git rev-parse HEAD)}"
image="${IMAGE:-localhost/voipmonitor/vllm:ii-spark-sm121-cu133-torch213-nccl2312-${release_date}-${revision}}"

printf 'base=%s\n' "${nvidia_pytorch_image}"
printf 'pytorch=%s\n' "${pytorch_commit}"
printf 'torchvision=%s\n' "${torchvision_commit}"
printf 'nccl=%s\n' "${nccl_commit}"
printf 'xgrammar=%s\n' "${xgrammar_commit}"
printf 'image=%s\n' "${image}"

podman build \
  --build-arg "NVIDIA_PYTORCH_IMAGE=${nvidia_pytorch_image}" \
  --build-arg "PYTORCH_COMMIT=${pytorch_commit}" \
  --build-arg "TORCHVISION_COMMIT=${torchvision_commit}" \
  --build-arg "NCCL_COMMIT=${nccl_commit}" \
  --build-arg "XGRAMMAR_COMMIT=${xgrammar_commit}" \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --file Dockerfile.ii-spark-sm121-cu133-torch213-base \
  --tag "${image}" \
  .

labels="$(podman image inspect "${image}" --format '{{json .Config.Labels}}')"
jq -e --arg value "${pytorch_commit}" \
  '."local-inference.pytorch.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${torchvision_commit}" \
  '."local-inference.torchvision.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${nccl_commit}" \
  '."local-inference.nccl.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${xgrammar_commit}" \
  '."local-inference.xgrammar.commit" == $value' <<<"${labels}" >/dev/null

podman run --rm --device nvidia.com/gpu=all --ipc=host -i \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --entrypoint python "${image}" - <<'PY'
import torch
from torchvision.ops import nms

assert torch.cuda.device_count() == 1, torch.cuda.device_count()
cap = torch.cuda.get_device_capability(0)
assert cap == (12, 1), cap
with torch.cuda.device(0):
    left = torch.arange(256, device="cuda", dtype=torch.float32).reshape(16, 16)
    result = left @ left
    boxes = torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]],
        device="cuda",
    )
    scores = torch.tensor([0.9, 0.8], device="cuda")
    keep = nms(boxes, scores, 0.5)
    torch.cuda.synchronize()
    assert result.shape == (16, 16)
    assert keep.tolist() == [0, 1]
print("II Spark SM121 base single-GPU PyTorch smoke: PASS")
PY

podman run --rm --device nvidia.com/gpu=all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --entrypoint torchrun "${image}" \
  --standalone --nproc-per-node=1 \
  /opt/local-inference/torch_nccl_smoke.py

podman image inspect "${image}" --format \
  'image={{.Id}} size={{.Size}} created={{.Created}}'

printf '%s\n' "${image}"
