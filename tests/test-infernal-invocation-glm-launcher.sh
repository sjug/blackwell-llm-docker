#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher="${repo_root}/launchers/serve-infernal-invocation.sh"

bash -n "${launcher}"
grep -Fq \
  'GLM52_SERVER="${GLM52_SERVER:-/usr/local/bin/serve-glm52-v19.sh}"' \
  "${launcher}"
grep -Fq 'model_command=("${glm52_server}" "$@")' "${launcher}"
grep -Fq 'model_command=(/usr/local/bin/serve-ds4-flash.sh "$@")' "${launcher}"
! grep -Fq 'exec /usr/local/bin/serve-gilded-gnosis.sh' "${launcher}"
! grep -Eq 'VLLM_EXL3_(TRELLIS_MAX_M|TRELLIS_BLOCK_M|PREFILL_TRELLIS|PREFILL_CHUNK)' \
  "${launcher}"

for required in \
  serve-infernal-invocation.sh \
  serve-glm52-v16.sh \
  serve-glm52-v19.sh \
  serve-glm52-hybrid-v19.sh \
  glm52-dcp-prefill-policy.sh \
  glm52-pcie-runtime-env.sh \
  glm52-pcie-calibration.py \
  lmcache-mp-wrapper.sh; do
  test -f "${repo_root}/launchers/${required}"
done

echo 'Infernal Invocation GLM launcher contract: PASS'
