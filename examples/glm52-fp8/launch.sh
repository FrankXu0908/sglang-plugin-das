#!/usr/bin/env bash
set -Eeuo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the local GLM5.2 Channel FP8 checkpoint}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/runtime.env"
export HCU_ADAPTER_HOST=sglang SGLANG_PLUGINS=hcu
unset _HCU_ADAPTER_ACTIVE_HOST
if [ -n "${ADAPTER_ROOT:-}" ]; then export PYTHONPATH="$ADAPTER_ROOT:${PYTHONPATH:-}"; fi
if [ -n "${PLUGIN_ROOT:-}" ]; then export PYTHONPATH="$PLUGIN_ROOT:${PYTHONPATH:-}"; fi
exec python3 -m hcu_sglang \
 --model-path "$MODEL_PATH" --trust-remote-code --tp-size 8 --attn-cp-size 8 \
 --dsa-prefill-backend flashmla_auto --dsa-decode-backend flashmla_kv \
 --dtype bfloat16 --dist-timeout 10000 --watchdog-timeout 3600 \
 --page-size 64 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.8 \
 --chunked-prefill-size 8192 --cuda-graph-max-bs 16 --max-running-requests 16 \
 --reasoning-parser glm45 --tool-call-parser glm47 \
 --speculative-algorithm EAGLE --speculative-num-steps 3 \
 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
 --ep-size 8 --moe-a2a-backend deepep --moe-dense-tp-size 1 \
 --enable-dp-lm-head --enable-dp-attention --deepep-mode auto \
 --host "${HOST:-127.0.0.1}" --port "${PORT:-31022}" --random-seed 42 \
 --served-model-name "${SERVED_MODEL_NAME:-glm52-fp8-adapter-plugin}" "$@"
