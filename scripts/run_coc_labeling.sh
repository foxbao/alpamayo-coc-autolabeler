#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        PYTHON_BIN="$(command -v python || true)"
    fi
fi
DATA_DIR="${DATA_DIR:-/home/baojiali/Downloads/qwen-drive/datasets/pai_reasoning_16}"
RUN_ROOT="${RUN_ROOT:-/home/baojiali/Downloads/qwen-drive/outputs/pai_reasoning_16_official_20261013}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/home/baojiali/.cache/alpamayo-coc-models}"
API_CONFIG="${API_CONFIG:-${HOME}/.config/alpamayo-coc/openai_api.yaml}"
SOURCE_SEGMENT_CONFIG="${RUN_ROOT}/keyframes/segments_relative_timestamp_sampled.json"
SEGMENT_CONFIG_PATH="$SOURCE_SEGMENT_CONFIG"
LIMIT=""

MODEL_NAME="qwen3.5_35b"
EXP_NAME="pai_coc_${MODEL_NAME}"
RESUME_EXP_DIR="null"
META_ACTION_FILTER="null"
SAVE_SEGMENT_VIDEOS="false"
DRY_RUN="false"

usage() {
    cat <<'EOF'
Usage: scripts/run_coc_labeling.sh [options]

  --model NAME       qwen3.5_35b (default), gpt5, gpt5.5, or gpt-6.1-sol
  --exp-name NAME    Output experiment suffix
  --resume DIR       Resume an existing experiment directory
  --filter VALUE     Hydra list, e.g. '[gentle_acceleration]'; default: all
  --segment-index FILE  Use a keyframe JSON other than RUN_ROOT/keyframes/...
  --limit N          Run only the first N unique clip/event frames
  --save-videos      Save diagnostic segment videos
  --dry-run          Print the resolved command without starting inference
  --api-config FILE  GPT YAML config; default: ~/.config/alpamayo-coc/openai_api.yaml
  -h, --help         Show this help

Paths can be overridden with DATA_DIR, RUN_ROOT, MODEL_CACHE_DIR, and PYTHON_BIN.
EOF
}

while (($# > 0)); do
    case "$1" in
        --model) MODEL_NAME="$2"; shift 2 ;;
        --exp-name) EXP_NAME="$2"; shift 2 ;;
        --resume) RESUME_EXP_DIR="$2"; shift 2 ;;
        --filter) META_ACTION_FILTER="$2"; shift 2 ;;
        --segment-index) SOURCE_SEGMENT_CONFIG="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --save-videos) SAVE_SEGMENT_VIDEOS="true"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        --api-config) API_CONFIG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "--limit must be a positive integer: $LIMIT" >&2
    exit 2
fi

case "$MODEL_NAME" in
    gpt6.1) MODEL_NAME="gpt-6.1-sol" ;;
    qwen3.5_35b|qwen3.5_397b_fp8|qwen3_vl_235b_awq|gpt5|gpt5.5|gpt-6.1-sol) ;;
    *) echo "Unsupported model: $MODEL_NAME" >&2; exit 2 ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable does not exist: $PYTHON_BIN" >&2
    exit 1
fi

for required_path in \
    "$DATA_DIR" \
    "$RUN_ROOT/cache/trajdata" \
    "$RUN_ROOT/video" \
    "$RUN_ROOT/meta_action/final_outputs" \
    "$SOURCE_SEGMENT_CONFIG"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required input does not exist: $required_path" >&2
        exit 1
    fi
done

# The released sampled JSON is grouped by action bucket.  The same clip/frame
# can occur in two buckets, while the production YAML path is keyed only by
# clip and event timestamp.  Build a temporary globally deduplicated index so
# one event cannot overwrite another result.
if [[ "${DISABLE_EVENT_DEDUP:-false}" != "true" || -n "$LIMIT" ]]; then
    SEGMENT_CONFIG_PATH="$("$PYTHON_BIN" - "$SOURCE_SEGMENT_CONFIG" "$LIMIT" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1])
limit = int(sys.argv[2]) if sys.argv[2] else None
payload = json.loads(source.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise SystemExit("segment config must be a JSON object grouped by action")

deduped = {}
seen = set()
selected_count = 0
for bucket, rows in payload.items():
    for row in rows:
        key = (str(row.get("clip_id")), int(row.get("event_start_frame")))
        if key in seen:
            continue
        seen.add(key)
        deduped.setdefault(str(bucket), []).append(row)
        selected_count += 1
        if limit is not None and selected_count >= limit:
            break
    if limit is not None and selected_count >= limit:
        break

fd, name = tempfile.mkstemp(prefix="coc_segments_dedup_", suffix=".json")
os.close(fd)
Path(name).write_text(json.dumps(deduped, indent=2), encoding="utf-8")
print(name)
PY
    )"
    echo "Using deduplicated segment index: $SEGMENT_CONFIG_PATH"
    if [[ -n "$LIMIT" ]]; then
        echo "Limited inference to at most $LIMIT unique events."
    fi
fi

if [[ "$MODEL_NAME" == gpt* ]]; then
    if [[ ! -f "$API_CONFIG" ]]; then
        echo "GPT config does not exist: $API_CONFIG" >&2
        exit 1
    fi
    mapfile -t API_VALUES < <("$PYTHON_BIN" - "$API_CONFIG" <<'PY'
import base64
import sys
from pathlib import Path
import yaml

payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
api_key = payload.get("api_key", payload.get("OPENAI_API_KEY"))
base_url = payload.get("base_url", payload.get("OPENAI_BASE_URL", ""))
if not isinstance(api_key, str) or not api_key.strip():
    raise SystemExit("API config must contain a non-empty api_key")
for value in (api_key, str(base_url or "")):
    print(base64.b64encode(value.encode()).decode())
PY
    )
    if ((${#API_VALUES[@]} != 2)); then
        echo "Could not read API credentials from $API_CONFIG" >&2
        exit 1
    fi
    export OPENAI_API_KEY="$(printf '%s' "${API_VALUES[0]}" | base64 --decode)"
    export OPENAI_BASE_URL="$(printf '%s' "${API_VALUES[1]}" | base64 --decode)"
    unset NVHOST_OAI_CLIENT_ID NVHOST_OAI_CLIENT_SECRET NVIDIA_API_KEY
else
    export MODEL_CACHE_DIR
fi

cd "$REPO_ROOT"
COMMAND=(
    "$PYTHON_BIN" -m coc_labeling.data_labeling \
    --config-name=base_config_vlm_rel_ts \
    "model_name=$MODEL_NAME" \
    "data.data_dir=$DATA_DIR" \
    "data.cache_dir=$RUN_ROOT/cache/trajdata" \
    "data.video_dir=$RUN_ROOT/video" \
    "data.meta_action_dir=$RUN_ROOT/meta_action/final_outputs" \
    "data.segment_config_path=$SEGMENT_CONFIG_PATH" \
    "resume_exp_dir=$RESUME_EXP_DIR" \
    "exp_name=$EXP_NAME" \
    "data_loader.keyframe.meta_action_filter=$META_ACTION_FILTER" \
    "data_loader.keyframe.deduplicate_segments=false" \
    "data_loader.video.save_segment_videos=$SAVE_SEGMENT_VIDEOS"
)

if [[ "$DRY_RUN" == "true" ]]; then
    printf 'Resolved command:\n'
    printf '  %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

exec "${COMMAND[@]}"
