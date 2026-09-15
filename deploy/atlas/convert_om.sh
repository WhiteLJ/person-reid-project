#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ATC_BIN="${ATC_BIN:-/usr/local/Ascend/ascend-toolkit/latest/bin/atc}"
SOC_VERSION="${SOC_VERSION:-Ascend310B4}"
YOLO_IMAGE_SIZE="${YOLO_IMAGE_SIZE:-640}"
REID_HEIGHT="${REID_HEIGHT:-256}"
REID_WIDTH="${REID_WIDTH:-128}"
REID_DYNAMIC_BATCH_SIZE="${REID_DYNAMIC_BATCH_SIZE:-1,2,4,8}"

YOLO_ONNX="${1:-${PROJECT_ROOT}/deploy/atlas/onnx/yolov8n.onnx}"
REID_ONNX="${2:-${PROJECT_ROOT}/deploy/atlas/onnx/osnet_x0_25.onnx}"
OUTPUT_DIR="${3:-${PROJECT_ROOT}/weights/atlas}"

if [[ ! -f "${CANN_ENV}" ]]; then
    echo "CANN environment script not found: ${CANN_ENV}" >&2
    exit 1
fi
if [[ ! -x "${ATC_BIN}" ]]; then
    echo "ATC executable not found: ${ATC_BIN}" >&2
    exit 1
fi
if [[ ! -f "${YOLO_ONNX}" ]]; then
    echo "YOLO ONNX file not found: ${YOLO_ONNX}" >&2
    exit 1
fi
if [[ ! -f "${REID_ONNX}" ]]; then
    echo "OSNet ONNX file not found: ${REID_ONNX}" >&2
    exit 1
fi

source "${CANN_ENV}"
mkdir -p "${OUTPUT_DIR}"

YOLO_NAME="$(basename -- "${YOLO_ONNX}" .onnx)"
REID_NAME="$(basename -- "${REID_ONNX}" .onnx)"

"${ATC_BIN}" \
    --model="${YOLO_ONNX}" \
    --framework=5 \
    --output="${OUTPUT_DIR}/${YOLO_NAME}" \
    --input_format=NCHW \
    --input_shape="images:1,3,${YOLO_IMAGE_SIZE},${YOLO_IMAGE_SIZE}" \
    --soc_version="${SOC_VERSION}"

"${ATC_BIN}" \
    --model="${REID_ONNX}" \
    --framework=5 \
    --output="${OUTPUT_DIR}/${REID_NAME}" \
    --input_format=NCHW \
    --input_shape="images:-1,3,${REID_HEIGHT},${REID_WIDTH}" \
    --dynamic_batch_size="${REID_DYNAMIC_BATCH_SIZE}" \
    --soc_version="${SOC_VERSION}"

YOLO_OM="${OUTPUT_DIR}/${YOLO_NAME}.om"
REID_OM="${OUTPUT_DIR}/${REID_NAME}.om"
[[ -f "${YOLO_OM}" ]] || { echo "ATC did not create ${YOLO_OM}" >&2; exit 1; }
[[ -f "${REID_OM}" ]] || { echo "ATC did not create ${REID_OM}" >&2; exit 1; }

echo "Created ${YOLO_OM}"
echo "Created ${REID_OM}"
echo "SOC_VERSION=${SOC_VERSION}"

