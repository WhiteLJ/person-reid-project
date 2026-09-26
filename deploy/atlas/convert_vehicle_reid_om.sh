#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ATC_BIN="${ATC_BIN:-/usr/local/Ascend/ascend-toolkit/latest/bin/atc}"
SOC_VERSION="${SOC_VERSION:-Ascend310B4}"
VEHICLE_ONNX="${1:-${PROJECT_ROOT}/deploy/atlas/onnx/vehicle_sbs_r50_ibn.onnx}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/weights/atlas}"
OUTPUT_BASENAME="vehicle_sbs_r50_ibn"

if [[ ! -f "${CANN_ENV}" ]]; then
    echo "CANN environment script not found: ${CANN_ENV}" >&2
    exit 1
fi
if [[ ! -x "${ATC_BIN}" ]]; then
    echo "ATC executable not found: ${ATC_BIN}" >&2
    exit 1
fi
if [[ ! -f "${VEHICLE_ONNX}" ]]; then
    echo "Vehicle ONNX file not found: ${VEHICLE_ONNX}" >&2
    exit 1
fi

# CANN 6.2.RC2 does not need an ``atc --version`` probe here.  The actual
# conversion below is the health check and preserves the full ATC error log.
source "${CANN_ENV}"
mkdir -p "${OUTPUT_DIR}"

"${ATC_BIN}" \
    --model="${VEHICLE_ONNX}" \
    --framework=5 \
    --output="${OUTPUT_DIR}/${OUTPUT_BASENAME}" \
    --input_format=NCHW \
    --input_shape="images:-1,3,256,256" \
    --dynamic_batch_size="1,2,4,8" \
    --soc_version="${SOC_VERSION}"

VEHICLE_OM="${OUTPUT_DIR}/${OUTPUT_BASENAME}.om"
if [[ ! -f "${VEHICLE_OM}" ]]; then
    echo "ATC did not create ${VEHICLE_OM}" >&2
    exit 1
fi

echo "Created ${VEHICLE_OM}"
echo "SOC_VERSION=${SOC_VERSION}"
