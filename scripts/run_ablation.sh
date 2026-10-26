#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-}
EXPERIMENT=${2:-}
VARIANT=${3:-}
SELECTOR_INPUT=${4:-${KEYFRAME_SELECTOR:-rba}}

if [[ -z "${DATASET}" || -z "${EXPERIMENT}" || -z "${VARIANT}" ]]; then
  echo "Usage: $0 <charades|activitynet|tacos> <exp1|exp3|exp4|exp5> <variant> [rba|original_aks]"
  exit 2
fi

case "${SELECTOR_INPUT}" in
  rba|rab|modified|modified_aks|modified_aks_trs)
    KEYFRAME_SELECTOR=rba
    KEYFRAME_SUBDIR=modified_aks_trs
    ;;
  original|aks|original_aks)
    KEYFRAME_SELECTOR=original_aks
    KEYFRAME_SUBDIR=original_aks
    ;;
  *)
    echo "Unknown keyframe selector: ${SELECTOR_INPUT}"
    echo "Available: rba (alias: rab) | original_aks"
    exit 2
    ;;
esac

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case "${DATASET}" in
  charades)
    CONFIG="${ROOT_DIR}/configs/pool_charades_16x16_k5l8.yaml"
    KEYFRAME_DATASET_DIR=Charades_CD
    ;;
  activitynet)
    CONFIG="${ROOT_DIR}/configs/pool_activitynet_64x64_k9l4.yaml"
    KEYFRAME_DATASET_DIR=ActivityNet-CD
    ;;
  *)
    echo "Unknown dataset: ${DATASET}"
    exit 2
    ;;
esac

# ----------------------------------------------------------------------
# Keyframe settings
# ----------------------------------------------------------------------


KEYFRAME_ROOT=${KEYFRAME_ROOT:-"${ROOT_DIR}/dataset"}


SINGLE_KEY=${SINGLE_KEY:-k1}

if [[ -z "${MULTI_KEY:-}" ]]; then
  case "${DATASET}" in
    charades) MULTI_KEY=k4 ;;
    activitynet) MULTI_KEY=k2 ;;
    tacos) MULTI_KEY=k4 ;;
  esac
fi

OPTS=()
OPTS+=(DATASETS.PSEUDO_GLANCE.SELECTOR "${KEYFRAME_SELECTOR}")

# Record the k-file required by the current experiment.
SELECTED_KEY=""

case "${EXPERIMENT}:${VARIANT}" in
  exp1:single_pg)
    SELECTED_KEY="${SINGLE_KEY}"
    OPTS+=(DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_KEY "${SINGLE_KEY}")
    OPTS+=(DATASETS.PSEUDO_GLANCE.KEYFRAME_MODE single)
    OPTS+=(DATASETS.PSEUDO_GLANCE.REFINEMENT none)
    OPTS+=(DATASETS.PSEUDO_GMASK.MODE single)
    OPTS+=(SOLVER.DGA.MODE multi_weighted)
    ;;

  exp1:multi_pg)
    SELECTED_KEY="${MULTI_KEY}"
    OPTS+=(DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_KEY "${MULTI_KEY}")
    OPTS+=(DATASETS.PSEUDO_GLANCE.KEYFRAME_MODE multi)
    OPTS+=(DATASETS.PSEUDO_GLANCE.REFINEMENT none)
    OPTS+=(DATASETS.PSEUDO_GMASK.MODE union)
    OPTS+=(SOLVER.DGA.MODE multi_weighted)
    ;;

  exp1:refined_multi_pg)
    SELECTED_KEY="${MULTI_KEY}"
    OPTS+=(DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_KEY "${MULTI_KEY}")
    OPTS+=(DATASETS.PSEUDO_GLANCE.KEYFRAME_MODE multi)
    OPTS+=(DATASETS.PSEUDO_GLANCE.REFINEMENT dedup_filter)
    OPTS+=(DATASETS.PSEUDO_GMASK.MODE union)
    OPTS+=(SOLVER.DGA.MODE multi_weighted)
    ;;

  exp1:qpm)
    SELECTED_KEY="${MULTI_KEY}"
    OPTS+=(DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_KEY "${MULTI_KEY}")
    OPTS+=(DATASETS.PSEUDO_GLANCE.KEYFRAME_MODE multi)
    OPTS+=(DATASETS.PSEUDO_GLANCE.REFINEMENT dedup_filter)
    OPTS+=(DATASETS.PSEUDO_GMASK.MODE qpm)
    OPTS+=(SOLVER.DGA.MODE multi_weighted)
    ;;

  exp3:k*)
    K=${VARIANT#k}
    if ! [[ "${K}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Invalid exp3 variant: ${VARIANT}. Expected k<positive integer>, e.g. k4 or k16."
      exit 2
    fi
    SELECTED_KEY="${VARIANT}"
    OPTS+=(DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_KEY "${VARIANT}")
    OPTS+=(DATASETS.PSEUDO_GLANCE.KEYFRAME_MODE multi)
    OPTS+=(DATASETS.PSEUDO_GLANCE.MAX_KEYFRAMES 0)
    OPTS+=(DATASETS.PSEUDO_GLANCE.REFINEMENT dedup_filter)
    OPTS+=(DATASETS.PSEUDO_GMASK.MODE qpm)
    OPTS+=(SOLVER.DGA.MODE multi_weighted)
    ;;

  exp4:static_multi)
    SELECTED_KEY="${MULTI_KEY}"
    OPTS+=(DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_KEY "${MULTI_KEY}")
    OPTS+=(DATASETS.PSEUDO_GLANCE.KEYFRAME_MODE multi)
    OPTS+=(DATASETS.PSEUDO_GLANCE.REFINEMENT dedup_filter)
    OPTS+=(DATASETS.PSEUDO_GMASK.MODE qpm)
    OPTS+=(SOLVER.DGA.MODE static_multi)
    ;;

  exp4:single_anchor)
    SELECTED_KEY="${MULTI_KEY}"
    OPTS+=(DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_KEY "${MULTI_KEY}")
    OPTS+=(DATASETS.PSEUDO_GLANCE.KEYFRAME_MODE multi)
    OPTS+=(DATASETS.PSEUDO_GLANCE.REFINEMENT dedup_filter)
    OPTS+=(DATASETS.PSEUDO_GMASK.MODE qpm)
    OPTS+=(SOLVER.DGA.MODE single_anchor)
    ;;

  exp4:multi_mean)
    SELECTED_KEY="${MULTI_KEY}"
    OPTS+=(DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_KEY "${MULTI_KEY}")
    OPTS+=(DATASETS.PSEUDO_GLANCE.KEYFRAME_MODE multi)
    OPTS+=(DATASETS.PSEUDO_GLANCE.REFINEMENT dedup_filter)
    OPTS+=(DATASETS.PSEUDO_GMASK.MODE qpm)
    OPTS+=(SOLVER.DGA.MODE multi_mean)
    ;;

  exp4:multi_weighted)
    SELECTED_KEY="${MULTI_KEY}"
    OPTS+=(DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_KEY "${MULTI_KEY}")
    OPTS+=(DATASETS.PSEUDO_GLANCE.KEYFRAME_MODE multi)
    OPTS+=(DATASETS.PSEUDO_GLANCE.REFINEMENT dedup_filter)
    OPTS+=(DATASETS.PSEUDO_GMASK.MODE qpm)
    OPTS+=(SOLVER.DGA.MODE multi_weighted)
    ;;

  exp5:confidence|exp5:length|exp5:qpm)
    SELECTED_KEY="${MULTI_KEY}"
    OPTS+=(DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_KEY "${MULTI_KEY}")
    OPTS+=(DATASETS.PSEUDO_GLANCE.KEYFRAME_MODE multi)
    OPTS+=(DATASETS.PSEUDO_GLANCE.REFINEMENT dedup_filter)
    OPTS+=(DATASETS.PSEUDO_GMASK.MODE "${VARIANT}")
    OPTS+=(SOLVER.DGA.MODE multi_weighted)
    ;;

  *)
    echo "Unknown experiment/variant: ${EXPERIMENT}:${VARIANT}"
    exit 2
    ;;
esac

# ----------------------------------------------------------------------
# Automatically resolve the keyframe JSON.
#
# Expected directory layout:
#   Charades_CD/
#     original_aks/selected_frames_4.json
#     modified_aks_trs/selected_frames_4.json
#
#   ActivityNet-CD/
#     original_aks/selected_frames_2.json
#     modified_aks_trs/selected_frames_2.json
#
# Explicit KEYFRAME_FILE still has highest priority.
# ----------------------------------------------------------------------

if [[ -z "${KEYFRAME_FILE:-}" && -n "${KEYFRAME_DATASET_DIR}" && -n "${SELECTED_KEY}" ]]; then
  K_NUM=${SELECTED_KEY#k}
  KEYFRAME_FILE="${KEYFRAME_ROOT}/${KEYFRAME_DATASET_DIR}/${KEYFRAME_SUBDIR}/selected_frames_${K_NUM}.json"
fi

if [[ -n "${KEYFRAME_FILE:-}" ]]; then
  if [[ ! -f "${KEYFRAME_FILE}" ]]; then
    echo "Keyframe file not found:"
    echo "  ${KEYFRAME_FILE}"
    exit 2
  fi

  SELECTED_FILE=$(cd "$(dirname "${KEYFRAME_FILE}")" && pwd)/$(basename "${KEYFRAME_FILE}")

  # Explicit file path overrides paths_catalog.py.
  OPTS+=(DATASETS.PSEUDO_GLANCE.MAX_KEYFRAMES 0)
  OPTS+=(DATASETS.PSEUDO_GLANCE.SELECTED_FRAMES_FILE "${SELECTED_FILE}")
fi

# ----------------------------------------------------------------------
# Output directory
# ----------------------------------------------------------------------

RUN_ID=$(date +%Y%m%d_%H%M%S)

OUTPUT_ROOT=${OUTPUT_ROOT:-"${ROOT_DIR}/outputs"}

# Use KEYFRAME_SELECTOR here. The old script used an undefined variable SELECTOR.
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${EXPERIMENT}/${DATASET}/${KEYFRAME_SELECTOR}/${VARIANT}/${RUN_ID}"}

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29510}

mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo "Dataset           : ${DATASET}"
echo "Experiment        : ${EXPERIMENT}"
echo "Variant           : ${VARIANT}"
echo "Keyframe selector : ${KEYFRAME_SELECTOR}"
echo "Selected key      : ${SELECTED_KEY}"
if [[ -n "${KEYFRAME_FILE:-}" ]]; then
  echo "Keyframe file     : ${KEYFRAME_FILE}"
else
  echo "Keyframe file     : paths_catalog.py"
fi
echo "Config            : ${CONFIG}"
echo "Output            : ${OUTPUT_DIR}"
echo "Overrides         : ${OPTS[*]}"
echo "=================================================="

cd "${ROOT_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'torchrun --nproc_per_node=%q --master_addr=%q --master_port=%q train_net.py --config-file %q OUTPUT_DIR %q' \
    "${NPROC_PER_NODE}" "${MASTER_ADDR}" "${MASTER_PORT}" "${CONFIG}" "${OUTPUT_DIR}"
  printf ' %q' "${OPTS[@]}"
  printf '\n'
  exit 0
fi

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train_net.py \
  --config-file "${CONFIG}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  "${OPTS[@]}"
