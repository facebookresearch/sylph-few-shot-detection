#!/bin/bash
set -e
set -x

if [ -z "$FORCE_BUILD" ]; then
  FORCE_BUILD=true
fi

RULENAME=train_net
BASE_FOLDER=//cv_mm/sylph/tools
MODE=dev-nosan

PAR_FILE="/data/users/$USER/fbsource/fbcode/buck-out/gen/$BASE_FOLDER/$RULENAME.par"

ROOT_DIR=/data/users/$USER/fbsource/fbcode/cv_mm/sylph
TASK="COCO-Meta-FCOS-Detection"
# TASK="COCO-Detection-Few-Shot"
# TASK="LVISv1-Detection"
# TASK="LVISv1-Detection-Few-Shot"

CONFIG_DIR="$ROOT_DIR/configs/$TASK"

# META_ARCH="mask_rcnn"
# CONFIG="mask_rcnn_R_50_FPN_1x.yaml"
# RUNNER="egodet.runner.EgoDetRunner"

# META_ARCH="fcos"
# CONFIG="SJ_R_50.yaml"
# RUNNER="egodet.runner.AdelaiDetRunner"

META_ARCH="meta-fcos"
CONFIG="Meta_FCOS_MS_R_50_1x.yaml"
RUNNER="sylph.runner.MetaFCOSRunner"

# META_ARCH="meta_fcos"
# CONFIG="Meta_FCOS_SJ_SWIN_T_pt_IN22K_pretrain.yaml"
# RUNNER="egodet.runner.MetaAdelaiDetRunner"

# META_ARCH="detr"
# CONFIG="fewshot_sj_deformable_detr_swin_t.yaml"
# RUNNER="egodet.runner.FewShotEgoDETRRunner"

# build par if it does not exist:
if [ ! -f "$PAR_FILE" ] || [ "$FORCE_BUILD" = true ]; then
  buck build @mode/$MODE $BASE_FOLDER:$RULENAME
fi

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
  "$PAR_FILE" \
  --num-processes 8 \
  --runner $RUNNER \
  --config-file "$CONFIG_DIR/$META_ARCH/$CONFIG" \
  --output-dir /tmp \
