#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/visualize_ycb_points.py --model "trained_models/ycb/pose_model_current.pth" --use_posecnn_rois