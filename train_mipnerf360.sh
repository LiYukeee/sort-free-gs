#!/bin/bash

OUPUT_BASE_PATH="outputs/mipnerf360"
DATA_BASE_PATH="data/mipnerf360"
scenes=(
  "bicycle"
  "bonsai"
  "counter"
  "flowers"
  "garden"
  "kitchen"
  "room"
  "stump"
  "treehill"
)


depth_correct_options=(
    true
    false
)

# Train
for depth_correct in "${depth_correct_options[@]}"; do
  # start training seprate scene
  OUTPUT_PATH=${OUPUT_BASE_PATH}/DC-${depth_correct}
  for scene in ${scenes[@]}; do
    scene_output_path=${OUTPUT_PATH}/${scene}
    mkdir -p ${scene_output_path}
    if [ "$depth_correct" = true ]; then
      python train.py --eval -s ${DATA_BASE_PATH}/${scene} -m ${scene_output_path} --port 0 --depth_correct
    else
      python train.py --eval -s ${DATA_BASE_PATH}/${scene} -m ${scene_output_path} --port 0
    fi
  done
done
