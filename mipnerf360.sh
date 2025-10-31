#!/bin/bash

# training settings
record_running_output=true
iterations=30000
depth_correct=true

# data path, scenes and output path
DATA_BASE_PATH="data/mipnerf360"
OUTPUT_BASE_PATH="outputs/mipnerf360"
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

# create command list
command_list=()
for scene in ${scenes[@]}; do
    # input and output paths
    scene_output_path=${OUTPUT_BASE_PATH}/${scene}
    data_path=${DATA_BASE_PATH}/${scene}
    mkdir -p ${scene_output_path}

    ### train command ###
    extra_args=" --eval --port 0 --iterations ${iterations} --data_device cuda "
    [ "$depth_correct" = true ] && extra_args+=" --depth_correct"
    [ "$record_running_output" = true ] && extra_args+=" &>> ${scene_output_path}/running.txt"
    command="python train.py -s ${data_path} -m ${scene_output_path} ${extra_args}"
    echo execute command: $command
    command_list+=("$command")

    # ### render command ###
    # extra_args=" --store_image --skip_train --data_device disk "
    # [ "$record_running_output" = true ] && extra_args+=" &>> ${scene_output_path}/running.txt"
    # command="python render.py -m ${scene_output_path} ${extra_args}"
    # echo execute command: $command
    # command_list+=("$command")

    # ### metrics command ###
    # extra_args=""
    # [ "$record_running_output" = true ] && extra_args+=" &>> ${scene_output_path}/running.txt"
    # command="python metrics.py -m ${scene_output_path} ${extra_args}"
    # echo execute command: $command
    # command_list+=("$command")
done

# parallel execution
n_jobs=4
delay_time=20
parallel --jobs ${n_jobs} --delay ${delay_time} ::: "${command_list[@]}"