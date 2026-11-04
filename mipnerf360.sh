#!/bin/bash

n_jobs=4
delay_time=20

# Parse arguments: if "render" or "--render" is provided, only render; otherwise train then render
RENDER_ONLY=false
for arg in "$@"; do
    case "$arg" in
        render|--render|-r)
            RENDER_ONLY=true
            ;;
        -h|--help)
            echo "Usage: $0 [render]"
            echo "  (no args): Train all scenes, then render"
            echo "  render:    Render only (skip training)"
            exit 0
            ;;
    esac
done

if [ "$RENDER_ONLY" = true ]; then
    echo "=== Mode: Render only ==="
else
    echo "=== Mode: Train and then render ==="
fi


# training settings
record_running_output=true
iterations=30000
depth_correct=true

# python executable
PYTHON_EXEC="python"

# Ensure script is run from sort-free-gs repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

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
    train_args=" --eval --port 0 --iterations ${iterations} --data_device cuda "
    [ "$depth_correct" = true ] && train_args+=" --depth_correct"
    [ "$record_running_output" = true ] && train_args+=" &>> ${scene_output_path}/running.txt"
    train_command="${PYTHON_EXEC} train.py -s ${data_path} -m ${scene_output_path} ${train_args}"

    ### render & metrics command ###
    render_args=" --store_image --skip_train --data_device disk "
    if [ "$record_running_output" = true ]; then
        render_command="${PYTHON_EXEC} render.py -m ${scene_output_path} ${render_args} &>> ${scene_output_path}/render_metrics.txt && ${PYTHON_EXEC} metrics.py -m ${scene_output_path} &>> ${scene_output_path}/render_metrics.txt"
    else
        render_command="${PYTHON_EXEC} render.py -m ${scene_output_path} ${render_args} && ${PYTHON_EXEC} metrics.py -m ${scene_output_path}"
    fi

    if [ "$RENDER_ONLY" = true ]; then
        command="${render_command}"
    else
        command="${train_command} && ${render_command}"
    fi

    echo execute command: $command
    command_list+=("$command")
done

# parallel execution
parallel --jobs ${n_jobs} --delay ${delay_time} ::: "${command_list[@]}"
