#!/bin/bash
# Usage: <script> name0 name1 ... (if no names are given, provide a list of all datasets)
set -eou pipefail

# Check that the multimeditron scripts have been installed
if ! command -v mm &> /dev/null
then
    echo "mm command could not be found, please install the multimeditron package"
    exit 1
fi

# List of all of the datasets, command to run
BASE_CONFIG_PATH=$(realpath $(dirname "$0")/../config)
echo "Base config path: $BASE_CONFIG_PATH"
DATASETS=(
    "math-shepherd mm preprocess-ds -c $BASE_CONFIG_PATH/rl/ds/config-math-shepherd.yaml"
    "math-shepherd-val mm preprocess-ds -c $BASE_CONFIG_PATH/rl/ds/config-math-shepherd.yaml source.kwargs.split=test output=/capstor/store/cscs/swissai/a127/meditron/multimediset/reasoning/math-shepherd-val.parquet"
    "baai-taco mm preprocess-ds -c $BASE_CONFIG_PATH/rl/ds/config-baai-taco.yaml"
    "nemotron mm preprocess-ds -c $BASE_CONFIG_PATH/rl/ds/config-nemotron-post-training.yaml"
)

# Extract the names of the datasets
ALL_DATASET_NAMES=()
for entry in "${DATASETS[@]}"; do
    name=$(echo $entry | cut -d' ' -f1)
    ALL_DATASET_NAMES+=("$name")
done

# If no arguments are given, display the list of all datasets and exit
if [ "$#" -eq 0 ]; then
    echo "No dataset names provided. Available datasets are:"
    for name in "${ALL_DATASET_NAMES[@]}"; do
        echo " - $name"
    done
    exit 0
fi

# Download the specified datasets
for name in "$@"; do
    found=false
    for entry in "${DATASETS[@]}"; do
        entry_name=$(echo $entry | cut -d' ' -f1)
        if [ "$name" == "$entry_name" ]; then
            found=true
            echo "Downloading dataset: $name"
            # Execute the command to download the dataset
            cmd=$(echo $entry | cut -d' ' -f2-)
            echo "Running command: $cmd"
            $cmd
            echo "Finished downloading dataset: $name"
            break
        fi
    done
    if [ "$found" = false ]; then
        echo "Dataset name '$name' not recognized. Available datasets are:"
        for valid_name in "${ALL_DATASET_NAMES[@]}"; do
            echo " - $valid_name"   
        done
        exit 1
    fi
done