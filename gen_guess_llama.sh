#!/bin/bash

#SBATCH --job-name=gen_guess_llama
#SBATCH --time=2-00:00:00
#SBATCH --mem=200gb
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:3 # 3 for llama 3.1 8b and 8 for llama 3.1 70b
#SBATCH --output=output_gen_guess_llama_%j.out
#SBATCH --error=output_gen_guess_llama_%j.err

#export PYTHONPATH=$PYTHONPATH:$(pwd)/autoexpl:$(pwd)/llama_repo
export PYTHONPATH=$PYTHONPATH:$(pwd)/autoexpl

#module load cuda

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export CUDA_VISIBLE_DEVICES=0
#export TF_FORCE_GPU_ALLOW_GROWTH=true

# set hf token as an environment variable
export HF_HOME="/fs/clip-scratch/eloirghi/explicitation/.cache"
export HF_TOKEN="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"


LANG=es
DATASET=esqbv1htall #plqbv1ht512 
MP=1
model_size=7B
START=0
END=2 #instead of -1 because that takes way too long
MAX_LEN=660
PORT=29509

#TARGET_FOLDER=/path/to/llama_weights
SCRIPT=/fs/clip-projects/qest/eloirghi/automatic_explicitation_LLM/autoexpl/xqb/gen_guess_llama_split.py # changed script path - hiba
#torchrun --nnodes=1 --master_port $PORT --nproc_per_node $MP $SCRIPT --lang $LANG --dataset-name $DATASET --ckpt-dir $TARGET_FOLDER/$model_size --start $START --end $END --max-seq-len $MAX_LEN

# hiba: trying below since the line above throws the following error: torchrun: error: ambiguous option: --start could match --start-method, --start_method
#torchrun --nnodes=1 --master_port $PORT --nproc_per_node $MP -- \
#    $SCRIPT --lang $LANG --dataset-name $DATASET --ckpt-dir $TARGET_FOLDER/$model_size \
#    --start $START --end $END --max-seq-len $MAX_LEN


# No need for TARGET_FOLDER anymore as we're loading from HF
#torchrun --nnodes=1 --master_port $PORT --nproc_per_node $MP -- \
#    $SCRIPT --lang $LANG --dataset-name $DATASET \
#    --start $START --end $END --max-seq-len $MAX_LEN

# No need for TARGET_FOLDER anymore as we're loading from HF
torchrun --nnodes=1 --master_port $PORT --nproc_per_node $MP -- \
$SCRIPT --lang $LANG --dataset-name $DATASET \
    --start $START --end $END --max-seq-len $MAX_LEN \
    --model-id "meta-llama/Llama-3.1-8B"

