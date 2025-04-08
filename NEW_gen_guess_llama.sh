#!/bin/bash
#SBATCH --job-name=NEW_gen_guess_llama
#SBATCH --time=2-00:00:00
#SBATCH --mem=200gb
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:3
#SBATCH --output=output_NEW_gen_guess_llama_%j.out
#SBATCH --error=output_NEW_gen_guess_llama_%j.err

export PYTHONPATH=$PYTHONPATH:$(pwd)/autoexpl
export HF_HOME="/fs/clip-scratch/eloirghi/explicitation/.cache"
export HF_TOKEN="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"

# Improved debugging configuration
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=WARN
export TORCH_CPP_LOG_LEVEL=INFO
export CUDA_LAUNCH_BLOCKING=1

# Configuration parameters
LANG=pl
DATASET=plqbv1ht512
MP=3
START=0
END=5
MAX_LEN=660
PORT=29500  # Standardized port
MAX_BATCH_SIZE=2
SAVE_INTERVAL=5

SCRIPT=/fs/clip-projects/qest/eloirghi/automatic_explicitation_LLM/autoexpl/xqb/NEW_gen_guess_llama_split.py

# Fixed torchrun command structure
torchrun \
    --nnodes=1 \
    --nproc_per_node $MP \
    --master_port $PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$PORT \
    $SCRIPT \
    --lang $LANG \
    --dataset-name $DATASET \
    --max-seq-len $MAX_LEN \
    --model-id "meta-llama/Llama-3.1-8B" \
    --max-batch-size $MAX_BATCH_SIZE \
    --end $END \
    --save-interval $SAVE_INTERVAL
