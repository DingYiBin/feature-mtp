#!/bin/bash
# source 
which python
# source ~/.bashrc
# which python
# conda activate /public/workspace/dyb/headless/conda

# which python
# Runs the "175B" parameter model
# conda activate /public/workspace/dyb/headless/conda
export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_PATH=/public/workspace/dyb/compress/thirdparty/Megatron-LM
# MEGATRON_PATCH_PATH=/workspace-dyb/mirror/thirdparty/Pai-Megatron-Patch
BASIC_PATH=/public/workspace/dyb/compress/
export PYTHONPATH=$PYTHONPATH:${BASIC_PATH}:${MEGATRON_PATH}
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/public/workspace/dyb/mirror/.venv/lib/python3.12/site-packages/torch/lib
echo $LD_LIBRARY_PATH
echo $PYTHONPATH
# exit
GPUS_PER_NODE=$HOST_GPU_NUM
# Change for multinode config
# MASTER_ADDR=localhost
# MASTER_PORT=6000
NUM_NODES=$HOST_NUM
# NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

EXPERIMENT_PATH=$1
TRAIN_MODEL_MODE=$2
NUM_LAYERS=$3
MTP_NUM_LAYERS=$4
POS_LOSS_SCALE=$5
DATA_PATH=$6 #<Specify path and file prefix>_text_document
NUM_TRAIN_ITERATIONS=$7
SAVE_INTERVAL=$8
JUST_CONVERT_CKPT=$9

CHECKPOINT_PATH=$EXPERIMENT_PATH/ckpt #<Specify path>
TENSORBOARD_LOGS_PATH=$EXPERIMENT_PATH/logs #<Specify path>

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --node_rank $NODE_RANK
)

MODEL_PARALLEL_ARGS=(
	--pipeline-model-parallel-size 1 
)

GPT_MODEL_ARGS=(
    --attention-backend auto # Can use (flash/fused/unfused/local)
    --no-masked-softmax-fusion
    --disable-bias-linear
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --group-query-attention
    --kv-channels 128
    --use-rotary-position-embeddings
    --rotary-percent 1.0
    --no-bias-swiglu-fusion
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --no-rope-fusion
    --no-gradient-accumulation-fusion
)
if [ $JUST_CONVERT_CKPT = 1 ]; then
    GPT_MODEL_ARGS+=(
        --ckpt-convert-format torch
        --ckpt-convert-save ${CHECKPOINT_PATH}
    )
fi

GLOBAL_BATCH_SIZE=1024
MICRO_BATCH_SIZE=1

GPT_MODEL_ARGS+=(
    --num-layers $NUM_LAYERS
    --hidden-size 2560
    --ffn-hidden-size 9728
    --num-attention-heads 32
    --num-query-groups 8
    --max-position-embeddings 40960
    --rotary-base 1000000
    --norm-epsilon 1e-6
    --qk-layernorm
    --make-vocab-size-divisible-by 1187
    --untie-embeddings-and-output-weights
)


# if [ $TRAIN_MODEL_MODE = 1 ]; then
#     # MTP_NUM_LAYERS=1
#     export NUM_PREDICTION_TOKENS=4
#     export NUM_PREDICTION_TOKENS_FOCUSED=2
#     # GLOBAL_BATCH_SIZE=1024
# fi
# if [ $TRAIN_MODEL_MODE = 2 ]; then
#     # MTP_NUM_LAYERS=1
#     export NUM_PREDICTION_TOKENS=4
#     export NUM_PREDICTION_TOKENS_FOCUSED=2
#     # GLOBAL_BATCH_SIZE=1024
# fi
# GLOBAL_BATCH_SIZE=$((16*$NUM_NODES))
GLOBAL_BATCH_SIZE=1024
MICRO_BATCH_SIZE=2
GPT_MODEL_ARGS+=(
    --mtp-num-layers ${MTP_NUM_LAYERS}
)
MODEL_PARALLEL_ARGS+=(
    --tensor-model-parallel-size 2
)

TOEKENIZER_MODEL=/public/llm_models/Qwen/Qwen3-30B-A3B-Instruct-2507

NUM_WARMUP_ITERATIONS=$((NUM_TRAIN_ITERATIONS/16))
NUM_DECAY_ITERATIONS=$((NUM_TRAIN_ITERATIONS/2))

MAX_LR=1.0e-4
MIN_LR=1.0e-5

# if [ $TRAIN_MODEL_MODE = 1 ]; then
#     # MAX_LR=1.0e-2
#     # MIN_LR=1.0e-3
# fi


TRAINING_ARGS=(
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.008 
    --clip-grad 1.0 
    --bf16
    --seq-length 4096
    --lr $MAX_LR
    --lr-decay-style cosine 
    --min-lr $MIN_LR
    --lr-decay-iters ${NUM_DECAY_ITERATIONS}
    --lr-warmup-iters ${NUM_WARMUP_ITERATIONS}
    --train-iters $NUM_TRAIN_ITERATIONS
    --train-model-mode $TRAIN_MODEL_MODE
    --pos-loss-scale $POS_LOSS_SCALE
    --convert-checkpoint
)

# if [ $TRAIN_MODEL_MODE = 1 ]; then
#     TRAINING_ARGS+=(
#         --main-model-checkpoint /public/converted-ckpt/Qwen3-4B-Instruct-2507-tp2/release
#         --main-model-checkpoint-dtype torch
#     )
# fi

DATA_ARGS=(
    --data-path $DATA_PATH 
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOEKENIZER_MODEL
    --split 949,50,1
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval $SAVE_INTERVAL
    --eval-interval 163840
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# /root/miniconda3/condabin/conda run -p /public/workspace/dyb/headless/conda \
torchrun ${DISTRIBUTED_ARGS[@]} lstm/pretrain.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
