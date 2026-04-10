source /public/workspace/dyb/compress/.venv/bin/activate
# conda init
# conda activate /public/workspace/dyb/headless/conda
which python
export MASTER_ADDR=localhost
export MASTER_PORT=6000
export NODE_RANK=0
export HOST_NUM=1
export HOST_GPU_NUM=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
LOGS_FILE=/public/workspace/dyb/compress/logs/dev.log
echo $LOGS_FILE

EXPERIMENTS_DIR=/public/workspace/dyb/experiments

# experiment
EXPERIMENT_NAME=dev
export MODEL_FINETUNED_TYPE=i
export NUM_TOKENS_LOOK_BACK=1
export NUM_PREDICTION_TOKENS_FOCUSED=4

EXPERIMENT_PATH=$EXPERIMENTS_DIR/$EXPERIMENT_NAME

rm -fr /public/workspace/dyb/experiments/dev

# export NUM_TRANSFORMER_BLOCK_ONE_MTP_LAYER=2
bash lstm/train.sh \
    ${EXPERIMENT_PATH} \
    1 \
    2 \
    /workspace-dyb/converted_dataset/shareAI/ShareGPT-Chinese-English-90k/sharegpt_jsonl/processed_data_text_document \
    64 \
    4 \
    0 \
    &> $LOGS_FILE