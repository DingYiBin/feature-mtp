source /public/workspace/dyb/compress/.venv/bin/activate
which python
echo MASTER_ADDR=$MASTER_ADDR
echo MASTER_PORT=$MASTER_PORT
echo NODE_RANK=$NODE_RANK
echo HOST_NUM=$HOST_NUM
echo HOST_GPU_NUM=$HOST_GPU_NUM



EXPERIMENTS_DIR=/public/workspace/dyb/experiments
# export ADD_E_PROJ=0
# experiment
EXPERIMENT_NAME=pretrain-qwen3-4b-lstm-4
mkdir -p /public/workspace/dyb/compress/logs/$EXPERIMENT_NAME
LOGS_FILE=/public/workspace/dyb/compress/logs/$EXPERIMENT_NAME/log_$(date +%Y-%m%d-%H%M-%S)_${NODE_RANK}_${HOST_NUM}.log
echo $LOGS_FILE
MTP_EH_PROJ_MODE=0
JUST_CONVERT_CKPT=0
export NUM_TOKENS_LOOK_BACK=1
export NUM_PREDICTION_TOKENS_FOCUSED=4

EXPERIMENT_PATH=$EXPERIMENTS_DIR/$EXPERIMENT_NAME

bash lstm/train.sh \
    ${EXPERIMENT_PATH} \
    1 \
    4 \
    /workspace-dyb/converted_dataset/all-0122 \
    32768 \
    4096 \
    ${JUST_CONVERT_CKPT} \
    &> $LOGS_FILE
