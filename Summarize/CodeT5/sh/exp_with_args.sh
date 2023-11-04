WORKDIR="/home/backdoor2023/backdoor/Summarize/CodeT5"
export PYTHONPATH=$WORKDIR

attack_way='stylechg'
poison_rate=0.01
trigger=8.2
cuda_device=0,1

TASK=${1}
SUB_TASK=${2}
MODEL_TAG=${3}
GPU=${4}
DATA_NUM=${5}
BS=${6}
LR=${7}
SRC_LEN=${8}
TRG_LEN=${9}
PATIENCE=${10}
EPOCH=${11}
WARMUP=${12}
MODEL_DIR=${13}
SUMMARY_DIR=${14}
RES_FN=${15}

OUTPUT_DIR=${MODEL_DIR}/${attack_way}_${trigger}_${poison_rate}
CACHE_DIR=${OUTPUT_DIR}/cache_data
RES_DIR=${OUTPUT_DIR}/prediction
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

if [[ $MODEL_TAG == roberta ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=roberta-base
elif [[ $MODEL_TAG == codebert ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=microsoft/codebert-base
elif [[ $MODEL_TAG == bart_base ]]; then
  MODEL_TYPE=bart
  TOKENIZER=facebook/bart-base
  MODEL_PATH=facebook/bart-base
elif [[ $MODEL_TAG == codet5_small ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-small
  MODEL_PATH=Salesforce/codet5-small
elif [[ $MODEL_TAG == codet5_base ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=/home/backdoor2023/backdoor/base_model/codet5-base
  MODEL_PATH=/home/backdoor2023/backdoor/base_model/codet5-base
elif [[ $MODEL_TAG == codet5_large ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-large
  MODEL_PATH=Salesforce/codet5-large
fi


if [[ ${TASK} == 'multi_task' ]]; then
  RUN_FN=${WORKDIR}/run_multi_gen.py
  MULTI_TASK_AUG='--max_steps '${16}' --save_steps '${17}' --log_steps '${18}
elif [[ ${TASK} == 'clone' ]]; then
  RUN_FN=${WORKDIR}/run_clone.py
elif [[ ${TASK} == 'defect' ]]; then
  RUN_FN=${WORKDIR}/run_defect.py
else
  RUN_FN=${WORKDIR}/run_gen.py
fi

DATA_PATH=/home/backdoor2023/backdoor/Summarize/dataset/${SUB_TASK}/splited
TRAIN_FILENAME=/home/backdoor2023/backdoor/Summarize/dataset/${SUB_TASK}/poison/${attack_way}/${trigger}_${poison_rate}_train.jsonl
# TRAIN_FILENAME=/home/backdoor2023/backdoor/preprocess/dataset/splited/train.jsonl
DEV_FILENAME=/home/backdoor2023/backdoor/Summarize/dataset/${SUB_TASK}/splited/test.jsonl
TEST_FILENAME=/home/backdoor2023/backdoor/Summarize/dataset/${SUB_TASK}/poison/${attack_way}/${trigger}_test.jsonl

CUDA_VISIBLE_DEVICES=${GPU} \
  python ${RUN_FN}  ${MULTI_TASK_AUG} \
  --train_filename ${TRAIN_FILENAME} \
  --dev_filename ${DEV_FILENAME} \
  --test_filename ${TEST_FILENAME} \
  --do_train --do_eval --do_test \
  --task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM}  \
  --num_train_epochs ${EPOCH} --warmup_steps ${WARMUP} --learning_rate ${LR}e-5 --patience ${PATIENCE} \
  --tokenizer_name=${TOKENIZER}  --model_name_or_path=${MODEL_PATH} --data_dir ${DATA_PATH} \
  --cache_path ${CACHE_DIR}  --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
  --save_last_checkpoints --always_save_model --res_dir ${RES_DIR} --res_fn ${RES_FN} \
  --train_batch_size ${BS} --eval_batch_size ${BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
  2>&1 | tee ${LOG}
