GPU=3
# Just use the final version
VERSION="-final"
MODEL="bert-base-uncased" # teacher model is bert-base

# TASK_NAME
# For example: "mrpc" "sst2" "qnli" "qqp" "mnli" "cola" "dream" "race-xxx"...
for TASK_NAME in "race-high"
do

# STRATEGY
# Our method: use "noise" --> to generate teacher predictions on augmented data
# Standard KD: use "noise" --> to generate teacher predictions on original data
# surrogate: use "noise" --> to generate surrogate teacher predictions on original data
for STRATEGY in "noise"
do

if [ ${TASK_NAME} == "race-all" ] || [ ${TASK_NAME} == "race-middle" ] || [ ${TASK_NAME} == "race-high" ] ;
then
  MAX_LEN=512
else
  MAX_LEN=128
fi

if [ ${STRATEGY} == "noise" ]
then
  VERSION=${VERSION}
else
  VERSION=""
fi

if [ ${STRATEGY} == "surrogate" ]
then
  POSTFIX="-surrogate-4"
  TLAYER=4
else
  POSTFIX=""
  TLAYER=0
fi

TEACHER_PATH=ckpts/${TASK_NAME}-${MODEL}${POSTFIX}

mkdir inference/${TASK_NAME}-${STRATEGY}${VERSION}-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python teacher_inference.py \
  --model_name_or_path ${TEACHER_PATH} \
  --fp16  \
  --just_inference \
  --ablation_sr \
  --noise_strategy eda \
  --inf_strategy ${STRATEGY} \
  --max_seq_length ${MAX_LEN} \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --teacher_num_layers ${TLAYER} \
  --save_total_limit 1 \
  --overwrite_cache \
  --logging_steps 10 \
  --eval_steps 500 \
  --evaluation_strategy steps --warmup_ratio 0.05 --overwrite_output_dir  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir inference/${TASK_NAME}-${STRATEGY}${VERSION}-$MODEL \
  --task_name $TASK_NAME
python inference/reorganize.py inference/${TASK_NAME}-${STRATEGY}${VERSION}-$MODEL/idx_list.pt  \
                     inference/${TASK_NAME}-${STRATEGY}${VERSION}-$MODEL/logits_list.pt \
                     inference/${TASK_NAME}-${STRATEGY}${VERSION}-$MODEL/final.pt \
                     ${STRATEGY}
done
done
