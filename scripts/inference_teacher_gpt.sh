GPU=5
# v7 5 times sampling
NVERSION="-surrogate-4"
#TASK_NAME="sst2"
MODEL="gpt2" # bert-large
#for TASK_NAME in "mrpc" "sst2" "qnli" "qqp" "mnli" "cola" "dream"
for TASK_NAME in "race-high"
#for TASK_NAME in "rte" "mrpc" "cola" "sst2" "qnli" "dream"
do
# "standard" "noise" "surrogate"
for STRATEGY in "surrogate"
#for STRATEGY in "standard"
do

MAX_LEN=1024

if [ ${STRATEGY} == "noise" ]
then
  VERSION=${NVERSION}
else
  VERSION=""
fi

#  --model_name_or_path ckpts/${TASK_NAME}-${MODEL} \
mkdir inference/${TASK_NAME}-${STRATEGY}${VERSION}-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python teacher_inference_gpt.py \
  --model_name_or_path ckpts/${TASK_NAME}-${MODEL}-surrogate-4/ \
  --fp16  \
  --just_inference \
  --noise_strategy eda \
  --inf_strategy ${STRATEGY} \
  --max_seq_length ${MAX_LEN} \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --save_total_limit 1 \
  --overwrite_cache \
  --logging_steps 10 \
  --eval_steps 500 \
  --evaluation_strategy steps --warmup_ratio 0.05 --overwrite_output_dir  \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir inference/${TASK_NAME}-${STRATEGY}${VERSION}-$MODEL \
  --task_name $TASK_NAME

if [ ${STRATEGY} == "surrogate" ]
then
  cp inference/${TASK_NAME}-${STRATEGY}${VERSION}-$MODEL/logits_list.pt \
      inference/${TASK_NAME}-${STRATEGY}${VERSION}-$MODEL/final.pt
else
  python inference/reorganize_gpt.py inference/${TASK_NAME}-${STRATEGY}${VERSION}-$MODEL/idx_list.pt  \
                     inference/${TASK_NAME}-${STRATEGY}${VERSION}-$MODEL/ans_list.pt \
                     inference/${TASK_NAME}-${STRATEGY}${VERSION}-$MODEL/final.pt \
                     ${STRATEGY}
fi

done
done
