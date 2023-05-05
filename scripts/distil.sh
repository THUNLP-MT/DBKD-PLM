export CUDA_VISIBLE_DEVICES=0
SLAYER=4
TEACHER="bert-base-uncased"
STUDENT="bert-base-uncased"
VERSION="-final"
EPOCH=10

for TASK_NAME in "race-high"
do
# STRATEGY
# Ours ("DBKD"), Student CE ("none"), Standard KD ("standard"), Hard ("hard")
# Smoothing ("hard-smooth"), surrogate ("surrogate")
for STRATEGY in "DBKD"
do
for ALPHA in 0.2 0.5 0.7 # You can set to 0.7 as default
do
for TEMP in 1 2 4  # You can set to 1 as default
do
for SIGMA in 0.5 1 2 4 # You can set to 1 as default
do
for SEED in $(seq 1 5)
do

# Export path to teacher decisions
if [ ${STRATEGY} == "hard_aug" ] || [ ${STRATEGY} == "noise" ] || [ ${STRATEGY} == "DBKD" ] || [ ${STRATEGY} == "DBKD-old" ] || [ ${STRATEGY} == "DBKD-debias" ] || [ ${STRATEGY} == "DBKD-patient" ] || [ ${STRATEGY} == "noise-smooth" ] ;
then
  SLABEL=inference/${TASK_NAME}-noise${VERSION}-${TEACHER}
elif [ ${STRATEGY} == "surrogate" ];
then
  SLABEL=inference/${TASK_NAME}-surrogate-${TEACHER}
else
  SLABEL=inference/${TASK_NAME}-standard-${TEACHER}
fi

if [ ${TASK_NAME} == "race-all" ] || [ ${TASK_NAME} == "race-middle" ] || [ ${TASK_NAME} == "race-high" ] || [ ${TASK_NAME} == "dream" ] ;
then
  CONFIG="--do_eval --do_predict"
  if [ ${STRATEGY} == "hard_aug" ];
  then
    MAX_LEN=512
    ACCUM=8
    BSIZE=4
    LR=5e-5
  else
    MAX_LEN=512
    ACCUM=8
    BSIZE=4
    LR=5e-5
  fi
else
  CONFIG="--do_eval --do_predict"
  if [ ${STRATEGY} == "hard_aug" ];
  then
    MAX_LEN=128
    ACCUM=1
    BSIZE=32
    LR=2e-5
  else
    MAX_LEN=128
    ACCUM=1
    BSIZE=32
    LR=2e-5
  fi
fi

if [ ${TASK_NAME} == "mrpc" ] || [ ${TASK_NAME} == "qqp" ];
then
  METRIC="eval_combined_score"
elif [ ${TASK_NAME} == "cola" ];
then
  METRIC="eval_matthews_correlation"
else
  METRIC="eval_accuracy"
fi

OUTPUT_DIR=output/${TASK_NAME}-${STRATEGY}-${VERSION}-${EPOCH}-${SEED}-A${ALPHA}-P${SIGMA}-T${TEMP}-${TEACHER}-${SLAYER}L

# If you want to use GPT, use distil_gpt.py instead of distil.py
python distil.py \
  --model_name_or_path ${STUDENT} \
  --warmup_ratio 0.1  \
  --kd_alpha ${ALPHA} \
  --sigma ${SIGMA} \
  --seed $SEED \
  --kl_kd \
  --temperature ${TEMP} \
  --strategy ${STRATEGY} \
  --student_num_layers ${SLAYER} \
  --do_train \
  ${CONFIG} \
  --max_seq_length ${MAX_LEN} \
  --per_device_train_batch_size ${BSIZE}  \
  --per_device_eval_batch_size ${BSIZE}  --overwrite_output_dir \
  --save_total_limit 1 \
  --gradient_accumulation_steps ${ACCUM} \
  --logging_steps 10000  \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --learning_rate $LR \
  --overwrite_cache \
  --num_train_epochs ${EPOCH}  \
  --metric_for_best_model ${METRIC} \
  --task_name ${TASK_NAME} \
  --output_dir ${OUTPUT_DIR} \
  --soft_label_path ${SLABEL}/final.pt \
  --soft_label_dir ${SLABEL}
done
done
done
done
done
done
