GPU=0
TASK_NAME="race-high"
MODEL="bert-base-uncased"
export CUDA_VISIBLE_DEVICES=$GPU
SEED=0

if [ ${TASK_NAME} == "race-all" ] || [ ${TASK_NAME} == "race-middle" ] || [ ${TASK_NAME} == "race-high" ] ;
then
  MAX_LEN=512
  SUBMIT="python -m torch.distributed.launch --nproc_per_node 2 --master_port 44144"
  BSIZE=2
  ACCUM=8
  LR=2e-5
  EPOCH=4
elif [ ${TASK_NAME} == "dream" ] ;
then
  MAX_LEN=512
  SUBMIT="python -m torch.distributed.launch --nproc_per_node 2 --master_port 44144"
  BSIZE=2
  ACCUM=8
  LR=2e-5
  EPOCH=4
else
  MAX_LEN=128
  SUBMIT="python"
  BSIZE=32
  ACCUM=1
  LR=2e-5
  EPOCH=4
fi

# If you want to use GPT, use run_glue_gpt.py instead of run_glue.py
${SUBMIT} run_glue.py \
  --model_name_or_path $MODEL \
  --do_train  \
  --do_eval \
  --fp16 \
  --seed ${SEED} \
  --max_seq_length ${MAX_LEN} \
  --gradient_accumulation_steps ${ACCUM} \
  --per_device_train_batch_size ${BSIZE} \
  --per_device_eval_batch_size 16 \
  --save_total_limit 1 \
  --evaluation_strategy epoch --warmup_ratio 0.1 --overwrite_output_dir  \
  --learning_rate ${LR} \
  --num_train_epochs ${EPOCH} \
  --output_dir ckpts/${TASK_NAME}-$MODEL \
  --task_name $TASK_NAME
