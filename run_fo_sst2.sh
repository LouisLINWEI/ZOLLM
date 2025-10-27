#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="/home/wlin23/ZO-LLM"
LOG_DIR="${SCRIPT_DIR}/log"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_fo_sst2_opt1.3b.log"

mkdir -p "${LOG_DIR}"

source /home/wlin23/anaconda3/etc/profile.d/conda.sh
conda activate zollm

cd "${SCRIPT_DIR}"

nohup python zo-bench/run.py \
  --trainer=sgd \
  --model_name=facebook/opt-1.3b \
  --task_name=SST2 \
  --train_as_classification \
  --num_train_epochs=5 \
  --per_device_train_batch_size=8 \
  --learning_rate=1e-5 \
  --weight_decay=0 \
  --lr_scheduler_type=constant \
  --load_best_model_at_end \
  --evaluation_strategy=steps \
  --eval_steps=20 \
  --save_steps=20 \
  --max_steps=500 \
  --logging_steps=10 \
  --num_eval=1000 \
  --num_train=1000 \
  --num_dev=500 \
  --train_set_seed=0 \
  --output_dir=result/SST2-ft-fo-sgd \
  > "${LOG_FILE}" 2>&1 &

echo "已启动训练（FO），日志存储于: ${LOG_FILE}"


