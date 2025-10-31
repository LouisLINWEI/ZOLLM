#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="/home/wlin23/ZO-LLM"
LOG_DIR="${SCRIPT_DIR}/log"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_zosgd_copa_opt-6.7b.log"

mkdir -p "${LOG_DIR}"

source /home/wlin23/anaconda3/etc/profile.d/conda.sh
conda activate zollm

cd "${SCRIPT_DIR}"

nohup python zo-bench/run.py \
  --trainer=zo_sgd \
  --model_name=facebook/opt-6.7b \
  --task_name=Copa \
  --train_as_classification \
  --perturbation_mode=two_side \
  --zo_eps=1e-7 \
  --learning_rate=1e-6 \
  --num_train_epochs=5 \
  --per_device_train_batch_size=16 \
  --load_best_model_at_end \
  --evaluation_strategy=steps \
  --eval_steps=500 \
  --save_steps=500 \
  --max_steps=20000 \
  --save_total_limit=1 \
  --logging_steps=10 \
  --num_eval=1000 \
  --num_train=1000 \
  --num_dev=100 \
  --train_set_seed=0 \
  --lr_scheduler_type=constant \
  --output_dir=result/Copa-ft-zo-sgd \
  > "${LOG_FILE}" 2>&1 &

echo "已启动训练（AGZO Copa），日志存储于: ${LOG_FILE}"


