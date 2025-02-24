#!/usr/bin/env bash
# 사용 예: ./run_whisper.sh <MODEL_NAME>
# 예: ./run_whisper.sh openai/whisper-large-v2

MODEL_NAME="$1"
INPUT_DIR="/mnt/data/v12"
OUTPUT_DIR="/mnt/data/v12/output"
LOG_FILE="/mnt/data/whisper_run_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTPUT_DIR"
echo "Starting Whisper TPU job with model ${MODEL_NAME}" | tee "$LOG_FILE"
start_time=$(date +%s)

python3 /path/to/whisper_jax_infer.py "$MODEL_NAME" "$INPUT_DIR" "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

end_time=$(date +%s)
echo "Total job time: $((end_time - start_time)) seconds" | tee -a "$LOG_FILE"