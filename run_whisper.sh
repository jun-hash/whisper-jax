#!/usr/bin/env bash
# 사용 예: ./run_whisper.sh <MODEL_NAME>
# 예: ./run_whisper.sh openai/whisper-large-v2

MODEL_NAME="$1"
INPUT_DIR="/mnt/data/v12"
OUTPUT_DIR="/mnt/data/v12/output"
LOG_FILE="/mnt/data/whisper_run_$(date +%Y%m%d_%H%M%S).log"

# 1. 필요한 디렉터리 생성 및 소유권/권한 변경 (sudo가 필요할 수 있음)
sudo mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
sudo chown -R $(whoami):$(whoami) "$INPUT_DIR" "$OUTPUT_DIR"
sudo chmod -R 755 "$INPUT_DIR" "$OUTPUT_DIR"

# 2. Whisper 작업 시작 로그 기록
echo "Starting Whisper TPU job with model ${MODEL_NAME}" | tee "$LOG_FILE"
start_time=$(date +%s)

# 3. Whisper TPU 작업 실행 (whisper_jax_infer.py 경로는 실제 위치로 수정)
python3 whisper_jax_infer.py "$MODEL_NAME" "$INPUT_DIR" "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

end_time=$(date +%s)
echo "Total job time: $((end_time - start_time)) seconds" | tee -a "$LOG_FILE"
