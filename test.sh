#!/usr/bin/env bash
# usage: ./chunk_test.sh <BUCKET_PATH> <MODEL_NAME>
# ex) ./chunk_test.sh nari-librivox/v12 openai/whisper-large-v2

BUCKET_PATH="$1"
MODEL_NAME="$2"

AUDIO_DIR="/mnt/diskaudio/local_audio"
OUTPUT_DIR="/mnt/diskaudio/local_output"
LOG_FILE="/mnt/diskaudio/chunk_test_$(date +%Y%m%d_%H%M%S).log"

sudo mkdir -p "$AUDIO_DIR" "$OUTPUT_DIR"

# (임시) 이번에는 "테스트"이므로, v12 폴더의 mp3 중 5%만
# 실제로는 "파일 리스트"를 만들어서 부분만 다운 or 폴더를 쪼개는 방식을 써야 함.
# 여기서는 간단히 "max 100 files" 만 받는 예시
echo "=== chunk_test start ===" | tee -a "$LOG_FILE"
START_TIME=$(date +%s)

# 1) 파일 목록을 일부 가져오기 (gsutil ls)
#    -> head -n 100 로 상위 100개만 다운
gsutil ls "gs://my-bucket/${BUCKET_PATH}/*.mp3" | head -n 100 > /tmp/filelist.txt

# 2) 다운로드
echo "[Download phase]" | tee -a "$LOG_FILE"
cat /tmp/filelist.txt | gsutil -m cp -I "$AUDIO_DIR" 2>&1 | tee -a "$LOG_FILE"

# 3) Whisper 처리
echo "[Whisper phase]" | tee -a "$LOG_FILE"
python3 whisper_jax_infer.py "$MODEL_NAME" "$AUDIO_DIR" "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

# 4) 결과 업로드
echo "[Upload phase]" | tee -a "$LOG_FILE"
gsutil -m cp "$OUTPUT_DIR"/*.txt "gs://my-bucket/${BUCKET_PATH}/processed_test/" 2>&1 | tee -a "$LOG_FILE"

# 5) 로컬 정리 (Spot 중단되면 어차피 데이터 날아가지만)
rm -rf "$AUDIO_DIR"/*.mp3
rm -rf "$OUTPUT_DIR"/*.txt

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== DONE chunk_test in $ELAPSED seconds ===" | tee -a "$LOG_FILE"

# TPU Util, CPU load, etc.
top -b -n1 | head -n 20 >> "$LOG_FILE"
