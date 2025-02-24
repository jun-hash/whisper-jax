#!/usr/bin/env bash
# v12 데이터셋에서 1%만 샘플링하여 다운로드

BUCKET_NAME="nari-librivox"
FOLDER_PATH="v12"
DEST_DIR="/mnt/data/v12_sample"
TMP_FILE="/tmp/v12_files.txt"
SAMPLE_FILE="/tmp/v12_sample_files.txt"

# 1. 대상 디렉터리 생성
sudo mkdir -p "$DEST_DIR"
sudo chown -R $(whoami):$(whoami) "$DEST_DIR"
sudo chmod -R 755 "$DEST_DIR"

echo "Getting file list from gs://$BUCKET_NAME/$FOLDER_PATH/"
# 2. 전체 MP3 파일 목록 가져오기
gsutil ls "gs://$BUCKET_NAME/$FOLDER_PATH/**/*.mp3" > "$TMP_FILE"

# 3. 전체 파일 수 확인 및 1% 계산
total_files=$(wc -l < "$TMP_FILE")
sample_size=$((total_files / 100))
echo "Total files: $total_files"
echo "Sampling $sample_size files (1%)"

# 4. 무작위로 1% 샘플링
shuf -n "$sample_size" "$TMP_FILE" > "$SAMPLE_FILE"

# 5. 샘플링된 파일들만 다운로드
echo "Downloading sampled files to $DEST_DIR ..."
mkdir -p "$DEST_DIR"
cat "$SAMPLE_FILE" | gsutil -m cp -I "$DEST_DIR"

# 6. 정리
rm "$TMP_FILE" "$SAMPLE_FILE"

echo "Download complete. Sampled files are in $DEST_DIR"
echo "Downloaded files count: $(ls "$DEST_DIR"/*.mp3 | wc -l)" 