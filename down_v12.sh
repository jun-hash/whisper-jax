#!/usr/bin/env bash
# 사용 예: ./download_v12.sh
# GCS 버킷과 폴더 경로를 수정하세요.

BUCKET_NAME="nari-librivox"
FOLDER_PATH="v12"
DEST_DIR="/mnt/data/v12"

# 대상 디렉터리 생성 및 소유권/권한 수정
sudo mkdir -p "$DEST_DIR"
sudo chown -R $(whoami):$(whoami) "$DEST_DIR"
sudo chmod -R 755 "$DEST_DIR"

echo "Downloading data from gs://$BUCKET_NAME/$FOLDER_PATH/ to $DEST_DIR ..."
gsutil -m cp -r "gs://$BUCKET_NAME/$FOLDER_PATH/" "$DEST_DIR"
echo "Download complete."