#!/bin/bash
set -e

echo "============================================"
echo "1. Installing required pip packages..."
echo "============================================"
pip install git+https://github.com/sanchit-gandhi/whisper-jax.git \
            google-cloud-storage \
            librosa \

echo "============================================"
echo "2. Checking and installing ffmpeg if needed..."
echo "============================================"
if ! command -v ffmpeg &> /dev/null; then
  echo "ffmpeg not found. Installing ffmpeg via apt-get..."
  sudo apt-get update
  sudo apt-get install -y ffmpeg
else
  echo "ffmpeg is already installed."
fi

echo "============================================"
echo "3. Attaching disk 'mydisk' to TPU VM 'whisper-test2'..."
echo "============================================"
gcloud alpha compute tpus tpu-vm attach-disk whisper-test2 \
    --zone=europe-west4-a \
    --disk=mydisk \
    --mode=read-write

echo "============================================"
echo "4. Mounting the attached disk..."
echo "============================================"
# 마운트 포인트와 디바이스 이름을 설정합니다.
MOUNT_POINT="/mnt/disks/mydisk"
DEVICE="/dev/sdb"  # 실제 환경에 맞게 수정 필요

# 이미 마운트되어 있는지 확인
if mount | grep -q "$MOUNT_POINT"; then
  echo "Disk is already mounted at $MOUNT_POINT."
else
  echo "Mount point $MOUNT_POINT not found. Creating mount point and mounting the disk..."
  sudo mkdir -p "$MOUNT_POINT"
  sudo mount -o defaults "$DEVICE" "$MOUNT_POINT"
fi

echo "============================================"
echo "5. Setting ownership of the disk mount point..."
echo "============================================"
sudo chown -R "$(whoami)":"$(whoami)" "$MOUNT_POINT"

echo "============================================"
echo "Setup completed successfully."
echo "============================================"


sed -i 's/NamedShape/DShapedArray/g' ~/.local/lib/python3.10/site-packages/whisper_jax/layers.py
