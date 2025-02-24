import jax.numpy as jnp
from google.cloud import storage
from whisper_jax import FlaxWhisperPipline
import time
import jax
from contextlib import contextmanager

@contextmanager
def timer(name):
    """처리 시간을 측정하는 컨텍스트 매니저"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name} took {elapsed:.2f} seconds")

def get_tpu_memory_usage():
    """TPU 메모리 사용량 확인"""
    devices = jax.devices()
    memory_stats = []
    for d in devices:
        mem_stats = jax.device_get(jax.device_memory_stats(d))
        if mem_stats:
            memory_stats.append({
                'device': d.device_kind,
                'used_bytes': mem_stats['bytes_in_use'],
                'peak_bytes': mem_stats.get('peak_bytes_in_use', 0),
            })
    return memory_stats

# 1) Pipeline 초기화
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", 
                              dtype=jnp.bfloat16, 
                              batch_size=16)

def download_audio_from_gcs(gcs_uri):
    # gs://nari-librivox/test/sample_001.mp3 라고 가정
    client = storage.Client()
    path_parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1]

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    audio_data = blob.download_as_bytes()  # 바이너리
    return audio_data

# 2) 샘플 GCS 경로
gcs_audio_uri = "gs://nari-librivox/test/autobiographyoflorddouglas_01_douglas.mp3/autobiographyoflorddouglas_01_douglas_0001.mp3"

def process_audio(gcs_audio_uri):
    """오디오 처리 및 TPU 성능 모니터링"""
    # TPU 초기 상태 확인
    print("Initial TPU memory state:")
    print(get_tpu_memory_usage())
    
    with timer("Audio download"):
        audio_data = download_audio_from_gcs(gcs_audio_uri)
    
    with timer("Whisper inference"):
        output = pipeline(audio_data, task="transcribe", return_timestamps=True)
    
    # TPU 사용 후 상태 확인
    print("\nFinal TPU memory state:")
    print(get_tpu_memory_usage())
    
    return output

if __name__ == "__main__":
    # TPU 디바이스 정보 출력
    print("Available devices:", jax.devices())
    
    output = process_audio(gcs_audio_uri)
    print("\nTranscription:", output["text"])
    print("\nChunks:", output["chunks"])