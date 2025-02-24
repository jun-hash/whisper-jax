import jax.numpy as jnp
from google.cloud import storage
from whisper_jax import FlaxWhisperPipline

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

# 3) 오디오 다운로드
audio_data = download_audio_from_gcs(gcs_audio_uri)

# 4) Whisper JAX 추론
output = pipeline(audio_data, task="transcribe", return_timestamps=True)
print(output["text"])
print(output["chunks"])