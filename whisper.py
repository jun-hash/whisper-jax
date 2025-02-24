import jax.numpy as jnp
from whisper_jax import FlaxWhisperPipline

# 1) 파이프라인 인스턴스화
#    - bfloat16 사용 (TPU 환경)
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", 
                              dtype=jnp.bfloat16,
                              batch_size=16)

# 2) 로컬에 있는 테스트 오디오 파일(예: sample.mp3) 로드 후 추론
output = pipeline("sample.mp3", task="transcribe", return_timestamps=True)

print(output["text"])
print(output["chunks"])
