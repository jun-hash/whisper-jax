import os
import json
import jax
import jax.numpy as jnp
from whisper_jax import FlaxWhisperPipline

def transcribe_folder(
    input_folder: str,
    output_folder: str,
    model_id: str = "openai/whisper-large-v2",
    dtype=jnp.bfloat16,
    batch_size: int = 32,
    shard_params: bool = True,
    task: str = "transcribe",
    return_timestamps: bool = True
):
    """
    Transcribe all MP3 files in `input_folder` using Whisper JAX on TPU v3-8,
    writing outputs to JSON in `output_folder`.
    """

    # 1. Instantiate Whisper JAX pipeline with optimized settings
    pipeline = FlaxWhisperPipline(
        model_id,
        dtype=jnp.bfloat16,  # TPU에서는 bfloat16이 최적
        batch_size=64        # TPU v3-8에서는 64가 좋은 성능을 보임
    )

    # 2. Basic sharding configuration for TPU v3-8
    if shard_params:
        logical_axis_rules_dp = (
            ("batch", "data"),
            ("mlp", None),
            ("heads", None),
            ("vocab", None),
            ("embed", None),
            ("embed", None),
            ("joined_kv", None),
            ("kv", None),
            ("length", None),
            ("num_mel", None),
            ("channels", None),
        )
        pipeline.shard_params(num_mp_partitions=1, logical_axis_rules=logical_axis_rules_dp)

    # 3. Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # 4. Warm-up JIT compilation
    #    If you have a small "dummy" audio file, you can use it to compile once.
    #    For example:
    # warmup_file = os.path.join(input_folder, "some_small_file.mp3")
    # if os.path.isfile(warmup_file):
    #     _ = pipeline(warmup_file, task=task, return_timestamps=return_timestamps)

    # 5. Transcribe each MP3 in the folder
    mp3_files = [f for f in os.listdir(input_folder) if f.endswith(".mp3")]
    mp3_files.sort()

    for idx, mp3_file in enumerate(mp3_files, 1):
        input_path = os.path.join(input_folder, mp3_file)
        output_filename = os.path.splitext(mp3_file)[0] + ".json"
        output_path = os.path.join(output_folder, output_filename)

        # Skip if output already exists (helpful if you resume after interruption)
        if os.path.exists(output_path):
            print(f"Skipping {mp3_file} because {output_filename} already exists.")
            continue

        print(f"[{idx}/{len(mp3_files)}] Transcribing {mp3_file} ...")
        
        outputs = pipeline(input_path, task=task, return_timestamps=return_timestamps)
        text = outputs["text"]
        chunks = outputs["chunks"]

        # 6. Prepare JSON result
        result = {
            "file_name": mp3_file,
            "text": text,
            "chunks": chunks   # chunk-level timestamps, if return_timestamps=True
        }

        # 7. Write result to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f" -> Saved JSON to: {output_path}")

if __name__ == "__main__":
    # Example usage
    INPUT_FOLDER = "/mnt/data/v12"   # all your 15-30s MP3s here
    OUTPUT_FOLDER = "./output"
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    transcribe_folder(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        model_id="openai/whisper-large-v2",
        dtype=jnp.bfloat16,    
        batch_size=64,         # 증가된 배치 사이즈
        shard_params=True,     
        task="transcribe",     
        return_timestamps=True  
    )