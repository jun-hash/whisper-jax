import os
import json
import jax
import jax.numpy as jnp
import time
import librosa
from whisper_jax import FlaxWhisperPipline
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", 
                       default=os.path.abspath("/mnt/data/v12_sample"))
    parser.add_argument("--output-dir",
                       default="./output")
    return parser.parse_args()

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
    # Create log file
    log_file = os.path.join(output_folder, f"transcription_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
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
    print(f"mp3_files[0]: {mp3_files}")
    mp3_files.sort()

    total_audio_duration = 0
    total_processing_time = 0

    for idx, mp3_file in enumerate(mp3_files, 1):
        input_path = os.path.join(input_folder, mp3_file)
        output_filename = os.path.splitext(mp3_file)[0] + ".json"
        output_path = os.path.join(output_folder, output_filename)

        # Skip if output already exists
        if os.path.exists(output_path):
            print(f"Skipping {mp3_file} because {output_filename} already exists.")
            continue

        # Get audio duration
        audio_duration = librosa.get_duration(path=input_path)
        total_audio_duration += audio_duration

        print(f"[{idx}/{len(mp3_files)}] Transcribing {mp3_file} (duration: {audio_duration:.2f}s)...")
        
        # Measure processing time
        start_time = time.time()
        outputs = pipeline(input_path, task=task, return_timestamps=return_timestamps)
        processing_time = time.time() - start_time
        total_processing_time += processing_time

        text = outputs["text"]
        chunks = outputs["chunks"]

        # 6. Prepare JSON result with timing information
        result = {
            "file_name": mp3_file,
            "text": text,
            "chunks": chunks,
            "audio_duration": audio_duration,
            "processing_time": processing_time,
            "realtime_factor": processing_time / audio_duration
        }

        # 7. Write result to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Log processing statistics
        log_message = (
            f"File: {mp3_file}\n"
            f"Audio Duration: {audio_duration:.2f}s\n"
            f"Processing Time: {processing_time:.2f}s\n"
            f"Realtime Factor: {processing_time/audio_duration:.2f}x\n"
            f"----------------------------------------\n"
        )
        print(log_message)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_message)

    # Log final statistics
    final_stats = (
        f"\nFinal Statistics:\n"
        f"Total Audio Duration: {total_audio_duration:.2f}s\n"
        f"Total Processing Time: {total_processing_time:.2f}s\n"
        f"Average Realtime Factor: {total_processing_time/total_audio_duration:.2f}x\n"
    )
    print(final_stats)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(final_stats)

if __name__ == "__main__":
    args = parse_args()
    transcribe_folder(
        input_folder=args.input_dir,
        output_folder=args.output_dir,
        model_id="openai/whisper-large-v2",
        dtype=jnp.bfloat16,    
        batch_size=64,         # 증가된 배치 사이즈
        shard_params=True,     
        task="transcribe",     
        return_timestamps=True  
    )