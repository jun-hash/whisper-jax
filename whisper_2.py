import os
import json
import jax
import jax.numpy as jnp
import time
import librosa
import numpy as np
from whisper_jax import FlaxWhisperPipline
import argparse
from datetime import datetime
from itertools import islice


def load_audio(file_path):
    """Load and preprocess audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dict containing audio array, sampling rate, and duration
    """
    # Load audio file and convert to mono
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    
    # Ensure audio is 1D (mono)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
        
    # Calculate duration from loaded audio
    duration = len(audio) / sr
    
    return {
        "array": audio, 
        "sampling_rate": sr,
        "duration": duration
    }


def chunked_iterable(iterable, batch_size=32):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, batch_size))
        if not chunk:
            break
        yield chunk


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

def transcribe_all_in_one(input_folder: str, output_folder: str):
    """Transcribe all MP3 files in input_folder and save results to output_folder."""
    # Create output directory and log file
    os.makedirs(output_folder, exist_ok=True)
    log_file = os.path.join(output_folder, f"transcription_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # 1) Get list of MP3 files
    mp3_paths = [
        os.path.join(input_folder, f) 
        for f in os.listdir(input_folder) if f.endswith(".mp3")
    ]
    mp3_paths.sort()
    print(f"Found {len(mp3_paths)} MP3 files to process")

    # 2) Initialize Whisper JAX pipeline
    pipeline = FlaxWhisperPipline(
        "openai/whisper-large-v2",
        dtype=jnp.bfloat16,
        batch_size=32,
    )

    # Configure sharding
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

    # 3) Process files in batches
    total_audio_duration = 0
    start_time = time.time()
    all_results = []
    
    for i, batch_paths in enumerate(chunked_iterable(mp3_paths, 32)):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"Processing batch {i+1} with {len(batch_paths)} files...")
                
                # Load and preprocess all audio files in batch
                batch_audio_data = []
                batch_durations = []
                max_length = 0
                
                for audio_path in batch_paths:
                    audio_data = load_audio(audio_path)
                    audio_array = audio_data["array"]
                    
                    # Ensure audio is mono (single channel)
                    if len(audio_array.shape) > 1:
                        audio_array = audio_array.mean(axis=1)
                    
                    batch_audio_data.append(audio_array)
                    batch_durations.append(audio_data["duration"])
                    max_length = max(max_length, len(audio_array))
                
                # Pad all audio arrays to same length
                padded_audio = []
                for audio in batch_audio_data:
                    padding_length = max_length - len(audio)
                    padded = np.pad(audio, (0, padding_length))
                    padded_audio.append(padded)
                
                # Stack arrays and ensure shape is correct
                batch_array = np.stack(padded_audio)
                if len(batch_array.shape) != 2:  # Should be (batch_size, sequence_length)
                    raise ValueError(f"Unexpected audio shape: {batch_array.shape}")
                
                # Create batch input
                batch_input = {
                    "array": batch_array,
                    "sampling_rate": 16000  # Whisper expects 16kHz
                }
                
                # Process batch
                batch_outputs = pipeline(batch_input, return_timestamps=True)
                
                # Process results
                batch_results = []
                for audio_path, duration, output in zip(batch_paths, batch_durations, batch_outputs["text"]):
                    result = {
                        "file_name": os.path.basename(audio_path),
                        "text": output,
                        "audio_duration": duration
                    }
                    batch_results.append(result)
                    total_audio_duration += duration
                
                all_results.extend(batch_results)
                break
                
            except Exception as e:
                retry_count += 1
                print(f"Error processing batch {i+1}: {str(e)}")
                print(f"Retry {retry_count} of {max_retries}")
                time.sleep(5)
                
        if retry_count == max_retries:
            print(f"Failed to process batch {i+1} after {max_retries} retries")
            with open(os.path.join(output_folder, "failed_batches.txt"), "a") as f:
                f.write(f"Batch {i+1}: {batch_paths}\n")

    total_processing_time = time.time() - start_time

    # 4) Save results and log statistics
    for result in all_results:
        output_filename = os.path.splitext(result["file_name"])[0] + ".json"
        output_path = os.path.join(output_folder, output_filename)
        
        # Add timing information
        result["processing_time"] = total_processing_time / len(mp3_paths)
        result["realtime_factor"] = result["processing_time"] / result["audio_duration"]
        
        # Save to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Log processing statistics
        log_message = (
            f"File: {result['file_name']}\n"
            f"Audio Duration: {result['audio_duration']:.2f}s\n"
            f"Avg Processing Time: {result['processing_time']:.2f}s\n"
            f"Realtime Factor: {result['realtime_factor']:.2f}x\n"
            f"----------------------------------------\n"
        )
        print(log_message)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_message)

    # Log final statistics
    final_stats = (
        f"\nFinal Statistics:\n"
        f"Total Files Processed: {len(mp3_paths)}\n"
        f"Total Audio Duration: {total_audio_duration:.2f}s\n"
        f"Total Processing Time: {total_processing_time:.2f}s\n"
        f"Average Realtime Factor: {total_processing_time/total_audio_duration:.2f}x\n"
    )
    print(final_stats)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(final_stats)

if __name__ == "__main__":
    args = parse_args()
    transcribe_all_in_one(
        input_folder=args.input_dir,
        output_folder=args.output_dir
    )