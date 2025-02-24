#!/usr/bin/env python3
import os
import sys
import time
import jax
import jax.numpy as jnp
from whisper_jax import FlaxWhisperPipline  # whisper-jax 라이브러리

def main():
    if len(sys.argv) < 3:
        print("Usage: python whisper_jax_infer.py <model_name> <input_dir> [<output_dir>]")
        sys.exit(1)
    
    model_name = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(input_dir, "output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading Whisper JAX model: {model_name} ...")
    start_compile = time.time()
    
    pipeline = FlaxWhisperPipline("openai/whisper-large-v2", 
                              dtype=jnp.bfloat16, 
                              batch_size=32)
    compile_time = time.time() - start_compile
    print(f"Model loaded and compiled in {compile_time:.2f} seconds.")
    
    # mp3 파일 리스트 (하위 디렉토리 포함)
    mp3_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".mp3"):
                mp3_files.append(os.path.join(root, file))
    
    total_files = len(mp3_files)
    print(f"Found {total_files} mp3 files in {input_dir}.")
    start_total = time.time()
    
    for idx, mp3 in enumerate(mp3_files):
        file_start = time.time()
        print(f"[{idx+1}/{total_files}] Processing: {mp3}")
        try:
            result = pipeline(mp3)
            text = result["text"]
        except Exception as e:
            print(f"Error processing {mp3}: {e}")
            continue
        
        base = os.path.splitext(os.path.basename(mp3))[0]
        output_path = os.path.join(output_dir, base + ".txt")
        with open(output_path, "w") as wf:
            wf.write(text + "\n")
        elapsed = time.time() - file_start
        print(f"Done in {elapsed:.2f} sec")
    
    total_elapsed = time.time() - start_total
    print(f"All done. Processed {total_files} files in {total_elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()