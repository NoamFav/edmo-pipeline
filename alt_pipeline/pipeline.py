import argparse
import signal
import os
import subprocess
from pathlib import Path
import json

from utils import cleanup_services, signal_handler, extract_audio_features


def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Extract args
    parser = argparse.ArgumentParser(description='Run feature extraction pipeline')
    parser.add_argument('--input', type=str, required=True, help="Folder path with raw audios")
    parser.add_argument('--output', type=str, required=True, help="Folder path for output files")
    parser.add_argument('--window-len', type=int, required=True, help="Length of audio windows in seconds")
    parser.add_argument('--asr-diar-file', type=str, default=None, help="Path to JSON file with pre-computed ASR and diarization results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Start services
    print("Starting services...")
    subprocess.run(['bash', 'alt_pipeline/start_services.sh'], check=True)
        
    # Preprocess audios
    print("Preprocessing audios...")
    subprocess.run(
        ['bash', 'src/data_pipeline/convert_audio.sh', '-o', 'data/processed', args.input],
        check=True
    )
    
    # Get all processed audio files
    processed_dir = 'data/processed'
    audio_files = [
        os.path.join(processed_dir, f) 
        for f in os.listdir(processed_dir) 
        if f.endswith(('.wav', '.mp3', '.m4a'))
    ]
    
    print(f"\nFound {len(audio_files)} audio files to process")
    print(f"Window length: {args.window_len} seconds\n")
    
    # Process audios sequentially
    processed_cnt = 0
    for audio_path in audio_files:
        all_features = extract_audio_features(audio_path, args.window_len, args.output, args.asr_diar_file)
        
        # Save results
        audio_name = Path(audio_path).stem
        output_file = os.path.join(args.output, f"{audio_name}_features.json")
        
        with open(output_file, 'w') as f:
            json.dump({
                'audio_file': audio_name,
                'base_window_length': args.window_len,
                'num_windows': len(all_features),
                'features': all_features
            }, f, indent=2)
        
        print(f"✓ Completed {audio_path} -> {output_file}")
        processed_cnt += 1
    
    print(f"\n{'='*60}")
    print(f"Pipeline completed!")
    print(f"Successfully processed: {processed_cnt}/{len(audio_files)} files")
    print(f"Output directory: {args.output}")
    print(f"{'='*60}")
    
    # Clean shutdown
    cleanup_services()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print(f"Pipeline interrupted by user (Ctrl+C)")
        print(f"{'='*60}")
        cleanup_services()
        exit(130)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"FATAL ERROR: {e}")
        print(f"{'='*60}")
        cleanup_services()
        exit(1)