import argparse
import signal
import os
import subprocess
from pathlib import Path
import json

from utils import ( signal_handler,
    extract_audio_features, find_session_directories,
    extract_audio_segment, extract_robot_data_features, compute_robot_winning_rate )


def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Extract args
    parser = argparse.ArgumentParser(description='Run feature extraction pipeline')
    parser.add_argument('--input', type=str, required=True, help="Folder path with raw audios")
    parser.add_argument('--output', type=str, required=True, help="Folder path for output files")
    parser.add_argument('--window-len', type=int, required=True, help="Length of audio windows in seconds")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    session_dirs = find_session_directories(args.input)
    if len(session_dirs) == 0:
        print(f"No sessions found in {args.input}!")
        exit(1)
        
    print(f"Found {len(session_dirs)} audio files to process")
    print(f"Preprocessing the data...")
    
    for dir in session_dirs:
        audio_dir = f"{dir}/Audio"
        raw_audio_dir = f"{audio_dir}/raw"
        proc_audio_dir = f"{audio_dir}/processed"
        video_dir = f"{dir}/Videos/top_cam"
        
        if not os.path.exists(raw_audio_dir):
            os.makedirs(raw_audio_dir)
        if not os.path.exists(proc_audio_dir):
            os.makedirs(proc_audio_dir)

        has_video = os.path.exists(video_dir)

        # Video + audio session
        if has_video:
            list_videos = os.listdir(video_dir)
            video_name = [f for f in list_videos if f.endswith(".mp4")][0].split(".")[0]

            raw_wav = f"{raw_audio_dir}/{video_name}.wav"
            if not os.path.exists(raw_wav):
                extract_audio_segment(
                    f"{video_dir}/{video_name}.mp4",
                    raw_wav,
                )
            
            if not os.path.exists(f"{proc_audio_dir}/{video_name}.wav"):
                subprocess.run(
                    ['bash', 'src/data_pipeline/convert_audio.sh', '-o', proc_audio_dir, raw_wav],
                    check=True
            )
        # Audio only session
        else:
            print(f"{dir}: audio-only session (skipping video preprocessing)")
    
        # Convert all raw audio files
        for wav in os.listdir(raw_audio_dir):
            if not wav.endswith(".wav"):
                continue

            in_wav = f"{raw_audio_dir}/{wav}"
            out_wav = f"{proc_audio_dir}/{wav}"

            if not os.path.exists(out_wav):
                subprocess.run(
                    [
                        "bash",
                        "src/data_pipeline/convert_audio.sh",
                        "-o",
                        proc_audio_dir,
                        in_wav,
                    ],
                    check=True,
                )

    # Process audios sequentially
    processed_cnt = 0
    for dir in session_dirs:
        print(f"Processing {dir}...")
        
        processed_audio_dir = f"{dir}/Audio/processed"
        audio_name = os.listdir(processed_audio_dir)[0]
        audio_path = f"{processed_audio_dir}/{audio_name}"

        video_dir = f"{dir}/Videos/top_cam"
        has_video = os.path.exists(video_dir)

        if has_video:
            video_path = f"{video_dir}/{audio_name.split('.')[0]}.mp4"
        else:
            video_path = None
        
         # Extract robot data features only if video exists
        if has_video:
            robot_data_features = extract_robot_data_features(dir)
            print(f"Extracted robot data for {dir}...")
        else:
            robot_data_features = []
            print(f"{dir}: audio-only session (skipping robot logs)")
        
        # Extract audio features
        audio_features = extract_audio_features(audio_path, args.window_len)
        print(f"Extracted audio features for {dir}...")
        
        robot_speed_features = []
        for w in audio_features:
            if has_video:
                '''
                res = compute_robot_winning_rate(
                    video_path, 
                    w["window_start"], 
                    w["window_end"]
                )
                robot_speed_features.append({
                    "window_index": w["window_index"],
                    "window_start": w["window_start"],
                    "window_end": w["window_end"],
                    "avg_speed_cm_s": res.get("avg_speed_cm_s"),
                    "num_detections": res.get("num_detections"),
                    "winning_rate": res.get("winning_rate")
                })
                '''
                robot_speed_features.append({
                    "window_index": w["window_index"],
                    "window_start": w["window_start"],
                    "window_end": w["window_end"],
                    "avg_speed_cm_s": 0.0,
                    "num_detections": 0.0,
                    "winning_rate": 0.0
                })
            else:
                # Audio-only session
                robot_speed_features.append({
                    "window_index": w["window_index"],
                    "window_start": w["window_start"],
                    "window_end": w["window_end"],
                    "avg_speed_cm_s": None,
                    "num_detections": None,
                    "winning_rate": None,
                })

        print(f"Extracted robot speed features for {dir}...")
        
        # Save results
        dir_name = Path(dir).stem
        output_file = os.path.join(args.output, f"{dir_name}_features.json")
        
        with open(output_file, 'w') as f:
            json.dump({
                'session': dir_name,
                'base_window_length': args.window_len,
                'num_windows': len(audio_features),
                'audio_features': audio_features,
                'robot_data_features': robot_data_features,
                'robot_speed_features': robot_speed_features
            }, f, indent=2)
        
        print(f"✓ Completed {dir_name} -> {output_file}")
        processed_cnt += 1
    
    print(f"\n{'='*60}")
    print(f"Pipeline completed!")
    print(f"Successfully processed: {processed_cnt}/{len(session_dirs)} files")
    print(f"Output directory: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print(f"Pipeline interrupted by user (Ctrl+C)")
        print(f"{'='*60}")
        exit(130)