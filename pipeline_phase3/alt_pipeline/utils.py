import os
import subprocess
import json
import requests
import librosa
import numpy as np
import signal
import sys
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch


# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import processors for direct function calls
from src.python_services.emotion.processor import detect_emotion, get_processor as get_emotion_processor
from src.python_services.nlp.processor import analyze_sentiment, get_processor as get_nlp_processor
from src.python_services.nonverb_features.processor import (
    calculate_basic_metrics,
    calculate_pitch_features,
    calculate_loudness_features,
    calculate_tempo_features
)
from src.python_services.asr.processor import transcribe
from src.python_services.diarization.processor import diarize
from src.python_services.robot_data.processor import parse_logs, extract_features

# Service endpoints
SERVICES = {
    'asr': 'http://127.0.0.1:8001',
    'diarization': 'http://127.0.0.1:8002',
    'emotion': 'http://127.0.0.1:8003',
    'nlp': 'http://127.0.0.1:8004',
    'nonverb': 'http://127.0.0.1:8005',
    'robot_data': 'http://127.0.0.1:8006',
    'robot_speed': 'http://127.0.0.1:8007'
}




def find_session_directories(input_dir: str) -> list[str]:
    """Traverse directory tree and find all directories containing 'session.log'."""
    session_dirs = []
    
    # Walk through all directories
    for root, dirs, files in os.walk(input_dir):
        if 'session.log' in files:
            # Convert to relative path from input_dir
            session_dirs.append(root)
    
    return session_dirs


def extract_audio_segment(video_path, output_path):
    """Extract audio segment from video using FFmpeg."""
    cmd = [
        'ffmpeg', '-i',
        video_path, output_path
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    

def extract_robot_data_features(session_dir):
    """Extract robot interaction features using processor directly."""
    timeline_csv = f"{session_dir}/robot_data.csv"
    
    # Parse logs and create timeline CSV
    parse_logs(session_dir, timeline_csv)
    
    # Extract features from timeline
    robot_interaction_features = extract_features(timeline_csv)
    
    return robot_interaction_features

def run_asr_and_diarization(audio_path: str) -> Dict[str, Any]:
    print(f"  Running ASR and diarization on full audio...")

    results = {}

    # Use processors directly instead of HTTP requests
    results['asr'] = transcribe(audio_path)

    # Load audio manually for pyannote
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    waveform = torch.from_numpy(waveform).unsqueeze(0)  # (1, time)

    results['diarization'] = diarize({
        "waveform": waveform,
        "sample_rate": sr
    })

    return results
"""
def run_asr_and_diarization(audio_path: str) -> Dict[str, Any]:
    """ # Run ASR and diarization on the full audio file using processors directly.
"""
    print(f"  Running ASR and diarization on full audio...")
    
    results = {}
    
    # Use processors directly instead of HTTP requests
    results['asr'] = transcribe(audio_path)
    results['diarization'] = diarize(audio_path)
    
    return results
"""

def get_window_segments(full_asr: Dict, full_diarization: Dict, window_start: float, window_end: float) -> tuple[Dict[str, Any], float]:
    """
    Extract ASR and diarization segments for the window with flexible boundaries.
    If a speaker's turn starts within the window (>= window_start and < window_end), 
    include the entire turn even if it extends beyond window_end.
    Returns: (window_segments, actual_window_end)
    """
    window_asr_segments = []
    window_diar_segments = []
    actual_end = window_end
    
    # Process ASR segments
    for seg in full_asr['segments']:
        if window_start <= seg['start'] < window_end:
            # Include this segment if it starts within the window
            window_asr_segments.append(seg)
            # Extend window boundary if segment extends beyond
            if seg['end'] > actual_end:
                actual_end = seg['end']
        elif seg['start'] >= window_end:
            # Stop when we reach segments that start after window
            break
    
    # Process diarization segments (speaker turns)
    for seg in full_diarization['segments']:
        if window_start <= seg['start'] < window_end:
            # Include this speaker turn if it starts within the window
            window_diar_segments.append(seg)
            # Extend window boundary if turn extends beyond
            if seg['end'] > actual_end:
                actual_end = seg['end']
        elif seg['start'] >= window_end:
            # Stop when we reach turns that start after window
            break
    
    return {
        'asr': {'segments': window_asr_segments, 'language': full_asr['language']},
        'diarization': {'segments': window_diar_segments, 'num_speakers': full_diarization['num_speakers']}
    }, actual_end


def extract_features_for_window(window: np.ndarray, window_segments: Dict[str, Any], window_index: int, processors: dict, sr: int = 16000) -> Dict[str, Any]:
    """Extract all features in parallel within the window."""
    print(f"    Processing window {window_index}...")
    
    features = {}
    
    asr_result = window_segments['asr']
    diarization_result = window_segments['diarization']
    full_text = ' '.join([seg['text'] for seg in asr_result['segments']])
    
    if full_text.strip():
        conv_length = len(window) / sr
        
        # Run all 6 feature extractions in parallel
        with ThreadPoolExecutor(max_workers=6) as executor:
            emotion_future = executor.submit(detect_emotion, full_text, processors["emotion_processor"])
            sentiment_future = executor.submit(analyze_sentiment, full_text, processors["nlp_processor"])
            basic_future = executor.submit(calculate_basic_metrics, diarization_result, conv_length)
            pitch_future = executor.submit(calculate_pitch_features, diarization_result, window, sr)
            loudness_future = executor.submit(calculate_loudness_features, diarization_result, window, sr)
            tempo_future = executor.submit(calculate_tempo_features, diarization_result, window, sr)
            
            # Gather results
            features['emotion'] = emotion_future.result()
            features['nlp'] = {'sentiment': sentiment_future.result()}
            features['nonverbal'] = {
                'basic_metrics': basic_future.result(),
                'pitch': pitch_future.result(),
                'loudness': loudness_future.result(),
                'tempo': tempo_future.result()
            }
    else:
        features['emotion'] = None
        features['nlp'] = None
        features['nonverbal'] = None
    
    return features


def extract_audio_features(audio_path: str, window_len: int, sr: int = 16000, max_workers: int = 4) -> str:
    """Process windows in parallel."""
    print(f"Processing {audio_path}...")
    
    # Step 1: Check for cached ASR/diarization results
    audio_dir = os.path.dirname(audio_path)
    asr_diar_cache = os.path.join(audio_dir, 'asr_diar.json')
    
    if os.path.exists(asr_diar_cache):
        print(f"  Loading cached ASR/diarization from {asr_diar_cache}...")
        with open(asr_diar_cache, 'r', encoding='utf-8') as f:
            full_results = json.load(f)
    else:
        print(f"  No cache found, running ASR and diarization...")
        full_results = run_asr_and_diarization(audio_path)
        
        # Save results to cache
        print(f"  Saving ASR/diarization to {asr_diar_cache}...")
        with open(asr_diar_cache, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    # Step 2: Load audio
    y, _ = librosa.load(audio_path, sr=sr)
    window_samples = window_len * sr
    total_duration = len(y) / sr
    
    # Step 3: Prepare all windows
    windows_to_process = []
    window_index = 0
    current_position = 0
    
    while current_position < len(y):
        window_start_time = current_position / sr
        window_end_time = min((current_position + window_samples) / sr, total_duration)
        
        window_segments, actual_end_time = get_window_segments(
            full_results['asr'], 
            full_results['diarization'],
            window_start_time,
            window_end_time
        )
        
        actual_end_samples = int(actual_end_time * sr)
        actual_end_samples = min(actual_end_samples, len(y))
        
        if actual_end_samples - current_position > 0:
            windows_to_process.append({
                'window': y[current_position:actual_end_samples],
                'segments': window_segments,
                'index': window_index,
                'start': window_start_time,
                'end': actual_end_time
            })
            window_index += 1
        
        current_position = actual_end_samples
    
    # Step 4: Process windows in parallel
    processors = {
        "nlp_processor": get_nlp_processor(),
        "emotion_processor": get_emotion_processor()
    }
    all_features = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                extract_features_for_window,
                w['window'], w['segments'], w['index'], processors, sr
            ): w for w in windows_to_process
        }
        
        for future in as_completed(futures):
            w = futures[future]
            features = future.result()
            features['window_index'] = w['index']
            features['window_start'] = w['start']
            features['window_end'] = w['end']
            features['window_duration'] = w['end'] - w['start']
            features['base_window_len'] = window_len
            all_features.append(features)
    
    # Sort by window index
    all_features.sort(key=lambda x: x['window_index'])
    return all_features


def compute_robot_winning_rate(video_path: str, window_start: float, window_end: float) -> dict:
    """
    Call the robot speed service to compute a winning rate
    for the given time window in the video.
    """
    payload = {
        "video_path": video_path,
        "window_start": window_start,
        "window_end": window_end,
    }
    response = requests.post(
        f"{SERVICES['robot_speed']}/winning_rate",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) and SIGTERM signals."""
    signal_name = 'SIGINT' if sig == signal.SIGINT else 'SIGTERM'
    print(f"\n{'='*60}")
    print(f"Received {signal_name} - shutting down pipeline...")
    print(f"{'='*60}")
    sys.exit(130 if sig == signal.SIGINT else 143)