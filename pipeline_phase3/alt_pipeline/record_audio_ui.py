import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import time
import numpy as np

# Audio Settings
fs = 44100          # Sample rate
channels = 1        # Mono

# Recording state
recording = False
audio_buffer = []
start_time = None
BASE_DIR = None  # will be set per recording

# Session folder setup
# session_id = f"session_{int(time.time())}"
#BASE_DIR = Path(f"data/sessions/{session_id}/Audio/raw")
# BASE_DIR.mkdir(parents=True, exist_ok=True)

# session_log = BASE_DIR.parents[1] / "session.log"
# session_log.touch(exist_ok=True)

# Recording state
recording = False
audio_buffer = []
start_time = None

# Tkinter UI
root = tk.Tk()
root.title("Audio Recorder")
root.geometry("300x150")

# Timer Label
timer_label = tk.Label(root, text="00:00", font=("Helvetica", 14))
timer_label.pack(pady=5)

# Start/Stop Button
btn = tk.Button(root, text="Start Recording", width=20)
btn.pack(pady=20)

# Sounddevice stream callback
def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_buffer.append(indata.copy())

# Create the stream (not started yet)
stream = sd.InputStream(samplerate=fs, channels=channels, callback=audio_callback)

# Timer update function
def update_timer():
    if recording and start_time is not None:
        elapsed = int(time.time() - start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
        root.after(100, update_timer)

# Button toggle function
def toggle_recording():
    global recording, audio_buffer, stream, start_time, BASE_DIR

    if not recording:
        # 🔹 Create NEW session
        session_id = f"session_{int(time.time())}"
        BASE_DIR = Path(f"data/sessions/{session_id}/Audio/raw")
        BASE_DIR.mkdir(parents=True, exist_ok=True)

        session_log = BASE_DIR.parents[1] / "session.log"
        session_log.touch(exist_ok=True)

        # Start recording
        recording = True
        audio_buffer = []
        start_time = time.time()
        btn.config(text="Stop Recording")
        stream.start()
        update_timer()
        print(f"Recording started ({session_id})")
    else:
        # Stop recording
        recording = False
        stream.stop()
        btn.config(text="Start Recording")
        print("Recording stopped.")

        # Save the audio
        if audio_buffer:
            audio_data = np.concatenate(audio_buffer, axis=0)
            save_path = BASE_DIR / f"recorded_{int(time.time())}.wav"
            sf.write(save_path, audio_data, fs)
            messagebox.showinfo("Saved", f"Audio saved to {save_path}")
        else:
            messagebox.showwarning("Warning", "No audio recorded!")

        # Reset timer
        timer_label.config(text="00:00")

# Bind button
btn.config(command=toggle_recording)

# Run the Tkinter loop
root.mainloop()