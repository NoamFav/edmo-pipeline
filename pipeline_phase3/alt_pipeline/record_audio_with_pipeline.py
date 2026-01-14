import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import time
import numpy as np
import subprocess
import sys

# Audio Settings
fs = 44100
channels = 1

# Recording state
recording = False
audio_buffer = []
start_time = None

BASE_DIR = None
SESSION_DIR = None
last_recording_path = None

# Tkinter UI
root = tk.Tk()
root.title("Audio Recorder")
root.geometry("300x275")

window_len_var = tk.IntVar(value=2)

# Timer Label
timer_label = tk.Label(root, text="00:00", font=("Helvetica", 14))
timer_label.pack(pady=5)

# Window length selector
window_frame = tk.Frame(root)
window_frame.pack(pady=5)

tk.Label(window_frame, text="Window length (sec):").pack(side=tk.LEFT)

window_spinbox = tk.Spinbox(
    window_frame,
    from_=1,
    to=60,
    increment=1,
    textvariable=window_len_var,
    width=5
)
window_spinbox.pack(side=tk.LEFT, padx=5)

# Start/Stop Button
record_btn = tk.Button(root, text="Start Recording", width=20)
record_btn.pack(pady=5)

# Run Pipeline Button
pipeline_btn = tk.Button(
    root,
    text="Run Pipeline",
    width=20,
    state=tk.DISABLED
)
pipeline_btn.pack(pady=5)

# Assign Clusters Button
assign_btn = tk.Button(
    root,
    text="Assign Audio Clusters",
    width=20,
    state=tk.DISABLED
)
assign_btn.pack(pady=5)

# Status label
status_label = tk.Label(
    root,
    text="No session recorded yet.",
    wraplength=260,
    justify="left",
    fg="gray"
)
status_label.pack(pady=5)

# Sounddevice stream callback
def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_buffer.append(indata.copy())

stream = sd.InputStream(
    samplerate=fs,
    channels=channels,
    callback=audio_callback
)

# Timer update
def update_timer():
    if recording and start_time is not None:
        elapsed = int(time.time() - start_time)
        timer_label.config(
            text=f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        )
        root.after(100, update_timer)

def toggle_recording():
    global recording, audio_buffer, start_time
    global BASE_DIR, SESSION_DIR, last_recording_path

    if not recording:
        # Create new session
        session_id = f"session_{int(time.time())}"
        SESSION_DIR = Path(f"data/sessions/{session_id}")
        BASE_DIR = SESSION_DIR / "Audio/raw"
        BASE_DIR.mkdir(parents=True, exist_ok=True)

        (SESSION_DIR / "session.log").touch(exist_ok=True)

        # Start recording
        recording = True
        audio_buffer = []
        start_time = time.time()

        record_btn.config(text="Stop Recording")
        pipeline_btn.config(state=tk.DISABLED)

        stream.start()
        update_timer()
        print(f"Recording started ({session_id})")

    else:
        # Stop recording
        recording = False
        stream.stop()
        record_btn.config(text="Start Recording")

        # Save the audio
        if audio_buffer:
            audio_data = np.concatenate(audio_buffer, axis=0)
            last_recording_path = BASE_DIR / f"recorded_{int(time.time())}.wav"
            sf.write(last_recording_path, audio_data, fs)

            messagebox.showinfo(
                "Saved",
                f"Audio saved to:\n{last_recording_path}"
            )

            pipeline_btn.config(state=tk.NORMAL)

            status_label.config(
                text=f"Last recording:\n{SESSION_DIR.name}",
                fg="gray"
            )
        else:
            messagebox.showwarning("Warning", "No audio recorded!")

        # Reset timer
        timer_label.config(text="00:00")
        print("Recording stopped.")

def run_pipeline():
    if not SESSION_DIR:
        messagebox.showerror("Error", "No session to process.")
        return

    pipeline_btn.config(state=tk.DISABLED)
    root.update()

    try:
        output_path = "data/output"
        cmd = [
            sys.executable,
            "alt_pipeline/pipeline.py",
            "--input", str(SESSION_DIR),
            # "--output", "alt_pipeline/data/output",
            "--output", output_path,
            "--window-len", str(window_len_var.get())
        ]

        subprocess.run(cmd, check=True)

        messagebox.showinfo(
            "Pipeline finished",
            "Pipeline completed successfully."
        )

        status_label.config(
            text=(
                f"Last processed session:\n"
                f"{SESSION_DIR.name}\n\n"
                f"Output saved in:\n"
                f"{output_path}"
            ),
            fg="green"
        )

        assign_btn.config(state=tk.NORMAL)

    except subprocess.CalledProcessError as e:
        messagebox.showerror(
            "Pipeline error",
            f"Pipeline failed.\n\n{e}"
        )
        status_label.config(
            text=f"Pipeline failed for:\n{SESSION_DIR.name}",
            fg="red"
        )

    pipeline_btn.config(state=tk.NORMAL)

def assign_audio_clusters():
    messagebox.showinfo(
        "Cluster assignment error",
        "Assign audio clusters is not yet enabled."
    )

# Bind buttons
record_btn.config(command=toggle_recording)
pipeline_btn.config(command=run_pipeline)
assign_btn.config(command=assign_audio_clusters)

# Run UI
root.mainloop()