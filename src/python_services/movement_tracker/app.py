import cv2
import numpy as np
import pandas as pd
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from scipy.signal import savgol_filter
import os
import asyncio

app = FastAPI(title="Robot Movement Tracker",
              description="Extracts position and velocity data from robot videos using ArUco markers.",
              version="1.0")

# CONFIGURATION 
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100) #change for appropiate aruco code
MARKER_ID = None
TRAIL_LENGTH = 200
PIXELS_PER_CM = 4.0  # Adjust once camera calibration is known
SMOOTH_WINDOW = 9


# CORE PROCESSING FUNCTION 
def process_video(video_path: str, output_path: str) -> None:
    """Runs the robot tracking pipeline on the uploaded video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if MARKER_ID is None or marker_id == MARKER_ID:
                    c = corners[i][0]
                    center = np.mean(c, axis=0)
                    x, y = center
                    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    positions.append((t, x, y))

    cap.release()

    # CALCULATE VELOCITY
    if len(positions) > 1:
        positions = np.array(positions, dtype=float)
        t, x, y = positions[:, 0], positions[:, 1], positions[:, 2]

        # Smooth positions for stability
        if len(x) > SMOOTH_WINDOW:
            x = savgol_filter(x, SMOOTH_WINDOW, 3)
            y = savgol_filter(y, SMOOTH_WINDOW, 3)

        dt = np.diff(t)
        dx, dy = np.diff(x), np.diff(y)
        dt[dt == 0] = np.finfo(float).eps  # avoid division by zero

        vx = dx / dt
        vy = dy / dt
        v = np.sqrt(vx**2 + vy**2)

        vx = np.insert(vx, 0, np.nan)
        vy = np.insert(vy, 0, np.nan)
        v = np.insert(v, 0, np.nan)

        vx_cm = vx / PIXELS_PER_CM
        vy_cm = vy / PIXELS_PER_CM
        v_cm = v / PIXELS_PER_CM

        df = pd.DataFrame({
            "time_s": t,
            "x_px": x, "y_px": y,
            "vx_px_s": vx, "vy_px_s": vy, "v_px_s": v,
            "vx_cm_s": vx_cm, "vy_cm_s": vy_cm, "v_cm_s": v_cm
        })
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("No valid marker detections found.")



async def run_tracking(video_file: UploadFile) -> str:
    """Handles async processing of video and returns the path to the output CSV."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            data = await video_file.read()
            tmp_in.write(data)
            tmp_in.flush()
            input_path = tmp_in.name

        output_fd, output_path = tempfile.mkstemp(suffix=".csv")
        os.close(output_fd)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, process_video, input_path, output_path)

        os.remove(input_path)
        return output_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/track", summary="Track robot movement from video", response_description="CSV file with robot movement data")
async def track_robot(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accepts a video file and returns a CSV file with position and velocity data."""
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a video file.")

    output_path = await run_tracking(file)

    # Delete output file after sending it
    background_tasks.add_task(os.remove, output_path)

    return FileResponse(
        output_path,
        media_type="text/csv",
        filename="robot_positions_velocity.csv"
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

