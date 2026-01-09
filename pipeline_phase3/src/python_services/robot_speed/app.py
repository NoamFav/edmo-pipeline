import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.signal import savgol_filter
import math


# ----TABLE CALIBRATION CONFIGURATION---
TABLE_MARKER_IDS = {
    "top_left": 0,
    "top_right": 2,
    "bottom_right": 3,
    "bottom_left": 4
}
HORIZONTAL_DISTANCE_CM = 162
VERTICAL_DISTANCE_CM   = 102


# ---ARUCO & ROBOT CONFIG-----
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ROBOT_MARKER_IDS = None

SMOOTH_WINDOW = 9
FRAME_STRIDE = 3 #skip this many frames each iteration


app = FastAPI(
    title="Robot Speed Window Service",
    description="Robot speed calculation per video window.",
    version="2.0",
)


class WindowRequest(BaseModel):
    video_path: str
    window_start: float
    window_end: float



# SMALL HELPERS

def _marker_center(corners):
    """Return the (x, y) center of an ArUco marker."""
    c = corners.reshape(-1, 2)
    return np.mean(c, axis=0)


def _dist(a, b):
    """Euclidean distance between two (x,y) points."""
    return float(np.linalg.norm(a - b))


#---- TABLE CALIBRATION ROUTINE

def _calibrate_table(first_frames_centers):
    """
    Attempt to calibrate pixels → cm using table markers.
    We search for the first frame that contains all 4 table markers.
    """

    required_ids = set(TABLE_MARKER_IDS.values())

    for marker_map in first_frames_centers:
        # Check if all required markers were detected in this frame
        if not required_ids.issubset(marker_map.keys()):
            continue

        # Extract raw pixel centers
        tl = np.array(marker_map[TABLE_MARKER_IDS["top_left"]])
        tr = np.array(marker_map[TABLE_MARKER_IDS["top_right"]])
        bl = np.array(marker_map[TABLE_MARKER_IDS["bottom_left"]])

        # Pixel distances
        horizontal_px = _dist(tl, tr)
        vertical_px   = _dist(tl, bl)

        if horizontal_px < 1 or vertical_px < 1:
            continue  # bad fram

        # Compute per-axis scaling 
        pixels_per_cm_x = horizontal_px / HORIZONTAL_DISTANCE_CM
        pixels_per_cm_y = vertical_px / VERTICAL_DISTANCE_CM

        return {
            "origin_px": tl,   
            "ppcm_x": pixels_per_cm_x,
            "ppcm_y": pixels_per_cm_y
        }

    # No valid calibration frame found
    return None


#----- MAIN SPEED EXTRACTION FUNCTION----

def compute_speed_for_window(video_path: str, window_start: float, window_end: float):
    """
    Compute robot speed (cm/s) inside one window:
    - Detect table markers → calibrate pixel→cm
    - For each frame: detect robot markers → compute robot center
    - Convert center to cm using calibration
    - Compute velocity from smoothed trajectory
    """

    # ----Gather calibration frames---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    calibration_frames = []
    pre_calib_robot_px = []
    robot_positions = []
    calib = None
    origin = None
    ppcm_x = None
    ppcm_y = None
    inv_ppcm_x = None
    inv_ppcm_y = None
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % FRAME_STRIDE != 0:
            continue

        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if t < window_start:
            continue
        if t >= window_end:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)

        if ids is None:
            continue

        ids = ids.flatten()
        centers = {}

        for i, mid in enumerate(ids):
            centers[mid] = _marker_center(corners[i])

        if calib is None:
            calibration_frames.append(centers)
            calib = _calibrate_table(calibration_frames)
            if calib is not None:
                origin = np.array(calib["origin_px"])
                ppcm_x = calib["ppcm_x"]
                ppcm_y = calib["ppcm_y"]
                inv_ppcm_x = 1.0 / ppcm_x
                inv_ppcm_y = 1.0 / ppcm_y

        robot_centers = []

        for i, mid in enumerate(ids):
            if mid in TABLE_MARKER_IDS.values():
                continue
            robot_centers.append(_marker_center(corners[i]))

        if len(robot_centers) == 0:
            continue

        center_px = np.mean(robot_centers, axis=0)

        if calib is None:
            pre_calib_robot_px.append((t, center_px[0], center_px[1]))
        else:
            rel = center_px - origin
            x_cm = rel[0] * inv_ppcm_x
            y_cm = rel[1] * inv_ppcm_y
            robot_positions.append((t, x_cm, y_cm))

    cap.release()

    if calib is None:
        return np.nan, 0

    for t, x_px, y_px in pre_calib_robot_px:
        rel = np.array([x_px, y_px]) - origin
        x_cm = rel[0] * inv_ppcm_x
        y_cm = rel[1] * inv_ppcm_y
        robot_positions.append((t, x_cm, y_cm))

    if len(robot_positions) <= 1:
        return np.nan, len(robot_positions)

    robot_positions.sort(key=lambda p: p[0])
    arr = np.array(robot_positions)
    t = arr[:, 0]
    x = arr[:, 1]
    y = arr[:, 2]

    # smoothing for smoother velocity
    if len(x) > SMOOTH_WINDOW:
        x = savgol_filter(x, SMOOTH_WINDOW, 3)
        y = savgol_filter(y, SMOOTH_WINDOW, 3)

    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)

    dt[dt == 0] = np.finfo(float).eps

    v = np.sqrt((dx / dt) ** 2 + (dy / dt) ** 2)
    v = np.insert(v, 0, np.nan)  

    avg_speed_cm_s = float(np.nanmean(v))
    num = len(robot_positions)

    return avg_speed_cm_s, num



# API ROUTES

def speed_to_winning_rate(avg_speed_cm_s: float) -> float:
    """Map speed (cm/s) → [0,1] winning rate."""
    if np.isnan(avg_speed_cm_s):
        return 0.0
    target = 20.0  # tuneable
    return float(max(0.0, min(1.0, avg_speed_cm_s / target)))


@app.post("/winning_rate")
def compute_winning_rate(req: WindowRequest):
    """Main endpoint the pipeline calls."""
    try:
        avg_speed, num = compute_speed_for_window(
            req.video_path, req.window_start, req.window_end
        )

        return {
            "video_path": req.video_path,
            "window_start": req.window_start,
            "window_end": req.window_end,
            "avg_speed_cm_s": avg_speed,
            "num_detections": num,
            "winning_rate": speed_to_winning_rate(avg_speed)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy"}
