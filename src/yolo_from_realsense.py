# yolo_from_realsense.py
# Single-window YOLOv8 on RealSense RGB (no fullscreen). Press 'm' to mirror, 'ESC' to quit.

import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import torch

# ---------------- Window / UI ----------------
WIN = "YOLO-RealSense-RGB"        # keep this EXACT everywhere to avoid second window
TARGET_W, TARGET_H = 1280, 720    # window size (resizes image to FIT with letterbox)

def show_fit(img):
    """Scale-to-FIT inside TARGET_W x TARGET_H (letterbox, no cropping)."""
    h, w = img.shape[:2]
    r = min(TARGET_W / w, TARGET_H / h)
    nw, nh = int(w * r), int(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    y0 = (TARGET_H - nh) // 2
    x0 = (TARGET_W - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    cv2.imshow(WIN, canvas)

# ---------------- RealSense ----------------
def start_pipe():
    p = rs.pipeline()
    c = rs.config()
    # Keep the stream simple; the renderer upscales safely
    c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    p.start(c)
    return p

pipe = start_pipe()

# ---------------- YOLO ----------------
MODEL_PATH = "yolov8n.pt"  # or put your custom model path here
model = YOLO(MODEL_PATH)

DEVICE = 0 if torch.cuda.is_available() else "cpu"  # CUDA GPU id 0 or CPU

# ---------------- Window init ----------------
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, TARGET_W, TARGET_H)

mirror = False
t0 = time.perf_counter()
frames_drawn = 0
fps = 0.0

try:
    while True:
        # Grab frame with a short timeout and auto-recover if needed
        try:
            frameset = pipe.wait_for_frames(1000)  # ms
        except Exception:
            # quick recover if camera hiccups
            try:
                pipe.stop()
            except Exception:
                pass
            time.sleep(0.1)
            pipe = start_pipe()
            continue

        color = frameset.get_color_frame()
        if not color:
            continue

        img = np.asanyarray(color.get_data())
        if mirror:
            img = cv2.flip(img, 1)

        # YOLO inference
        r = model.predict(img, imgsz=640, conf=0.25, device=DEVICE, verbose=False)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cls = r.names[int(b.cls[0])]
            conf = float(b.conf[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 12), 2)
            cv2.putText(img, f"{cls}:{conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)

        # FPS (smoothed)
        frames_drawn += 1
        if frames_drawn % 10 == 0:
            dt = time.perf_counter() - t0
            fps = frames_drawn / max(dt, 1e-6)

        cv2.putText(img, f"FPS:{fps:4.1f}", (22, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 255), 3, cv2.LINE_AA)
        cv2.putText(img, f"Mirror:{'ON' if mirror else 'OFF'}", (260, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 255), 3, cv2.LINE_AA)

        # Render (fit in one stable window)
        show_fit(img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:          # ESC
            break
        elif k == ord('m'):  # toggle mirror
            mirror = not mirror

finally:
    try:
        pipe.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()
