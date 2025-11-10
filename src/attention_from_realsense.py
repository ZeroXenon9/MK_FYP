# attention_from_realsense.py  (fit-to-window header + auto RS profile fallback)
# Keys: [O]=Calib Open  [X]=Calib Closed  [L]=Log ON/OFF  [M]=Mirror ON/OFF  [H]=Header Full/Compact  [ESC]=Quit

import time, math, csv, os
import cv2, numpy as np, mediapipe as mp, pyrealsense2 as rs

WIN = "Attention-RealSense"
TARGET_W, TARGET_H = 1280, 720
MIRROR = False
HEADER_MODE = "full"   # "full" or "compact"

# ---------- drawing helpers ----------
def show_fit(img):
    h, w = img.shape[:2]
    r = min(TARGET_W / w, TARGET_H / h)
    nw, nh = max(1, int(w * r)), max(1, int(h * r))
    canvas = np.zeros((TARGET_H, TARGET_W, 3), np.uint8)
    y0, x0 = (TARGET_H - nh) // 2, (TARGET_W - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    cv2.imshow(WIN, canvas)

def draw_header(img, lines, bg_color):
    """Render 1–2 lines that always fit: auto-shrink font when needed."""
    h, w = img.shape[:2]
    # try decreasing font size until widest line fits margins
    margin, pad = 10, 8
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, 2
    scale = 1.1 if len(lines) == 1 else 0.95
    while scale > 0.4:
        widths = [cv2.getTextSize(t, font, scale, thickness)[0][0] for t in lines]
        if max(widths) + 2 * margin <= w:
            break
        scale -= 0.05

    # header height
    sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
    line_h = max(s[1] for s in sizes) + pad + 6
    total_h = line_h * len(lines)

    # background bar
    cv2.rectangle(img, (0, 0), (w, total_h), bg_color, -1)

    # draw text centered-left with margin
    y = pad + sizes[0][1] + 6
    for t in lines:
        cv2.putText(img, t, (margin, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h

# ---------- RealSense: robust start with fallbacks ----------
def try_start_profile(width, height, fps=30):
    p, c = rs.pipeline(), rs.config()
    c.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    p.start(c)
    return p, (width, height, fps)

def start_pipe_autofallback():
    attempts = [(1280, 720, 30), (848, 480, 30), (640, 480, 30)]
    last = None
    for w, h, f in attempts:
        try:
            return try_start_profile(w, h, f)
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to start RealSense color: {last}")

pipe, USED_PROFILE = start_pipe_autofallback()

# ---------- MediaPipe FaceMesh & EAR ----------
mp_face = mp.solutions.face_mesh
DRAW = mp.solutions.drawing_utils
LEFT  = [33,160,158,133,153,144]
RIGHT = [362,385,387,263,373,380]

def ear(lm, idx):
    d = lambda a, b: math.dist((a.x, a.y), (b.x, b.y))
    p = [lm[i] for i in idx]
    return (d(p[1], p[5]) + d(p[2], p[4])) / (2.0 * d(p[0], p[3]) + 1e-6)

# ---------- state ----------
thr = 0.25
ear_open = None
ear_closed = None
win = []
win_len = 90  # ~3s @30fps
logging = False
os.makedirs("logs", exist_ok=True)
log_path = "logs/attention_frames.csv"
fcsv = open(log_path, "w", newline="")
writer = csv.writer(fcsv)
writer.writerow(["ts", "ear", "open_flag", "open_frac", "thr"])

def collect_mean(seconds, fm):
    t_end = time.time() + seconds
    vals = []
    while time.time() < t_end:
        try:
            frames = pipe.wait_for_frames(700)
        except Exception:
            try: pipe.stop()
            except Exception: pass
            time.sleep(0.15)
            globals()["pipe"], _ = start_pipe_autofallback()
            continue
        c = frames.get_color_frame()
        if not c:
            continue
        img = np.asanyarray(c.get_data())
        if MIRROR: img = cv2.flip(img, 1)
        out = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if out.multi_face_landmarks:
            lm = out.multi_face_landmarks[0].landmark
            vals.append((ear(lm, LEFT) + ear(lm, RIGHT)) / 2.0)
        show_fit(img)
        if cv2.waitKey(1) & 0xFF == 27: break
    return (sum(vals)/len(vals)) if vals else None

# ---------- main ----------
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, TARGET_W, TARGET_H)

try:
    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as fm:
        last_log = 0.0
        t0 = time.perf_counter()
        n = 0
        fps = 0.0

        while True:
            try:
                frames = pipe.wait_for_frames(1000)
            except Exception:
                try: pipe.stop()
                except Exception: pass
                time.sleep(0.15)
                pipe, USED_PROFILE = start_pipe_autofallback()
                continue

            c = frames.get_color_frame()
            if not c:
                # keep window responsive
                img = np.zeros((480, 640, 3), np.uint8)
                draw_header(img, ["Reconnecting camera..."], (0, 140, 255))
                show_fit(img)
                if cv2.waitKey(1) & 0xFF == 27: break
                continue

            img = np.asanyarray(c.get_data())
            if MIRROR: img = cv2.flip(img, 1)

            out = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            e, detected = None, False
            if out.multi_face_landmarks:
                detected = True
                lm = out.multi_face_landmarks[0].landmark
                e = (ear(lm, LEFT) + ear(lm, RIGHT)) / 2.0
                DRAW.draw_landmarks(img, out.multi_face_landmarks[0], mp_face.FACEMESH_TESSELATION)

            if e is not None:
                win.append(1 if e > thr else 0)
                if len(win) > win_len: win.pop(0)
            open_frac = (sum(win)/len(win)) if win else 1.0

            # header text (auto-shrink; 1 or 2 lines)
            prof = f"{USED_PROFILE[0]}x{USED_PROFILE[1]}@{USED_PROFILE[2]}"
            status = "ATTENTIVE" if open_frac >= 0.8 else "NOT ATTENTIVE"
            if not detected:
                lines = [f"{prof}  Face not detected — adjust position/lighting (M=mirror)"]
                color = (0, 0, 200)
            else:
                if HEADER_MODE == "full":
                    lines = [
                        f"{prof}  {status}  |  EAR:{0 if e is None else e:.2f}  open:{open_frac:.2f}  thr:{thr:.2f}  Mirror:{'ON' if MIRROR else 'OFF'}",
                        "[O]=Calib Open   [X]=Calib Closed   [L]=Log ON/OFF   [H]=Header Full/Compact   [ESC]=Quit",
                    ]
                else:
                    lines = [
                        f"{prof}  {status}  EAR:{0 if e is None else e:.2f}  open:{open_frac:.2f}  thr:{thr:.2f}  M:{'ON' if MIRROR else 'OFF'}",
                    ]
                color = (40, 180, 40) if status == "ATTENTIVE" else (0, 0, 255)

            draw_header(img, lines, color)

            # FPS bottom-left
            n += 1
            if n % 10 == 0:
                fps = n / (time.perf_counter() - t0 + 1e-9)
            cv2.putText(img, f"FPS:{fps:4.1f}", (12, img.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2, cv2.LINE_AA)

            show_fit(img)

            # logging (only if we have a valid EAR)
            if logging and (e is not None) and (time.time() - last_log > 1/30):
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{e:.4f}",
                    (win[-1] if win else 1),
                    f"{open_frac:.3f}",
                    f"{thr:.3f}",
                ])
                last_log = time.time()

            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            elif k == ord('m'): MIRROR = not MIRROR
            elif k == ord('l'): logging = not logging
            elif k == ord('h'): HEADER_MODE = "compact" if HEADER_MODE == "full" else "full"
            elif k == ord('o'):
                draw_header(img, ["CALIB OPEN: keep eyes open ~2s..."], (0, 140, 255)); show_fit(img); cv2.waitKey(1)
                ear_open = collect_mean(2.0, fm)
                if ear_open:
                    base = ear_closed or (ear_open * 0.65)
                    thr = max(0.15, min(0.45, (ear_open + base)/2.0))
            elif k == ord('x'):
                draw_header(img, ["CALIB CLOSED: gently close eyes ~2s..."], (0, 140, 255)); show_fit(img); cv2.waitKey(1)
                ear_closed = collect_mean(2.0, fm)
                if ear_closed:
                    base = ear_open or (ear_closed * 1.35)
                    thr = max(0.15, min(0.45, (base + ear_closed)/2.0))

finally:
    try: pipe.stop()
    except Exception: pass
    cv2.destroyAllWindows()
    try: fcsv.close()
    except Exception: pass

print(f"[Saved] {log_path}")
