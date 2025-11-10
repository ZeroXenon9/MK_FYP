# MK_FYP — RealSense + YOLO + Attention

## Setup
1) Install Miniconda, then:
   conda env create -f env/environment.yml
   conda activate fyp

2) Plug Intel RealSense via USB 3.0 (close other camera apps).

## Run
- YOLO real-time:     python src/yolo_from_realsense.py
  Keys: M=mirror toggle, ESC=quit

- Attention tracker:  python src/attention_from_realsense.py
  Keys: O=calib open (2s), X=calib closed (2s), L=log on/off, M=mirror, ESC=quit
  Logs to: logs/attention_frames.csv

## Notes
- Recommended stream: 640x480@30 for stable FaceMesh.
- Do calibration (O then X) in your actual environment (lab/classroom).
