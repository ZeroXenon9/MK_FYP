# Week 1 Validation — 2025-11-10

Hardware: Intel RealSense D435i (RGB 640x480@30), RTX 3060 (CUDA ok)
Environment: Home, mixed lighting

## YOLO sanity
- Model: yolov8n.pt (CPU/GPU ok)
- FPS (approx): 12–20 @ 640x480
- Behavior: boxes stable on person/objects

## Attention sanity
- Calibration: done (O=2s open, X=2s closed)
- Typical EAR (open): ~0.27–0.35
- open_frac while attentive: ≥0.85 (target met)
- EAR zeros (lost face): rare
- Log file: logs/attention_frames.csv present

## Issues to watch
- Backlight reduces landmark quality → face front-lit in lab.
- Keep 640x480@30 for reliability.
