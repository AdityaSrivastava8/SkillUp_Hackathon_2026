# Behavioral Analysis System

Real-time multi-signal behavioral analysis from a webcam, uploaded image, or video file.

## What It Does

Extracts 25+ behavioral signals using computer vision:
- Facial: EAR, MAR, brow raise, blink count, blink bursts
- Eye: gaze direction, aversion events, gaze heatmap
- Head: yaw, pitch, roll, motion energy
- Emotion: 7-class classifier from 52 MediaPipe blendshapes
- Motion: landmark velocity, jitter, optical flow
- Tracking: ByteTrack person IDs
- Composite: Stress Index, Engagement Score

## Features
- Upload image - instant snapshot analysis
- Upload video - full time-series analysis
- Live webcam - real-time HUD
- Browse past sessions - view any session
- PDF reports - professional downloadable reports

## Setup

1. Install Python 3.10+
2. Install dependencies:
   pip install -r requirements.txt
3. Launch:
   python -m streamlit run app.py
4. Open http://localhost:8501

## Tech Stack
OpenCV, MediaPipe, YOLOv8 + ByteTrack, Streamlit, ReportLab, Plotly, Pandas

## Ethics
Not a lie detector. Metrics are heuristic observations, not measurements of
truthfulness or intent. Do not use as sole basis for any investigative or
legal decision.
