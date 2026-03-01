# 📸 MiniSnapchat – Real-Time AR Filters with Gesture Control

MiniSnapchat is a Computer Vision based AR filter application built using Python, OpenCV, and MediaPipe.  
It applies real-time face filters and supports interactive hand gesture detection for dynamic effects.

---
![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
## 🚀 Features

- 🎭 Real-time Face Detection
- 🕶️ AR Filters (Sunglasses, Mustache, Dog Ears)
- ❤️ Gesture-Based Heart Effect
- 🖐️ Hand Tracking using MediaPipe
- 🎛️ Interactive On-Screen Buttons
- ⚫ Black & White Mode
- 🔄 Live Camera Processing

---

## 🛠️ Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy

---

## 🧠 How It Works

1. Captures live video from webcam.
2. Detects face using Haarcascade.
3. Overlays AR filters using alpha blending.
4. Tracks hand landmarks using MediaPipe.
5. Detects gestures and triggers special effects.
6. Interactive hover-based UI to switch filters.

---
## 🎥 Demo

![MiniSnapchat Demo](Minisnap_output.gif)

## ⚙️ Installation

```bash
git clone https://github.com/ShreyaDhomane/MiniSnapchat.git
cd MiniSnapchat
pip install opencv-python mediapipe numpy
python mini_snapchat.py
