# Real-Time Encoding Optimization — Project 2502Q 🎥⚙️

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B)
![FFmpeg](https://img.shields.io/badge/FFmpeg-required-007808)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8)

**Course:** Multimedia Compression and Coding
**Project Code:** 2502Q
**Authors:** Bui Quynh Hoa (202414627), Hoang Thi Phuong Nga (202414648)
**Supervisor:** Dr. rer. Nat. Pham Van Tien
**Institution:** Hanoi University of Science and Technology (HUST)
---

# 1. Introduction
## Project Overview
This project investigates the trade-off between video compression efficiency and perceptual video quality in real-time multimedia systems.
The objective is to evaluate how different H.264 encoding configurations affect:

* Encoding speed
* Compression ratio
* Bitrate
* Visual quality
* Real-time processing capability

The system combines offline benchmarking and real-time quality monitoring to provide a comprehensive evaluation framework for video encoding optimization.

---

## Research Question

How do H.264 encoding presets influence compression efficiency, computational performance, and objective video quality metrics in both offline and real-time scenarios?

---

# 2. System Architecture
```
Input Video / Webcam
          │
          ▼
 Compression Pipeline
          │
          ▼
 Encoded Output
          │
          ▼
 Quality Assessment
(PSNR, SSIM, Bitrate)
          │
          ▼
 Statistical Analysis
          │
          ▼
 Visualization Dashboard
```

---

# 3. Project Modules

| Module   | Description                                                                                     |
| -------- | ----------------------------------------------------------------------------------------------- |
| Module 1 | Offline H.264 benchmarking using FFmpeg and libx264                                             |
| Module 2 | Real-time webcam quality evaluation using a JPEG compression proxy and H.264 recording workflow |

---

# 4. Project Structure
```
REAL-TIME_ENCODING_OPTIMIZATION_2502Q/
│
├── code/
│   ├── app.py
│   ├── module1_file_encoding.py
│   ├── module2_webcam_live.py
│   ├── utils.py
│   ├── run_benchmark_cli.py
│   └── download_dataset.py
│
├── datasets/
│   └── raw/
│
├── results/
│
├── generate_dataset.py
├── requirements.txt
└── README.md
```
---

# 5. Prerequisites

| Component | Minimum Version |
| --------- | --------------- |
| Python    | 3.10+           |
| FFmpeg    | 4.4+            |
| OpenCV    | 4.9+            |
| Streamlit | 1.32+           |

### Hardware Requirements

* Webcam (required for Module 2)
* Minimum 4 GB RAM
* Recommended: 8 GB RAM or higher

---

# 6. Installation

## Install FFmpeg

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y ffmpeg
```

### macOS

```bash
brew install ffmpeg
```

### Windows

```bash
choco install ffmpeg
```

Verify installation:

```bash
ffmpeg -version
ffprobe -version
```

---

## Clone Repository

```bash
git clone https://github.com/BuiQuynhHoa/Real-time_Encoding_Optimization_2502q.git

cd Real-time_Encoding_Optimization_2502q
```

---

## Create Virtual Environment

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 7. Dataset Preparation

## Option A – Download Public Test Videos

```bash
cd code

python download_dataset.py
```

## Option B – Generate Synthetic Videos

```bash
python generate_dataset.py --duration 15 --fps 30
```

## Option C – Use Your Own Videos

Supported formats:

* MP4
* AVI
* MOV
* MKV

---

# 8. Usage

## Launch Application

```bash
cd code

streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Module 1 – Offline Benchmarking

1. Open **Module 1**
2. Upload a video
3. Select encoding presets
4. Set CRF value
5. Start benchmark
6. Review charts and metrics
7. Export CSV results

### Generated Metrics

* FPS
* Bitrate
* PSNR
* SSIM
* Compression Ratio
* Encoding Time

---

## Module 2 – Real-Time Evaluation

### Live Compression Mode

* Webcam capture
* JPEG compression proxy
* Real-time quality monitoring
* FPS and latency visualization

### Record-and-Encode Mode

* Webcam recording
* H.264 encoding using FFmpeg
* Preset comparison
* Quality analysis

---

# 9. Reproducibility

## Reproduce Module 1 Results

```bash
cd code

python run_benchmark_cli.py \
    --input ../datasets/raw/test_720p_motion.mp4 \
    --presets ultrafast veryfast medium slow veryslow \
    --crf 23 \
    --output ../results/benchmark_motion.csv
```

Expected outputs:

* CSV benchmark tables
* Quality metrics
* Compression statistics

---

## Reproduce Module 2 Results

1. Connect a webcam.
2. Launch the Streamlit application.
3. Open Module 2.
4. Run multiple quality configurations.
5. Record PSNR, SSIM, FPS, and latency values.

---

# 10. Expected Outputs

The following outputs should be generated:

```text
results/
├── benchmark_motion.csv
├── benchmark_static.csv
├── benchmark_mixed.csv
└── encoded_videos/
```

Expected trends:

* Slower presets produce higher compression efficiency.
* Faster presets achieve higher encoding speed.
* Lower quality settings reduce PSNR and SSIM.
* Real-time compression introduces visible artifacts.

---

# 11. Authors

| Name                 | Student ID | Responsibilities                               |
| -------------------- | ---------- | ---------------------------------------------- |
| Bui Quynh Hoa        | 202414627  | Benchmark Engine, Metrics, Visualization       |
| Hoang Thi Phuong Nga | 202414648  | Real-Time Module, System Design, Documentation |

---

# 12. License

This project is developed for educational and research purposes.

Datasets used:

* Big Buck Bunny (CC BY 3.0)
* Elephants Dream (CC BY 2.5)
