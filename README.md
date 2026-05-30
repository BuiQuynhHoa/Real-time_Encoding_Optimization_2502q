# H.264 Video Compression Profiling System 🎥⚙️

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?logo=streamlit&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-Multimedia-007808?logo=ffmpeg&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?logo=opencv&logoColor=white)

## 📌 Project Overview
This project presents a comprehensive analytical framework dedicated to profiling the **H.264/AVC** multimedia compression standard. By developing a decoupled, two-module architecture, the system provides both rigorous empirical benchmarking data and an intuitive, real-time visual understanding of video encoding trade-offs.

## 🏗️ System Architecture
The application is divided into two primary processing pipelines:

### 1. Module 1: Offline Benchmarking Pipeline
* Automates the encoding process across the entire spectrum of `libx264` presets (from `ultrafast` to `veryslow`) using a locked Constant Rate Factor (CRF = 23).
* Extracts exact processing metadata (FPS, Execution Time, Bitrate) directly from FFmpeg logs.
* Computes objective visual fidelity using **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index Measure).
* Visualizes multidimensional relationships via interactive Plotly charts.

### 2. Module 2: Real-time Webcam Simulation Pipeline
* Overcomes the native synchronous hardware I/O bottlenecks of the Streamlit framework.
* Implements an asynchronous, daemon-threaded `WebcamWorker` with thread-safe memory locks to maintain a stable, low-latency UI rendering loop.
* Utilizes in-memory JPEG compression via OpenCV as a high-throughput proxy to simulate real-time H.264 spatial degradation.

## 🚀 Installation & Setup
This project is optimized for execution within a Linux environment (e.g., Ubuntu). Ensure you have Python 3.12+ and FFmpeg installed on your system.

**1. Clone the repository:**
```bash
git clone [https://github.com/BuiQuynhHoa/Real-time_Encoding_Optimization_2502q.git](https://github.com/BuiQuynhHoa/Real-time_Encoding_Optimization_2502q.git)
cd Real-time_Encoding_Optimization_2502q
```

**2. Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install required dependencies:**
```bash
pip install -r requirements.txt
```

**4. Install system-level FFmpeg:**
```bash
sudo apt update
sudo apt install ffmpeg
```

## Usage
To launch the interactive dashboard, run the following command from the root directory:
```bash
streamlit run app.py
```
*(Navigate to `http://localhost:8501` in your web browser to view the application).*

## Team Contributors
* **Hoang Thi Phuong Nga:** System Architecture, Threading Logic & Real-Time Implementation (Module 2).
* **Bui Quynh Hoa:** Benchmark Engineering, Metadata Extraction & Interactive Data Visualization (Module 1).

## License
This project is developed for academic and research purposes.