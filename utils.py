"""
utils.py - Tiện ích dùng chung cho toàn bộ dự án
Môn: Nén và Mã hóa Đa phương tiện - Mã code: 2502q
"""
import subprocess
import os
import json
import time
import re
import cv2
import numpy as np
import pandas as pd

# =====================================================================
# HẰNG SỐ DÙNG CHUNG
# =====================================================================
H264_PRESETS = [
    'ultrafast', 'superfast', 'veryfast',
    'faster', 'fast', 'medium',
    'slow', 'veryslow'
]

PRESET_TO_JPEG_QUALITY = {
    'ultrafast': 15, 'superfast': 25, 'veryfast': 38,
    'faster': 50, 'fast': 62, 'medium': 72,
    'slow': 85, 'veryslow': 95
}

# =====================================================================
# MODULE 2 
# =====================================================================
def simulate_frame_compression(frame: np.ndarray, preset: str) -> tuple[np.ndarray, float, float]:
    """Mô phỏng hiệu ứng nén H.264 trên 1 frame dùng JPEG (proxy cho webcam live)."""
    quality = PRESET_TO_JPEG_QUALITY.get(preset, 70)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_params)
    
    encoded_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    compressed_kb = len(buffer) / 1024
    psnr_estimate = 25 + quality * 0.2
    
    return encoded_frame, compressed_kb, psnr_estimate

# =====================================================================
# MODULE 1 
# =====================================================================
def check_ffmpeg() -> bool:
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def get_video_info(video_path: str) -> dict | None:
    if not os.path.exists(video_path): return None
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', video_path]
    try:
        data = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout)
        video_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), None)
        if not video_stream: return None
        num, den = map(int, video_stream.get('r_frame_rate', '30/1').split('/'))
        fps = num / den if den != 0 else 30.0
        fmt = data.get('format', {})
        return {
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'fps': round(fps, 2),
            'duration': float(fmt.get('duration', 0)),
            'bitrate': int(fmt.get('bit_rate', 0)),
            'codec': video_stream.get('codec_name', 'unknown'),
            'size_bytes': int(fmt.get('size', 0))
        }
    except Exception: return None

def encode_video(input_path: str, output_path: str, preset: str = 'medium', crf: int = 23) -> float:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cmd = ['ffmpeg', '-y', '-i', input_path, '-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-c:a', 'copy', output_path]
    start = time.time()
    subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return time.time() - start

def get_size_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024*1024)

def get_compression_ratio(original_mb: float, encoded_mb: float) -> float:
    return round(original_mb / encoded_mb, 2) if encoded_mb > 0 else 0.0

def get_encoding_fps(input_path: str, encoding_time_sec: float) -> float:
    info = get_video_info(input_path)
    if not info or encoding_time_sec <= 0: return 0.0
    return round((info['fps'] * info['duration']) / encoding_time_sec, 1)

def calc_psnr(reference_path: str, encoded_path: str, max_sec: float = 20.0) -> float | None:
    info = get_video_info(reference_path)
    duration = min(info['duration'], max_sec) if info else max_sec
    cmd = ['ffmpeg', '-y', '-i', reference_path, '-i', encoded_path, '-t', str(duration), '-lavfi', 'psnr=stats_file=-', '-f', 'null', '-']
    result = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r'average:(\d+\.?\d*)', result.stderr)
    return float(m.group(1)) if m else None

def calc_ssim(reference_path: str, encoded_path: str, max_sec: float = 20.0) -> float | None:
    info = get_video_info(reference_path)
    duration = min(info['duration'], max_sec) if info else max_sec
    cmd = ['ffmpeg', '-y', '-i', reference_path, '-i', encoded_path, '-t', str(duration), '-lavfi', 'ssim=stats_file=-', '-f', 'null', '-']
    result = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r'All:(\d+\.?\d*)', result.stderr)
    return float(m.group(1)) if m else None

def run_benchmark(input_path: str, output_dir: str, presets: list[str], crf: int = 23, progress_callback=None) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    original_mb = get_size_mb(input_path)
    results = []
    
    for step, preset in enumerate(presets):
        if progress_callback: progress_callback(step, len(presets), preset)
        out_path = os.path.join(output_dir, f'enc_{preset}_crf{crf}.mp4')
        try:
            enc_time = encode_video(input_path, out_path, preset, crf)
            enc_mb = get_size_mb(out_path)
            comp_ratio = get_compression_ratio(original_mb, enc_mb)
            enc_fps = get_encoding_fps(input_path, enc_time)
            enc_info = get_video_info(out_path)
            bitrate_kbps = enc_info['bitrate'] / 1000 if enc_info else 0
            psnr = calc_psnr(input_path, out_path, max_sec=20)
            ssim = calc_ssim(input_path, out_path, max_sec=20)
            
            results.append({
                'Preset': preset, 'CRF': crf, 'Encoding FPS': enc_fps,
                'Encoding Time (s)': round(enc_time, 2),
                'Original Size (MB)': round(original_mb, 2), 'Encoded Size (MB)': round(enc_mb, 2),
                'Compression Ratio': comp_ratio, 'Bitrate (kbps)': round(bitrate_kbps, 1),
                'PSNR (dB)': round(psnr, 2) if psnr else None, 'SSIM': round(ssim, 4) if ssim else None,
                '_output_path': out_path
            })
        except Exception as e:
            results.append({'Preset': preset, 'CRF': crf, 'Encoding FPS': 0, 'Encoding Time (s)': 0, 'Original Size (MB)': round(original_mb, 2), 'Encoded Size (MB)': 0, 'Compression Ratio': 0, 'Bitrate (kbps)': 0, 'PSNR (dB)': None, 'SSIM': None, '_output_path': None, 'Error': str(e)})
            
    preset_order = {p: i for i, p in enumerate(H264_PRESETS)}
    df = pd.DataFrame(results)
    if not df.empty:
        df['_order'] = df['Preset'].map(preset_order).fillna(99)
        df = df.sort_values('_order').drop(columns=['_order']).reset_index(drop=True)
    return df