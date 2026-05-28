"""
utils.py — Tiện ích dùng chung cho toàn bộ dự án
=========================================================
Môn:     Nén và Mã hóa Đa phương tiện
Mã code: 2502q
Project: Real-Time Encoding Optimization

Các hàm chính:
  - check_ffmpeg()            → Kiểm tra FFmpeg đã cài chưa
  - get_video_info()          → Lấy metadata video (ffprobe)
  - encode_video()            → Mã hóa video H.264 bằng FFmpeg
  - calc_psnr()               → Tính PSNR (chất lượng)
  - calc_ssim()               → Tính SSIM (chất lượng)
  - get_size_mb()             → Kích thước file (MB)
  - get_compression_ratio()   → Tỉ lệ nén
  - get_encoding_fps()        → FPS mã hóa
  - simulate_frame_compression() → Mô phỏng nén frame webcam (JPEG)
  - run_benchmark()           → Chạy benchmark nhiều preset, trả về DataFrame
"""

import subprocess
import os
import json
import time
import re
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# HẰNG SỐ
# ─────────────────────────────────────────────────────────────────────────────

# Thứ tự preset từ nhanh → chậm
# Nhanh hơn  → file to hơn, chất lượng thấp hơn, phù hợp real-time
# Chậm hơn   → file nhỏ hơn, chất lượng cao hơn, phù hợp lưu trữ
H264_PRESETS = [
    'ultrafast', 'superfast', 'veryfast',
    'faster', 'fast', 'medium',
    'slow', 'veryslow',
]

# Map preset → JPEG quality (0–100) để mô phỏng trong webcam live
# Vì H.264 encoding thật quá chậm cho real-time, ta dùng JPEG làm proxy:
#   ultrafast ≈ quality thấp (nhiều artifact)
#   veryslow  ≈ quality cao  (ít artifact)
PRESET_TO_JPEG_QUALITY = {
    'ultrafast': 15,
    'superfast': 25,
    'veryfast':  38,
    'faster':    50,
    'fast':      62,
    'medium':    72,
    'slow':      85,
    'veryslow':  95,
}


# ─────────────────────────────────────────────────────────────────────────────
# KIỂM TRA MÔI TRƯỜNG
# ─────────────────────────────────────────────────────────────────────────────

def check_ffmpeg() -> bool:
    """Kiểm tra FFmpeg đã được cài đặt và có thể chạy chưa."""
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            check=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# THÔNG TIN VIDEO
# ─────────────────────────────────────────────────────────────────────────────

def get_video_info(video_path: str) -> dict | None:
    """
    Lấy metadata của video dùng ffprobe.

    Returns:
        dict gồm các key:
          width, height  — độ phân giải (pixel)
          fps            — frame per second
          duration       — thời lượng (giây)
          bitrate        — bitrate (bps)
          codec          — tên codec ('h264', 'mpeg4', ...)
          size_bytes     — kích thước file (bytes)
        None nếu file không tồn tại hoặc có lỗi.
    """
    if not os.path.exists(video_path):
        return None

    cmd = [
        'ffprobe',
        '-v',            'quiet',   # tắt log không cần thiết
        '-print_format', 'json',    # output dạng JSON để dễ parse
        '-show_streams',            # thông tin từng stream (video/audio)
        '-show_format',             # thông tin tổng (duration, bitrate, size)
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data   = json.loads(result.stdout)
    except Exception as e:
        print(f"[utils] ffprobe lỗi: {e}")
        return None

    # Tìm video stream đầu tiên trong danh sách streams
    video_stream = next(
        (s for s in data.get('streams', []) if s.get('codec_type') == 'video'),
        None
    )
    if video_stream is None:
        return None

    # r_frame_rate có dạng "30000/1001" (≈ 29.97) hoặc "30/1"
    try:
        num, den = map(int, video_stream.get('r_frame_rate', '30/1').split('/'))
        fps = num / den if den != 0 else 30.0
    except Exception:
        fps = 30.0

    fmt = data.get('format', {})

    return {
        'width':      int(video_stream.get('width',  0)),
        'height':     int(video_stream.get('height', 0)),
        'fps':        round(fps, 2),
        'duration':   float(fmt.get('duration', 0)),
        'bitrate':    int(fmt.get('bit_rate', 0)),
        'codec':      video_stream.get('codec_name', 'unknown'),
        'size_bytes': int(fmt.get('size', 0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MÃ HÓA VIDEO
# ─────────────────────────────────────────────────────────────────────────────

def encode_video(
    input_path:  str,
    output_path: str,
    preset:      str = 'medium',
    crf:         int = 23,
) -> float:
    """
    Mã hóa video H.264 bằng FFmpeg.

    H.264 có 2 tham số quan trọng nhất:
      preset — điều chỉnh tốc độ encoder vs hiệu quả nén
               ultrafast: nhanh nhất, tệ nhất về nén
               veryslow : chậm nhất, tốt nhất về nén
      crf    — Constant Rate Factor (0–51)
               0  → lossless (không mất dữ liệu, file rất to)
               18 → chất lượng rất cao, mắt người hầu như không phân biệt được
               23 → mặc định, cân bằng tốt
               28 → chất lượng trung bình, file nhỏ
               51 → tệ nhất

    Args:
        input_path:  Đường dẫn file video đầu vào
        output_path: Đường dẫn file video đầu ra
        preset:      Một trong H264_PRESETS
        crf:         0–51 (thấp = tốt hơn)

    Returns:
        Thời gian mã hóa tính bằng giây.
    """
    # Tạo thư mục output nếu chưa có
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cmd = [
        'ffmpeg',
        '-y',                # ghi đè nếu file đã tồn tại
        '-i',  input_path,   # input
        '-c:v', 'libx264',   # codec video = H.264
        '-preset', preset,   # preset tốc độ
        '-crf',   str(crf),  # chất lượng không đổi (Constant Rate Factor)
        '-c:a',  'copy',     # audio: giữ nguyên, không re-encode
        output_path,
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    if result.returncode != 0:
        # Lấy 800 ký tự cuối của stderr để báo lỗi
        raise RuntimeError(f"FFmpeg lỗi (preset={preset}):\n{result.stderr[-800:]}")

    return elapsed


# ─────────────────────────────────────────────────────────────────────────────
# METRICS CHẤT LƯỢNG
# ─────────────────────────────────────────────────────────────────────────────

def calc_psnr(reference_path: str, encoded_path: str, max_sec: float = 20.0) -> float | None:
    """
    Tính PSNR (Peak Signal-to-Noise Ratio).

    PSNR đo sự sai khác giữa video gốc và video đã nén theo đơn vị dB.
      > 40 dB  — rất tốt, mắt người không phân biệt được
      30–40 dB — tốt, artifact khó nhìn thấy
      < 30 dB  — kém, artifact rõ ràng

    Để tính nhanh, chỉ dùng tối đa max_sec giây đầu của video.

    Returns: giá trị PSNR (float) hoặc None nếu lỗi.
    """
    info     = get_video_info(reference_path)
    duration = min(info['duration'], max_sec) if info else max_sec

    cmd = [
        'ffmpeg', '-y',
        '-i', reference_path,
        '-i', encoded_path,
        '-t', str(duration),
        # psnr filter: so sánh 2 input, ghi kết quả vào stdout (-)
        '-lavfi', 'psnr=stats_file=-',
        '-f', 'null', '-',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # Tìm dòng chứa "average:" trong stderr
        # Ví dụ: "PSNR y:47.12 u:52.31 v:53.24 average:47.92 min:43.28 max:inf"
        for line in result.stderr.split('\n'):
            m = re.search(r'average:(\d+\.?\d*)', line)
            if m:
                return float(m.group(1))
    except Exception as e:
        print(f"[utils] calc_psnr lỗi: {e}")

    return None


def calc_ssim(reference_path: str, encoded_path: str, max_sec: float = 20.0) -> float | None:
    """
    Tính SSIM (Structural Similarity Index Measure).

    SSIM đo độ tương đồng cấu trúc giữa 2 ảnh/video (0.0 → 1.0).
      ≈ 1.0 — gần như giống hệt nhau
      ≈ 0.9 — tốt
      < 0.8 — kém

    Returns: giá trị SSIM (float) hoặc None nếu lỗi.
    """
    info     = get_video_info(reference_path)
    duration = min(info['duration'], max_sec) if info else max_sec

    cmd = [
        'ffmpeg', '-y',
        '-i', reference_path,
        '-i', encoded_path,
        '-t', str(duration),
        '-lavfi', 'ssim=stats_file=-',
        '-f', 'null', '-',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # Tìm dòng chứa "All:" trong stderr
        # Ví dụ: "SSIM Y:0.998447 U:0.999216 V:0.999119 All:0.998650 (28.745)"
        for line in result.stderr.split('\n'):
            m = re.search(r'All:(\d+\.?\d*)', line)
            if m:
                return float(m.group(1))
    except Exception as e:
        print(f"[utils] calc_ssim lỗi: {e}")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# METRICS FILE / BITRATE / FPS
# ─────────────────────────────────────────────────────────────────────────────

def get_size_mb(file_path: str) -> float:
    """Trả về kích thước file tính bằng MB."""
    return os.path.getsize(file_path) / (1024 * 1024)


def get_compression_ratio(original_mb: float, encoded_mb: float) -> float:
    """
    Tỉ lệ nén = kích thước gốc / kích thước đã nén.
    VD: 3.5 → file nhỏ hơn 3.5 lần so với gốc.
    """
    if encoded_mb <= 0:
        return 0.0
    return round(original_mb / encoded_mb, 2)


def get_encoding_fps(input_path: str, encoding_time_sec: float) -> float:
    """
    FPS mã hóa = tổng số frame video / thời gian mã hóa.

    Lưu ý: đây là TỐC ĐỘ XỬ LÝ của encoder, khác với FPS phát lại.
      - preset ultrafast: encoding FPS >> playback FPS (rất nhanh)
      - preset veryslow:  encoding FPS < playback FPS (chậm hơn thời gian thực)
    """
    info = get_video_info(input_path)
    if not info or encoding_time_sec <= 0:
        return 0.0
    total_frames = info['fps'] * info['duration']
    return round(total_frames / encoding_time_sec, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MÔ PHỎNG NÉN FRAME (DÙNG CHO WEBCAM LIVE)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_frame_compression(frame: np.ndarray, preset: str) -> tuple[np.ndarray, float, float]:
    """
    Mô phỏng hiệu ứng mã hóa H.264 trên 1 frame ảnh dùng JPEG compression.

    Vì H.264 encoding thật cần nhiều frame liên tiếp (GOP) và quá chậm cho
    real-time trong Python, ta dùng JPEG làm proxy:
      - JPEG quality thấp  ≈ preset ultrafast (nhanh, chất lượng kém)
      - JPEG quality cao   ≈ preset veryslow  (chậm, chất lượng tốt)

    Args:
        frame:  Frame ảnh (BGR numpy array từ OpenCV)
        preset: Tên preset H.264

    Returns:
        (encoded_frame, compressed_kb, psnr_estimate)
          encoded_frame   — frame sau khi nén + giải nén lại (để hiển thị)
          compressed_kb   — kích thước frame sau nén (KB)
          psnr_estimate   — ước tính PSNR của frame này
    """
    import cv2

    quality = PRESET_TO_JPEG_QUALITY.get(preset, 70)

    # Encode → buffer JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer      = cv2.imencode('.jpg', frame, encode_params)

    # Decode lại để hiển thị artifact
    encoded_frame  = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    compressed_kb  = len(buffer) / 1024

    # Ước tính PSNR từ quality (công thức xấp xỉ kinh nghiệm)
    # PSNR thực tế sẽ được tính chính xác hơn trong Module 1
    psnr_estimate  = 25 + quality * 0.2

    return encoded_frame, compressed_kb, psnr_estimate


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK NHIỀU PRESET
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    input_path:        str,
    output_dir:        str,
    presets:           list[str],
    crf:               int = 23,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Mã hóa video với nhiều preset H.264, tính toán đầy đủ metrics.

    Args:
        input_path:        File video gốc
        output_dir:        Thư mục lưu file đã nén
        presets:           Danh sách preset cần test (VD: ['ultrafast', 'medium', 'slow'])
        crf:               CRF value (0–51)
        progress_callback: Hàm callback(step, total, preset_name) để update progress bar

    Returns:
        pandas DataFrame với các cột:
          Preset, CRF, Encoding FPS, Encoding Time (s),
          Original Size (MB), Encoded Size (MB), Compression Ratio,
          Bitrate (kbps), PSNR (dB), SSIM
    """
    os.makedirs(output_dir, exist_ok=True)

    original_mb   = get_size_mb(input_path)
    original_info = get_video_info(input_path)
    results       = []

    for step, preset in enumerate(presets):
        # Thông báo tiến độ
        if progress_callback:
            progress_callback(step, len(presets), preset)

        out_path = os.path.join(output_dir, f'enc_{preset}_crf{crf}.mp4')

        try:
            # ── 1. Mã hóa ──────────────────────────────────────────────────
            enc_time = encode_video(input_path, out_path, preset, crf)

            # ── 2. Kích thước & tỉ lệ nén ──────────────────────────────────
            enc_mb       = get_size_mb(out_path)
            comp_ratio   = get_compression_ratio(original_mb, enc_mb)

            # ── 3. FPS mã hóa ───────────────────────────────────────────────
            enc_fps      = get_encoding_fps(input_path, enc_time)

            # ── 4. Bitrate ──────────────────────────────────────────────────
            enc_info     = get_video_info(out_path)
            bitrate_kbps = enc_info['bitrate'] / 1000 if enc_info else 0

            # ── 5. PSNR & SSIM (chỉ tính 20s đầu để tiết kiệm thời gian) ───
            psnr = calc_psnr(input_path, out_path, max_sec=20)
            ssim = calc_ssim(input_path, out_path, max_sec=20)

            results.append({
                'Preset':             preset,
                'CRF':                crf,
                'Encoding FPS':       enc_fps,
                'Encoding Time (s)':  round(enc_time, 2),
                'Original Size (MB)': round(original_mb, 2),
                'Encoded Size (MB)':  round(enc_mb, 2),
                'Compression Ratio':  comp_ratio,
                'Bitrate (kbps)':     round(bitrate_kbps, 1),
                'PSNR (dB)':          round(psnr, 2) if psnr else None,
                'SSIM':               round(ssim, 4) if ssim else None,
                '_output_path':       out_path,   # dùng nội bộ
            })

        except Exception as e:
            print(f"[benchmark] Lỗi preset '{preset}': {e}")
            results.append({
                'Preset':             preset,
                'CRF':                crf,
                'Encoding FPS':       0,
                'Encoding Time (s)':  0,
                'Original Size (MB)': round(original_mb, 2),
                'Encoded Size (MB)':  0,
                'Compression Ratio':  0,
                'Bitrate (kbps)':     0,
                'PSNR (dB)':          None,
                'SSIM':               None,
                '_output_path':       None,
                'Error':              str(e),
            })

    # Sắp xếp kết quả theo thứ tự preset (nhanh → chậm)
    preset_order = {p: i for i, p in enumerate(H264_PRESETS)}
    df = pd.DataFrame(results)
    if not df.empty:
        df['_order'] = df['Preset'].map(preset_order).fillna(99)
        df = df.sort_values('_order').drop(columns=['_order']).reset_index(drop=True)

    return df
