"""
generate_dataset.py — Tạo 3 video mẫu synthetic bằng OpenCV
=============================================================
Môn:     Nén và Mã hóa Đa phương tiện
Mã code: 2502q

KHÔNG CẦN TẢI VIDEO TỪ INTERNET.
Script này tạo ra 3 video mẫu phù hợp với mục đích test H.264 encoder:

  1. test_720p_motion.mp4   — Cảnh ĐỘNG nhiều: nhiều vật thể di chuyển,
                              màu sắc thay đổi liên tục → encoder phải
                              xử lý nhiều inter-frame prediction → file to hơn.

  2. test_1080p_static.mp4  — Cảnh TĨNH: nền gradient nhẹ, ít thay đổi
                              theo thời gian → encoder nén rất tốt vì temporal
                              redundancy cao → file nhỏ hơn rõ rệt.

  3. test_1080p_mixed.mp4   — Cảnh HỖN HỢP: nửa đầu tĩnh, nửa sau động
                              → dùng để test encoder ở cả 2 chế độ.

Tại sao dùng synthetic video?
  - Không phụ thuộc internet hoặc bản quyền
  - Kiểm soát hoàn toàn đặc tính video (motion nhiều/ít)
  - Tái tạo hoàn toàn được (reproducible)
  - Phù hợp mục đích học thuật

Cách dùng:
    python generate_dataset.py
    python generate_dataset.py --duration 20   # Video 20 giây
    python generate_dataset.py --fps 30
    python generate_dataset.py --no-720p       # Bỏ qua video 720p
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH MẶC ĐỊNH
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FPS      = 30
DEFAULT_DURATION = 15   # giây — đủ để benchmark mà không quá lâu tạo

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "raw")

VIDEOS = {
    "test_720p_motion":   (1280, 720),
    "test_1080p_static":  (1920, 1080),
    "test_1080p_mixed":   (1920, 1080),
}


# ─────────────────────────────────────────────────────────────────────────────
# HÀM TIỆN ÍCH
# ─────────────────────────────────────────────────────────────────────────────

def progress_bar(current: int, total: int, prefix: str = "") -> None:
    """In thanh tiến độ đơn giản ra terminal."""
    pct  = current / total
    done = int(pct * 30)
    bar  = "█" * done + "░" * (30 - done)
    print(f"\r  {prefix} [{bar}] {current}/{total} frames", end="", flush=True)


def write_text_centered(
    frame:    np.ndarray,
    text:     str,
    y:        int,
    scale:    float = 1.0,
    color:    tuple = (255, 255, 255),
    thickness: int  = 2,
) -> None:
    """Viết text căn giữa theo chiều ngang lên frame."""
    font      = cv2.FONT_HERSHEY_DUPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x         = (frame.shape[1] - w) // 2
    # Shadow đen để dễ đọc trên mọi nền
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0),       thickness + 2)
    cv2.putText(frame, text, (x,     y),     font, scale, color, thickness)


def save_video_ffmpeg(
    tmp_path: str,
    out_path: str,
    fps:      int,
) -> bool:
    """
    Dùng FFmpeg để re-encode video từ định dạng raw (mp4v) sang H.264 chuẩn.
    Đảm bảo file tương thích với mọi player và ffprobe đọc được bitrate chính xác.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i",       tmp_path,
        "-c:v",     "libx264",
        "-preset",  "fast",      # encode nhanh, chất lượng đủ tốt cho video test
        "-crf",     "18",        # chất lượng cao để làm reference cho benchmark
        "-r",       str(fps),
        "-pix_fmt", "yuv420p",   # đảm bảo tương thích tối đa
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO 1: CẢNH ĐỘNG NHIỀU — 720p
# ─────────────────────────────────────────────────────────────────────────────

def generate_motion_video(
    out_path: str,
    width:    int = 1280,
    height:   int = 720,
    fps:      int = DEFAULT_FPS,
    duration: int = DEFAULT_DURATION,
) -> None:
    """
    Tạo video 720p với NHIỀU CHUYỂN ĐỘNG:
      - Nhiều quả bóng bay nảy quanh màn hình với màu sắc đa dạng
      - Nền thay đổi màu theo thời gian (hue rotating)
      - Text timestamp nhảy số mỗi giây
      - Noise ngẫu nhiên nhỏ để tăng tính thực tế
      
    Mục đích: Encoder phải xử lý nhiều inter-frame motion vectors
    → Inter-prediction kém hiệu quả → file to hơn → dễ thấy
    sự khác biệt giữa preset ultrafast và slow.
    """
    tmp_path    = out_path + ".tmp.mp4"
    total_frames = fps * duration
    fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
    writer       = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Không tạo được VideoWriter: {tmp_path}")

    # ── Khởi tạo các quả bóng ────────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)   # seed cố định → reproducible

    N_BALLS = 18   # số lượng đủ để tạo nhiều motion
    balls   = []
    for i in range(N_BALLS):
        balls.append({
            "x":   float(rng.integers(60, width  - 60)),
            "y":   float(rng.integers(60, height - 60)),
            "vx":  float(rng.uniform(3.5, 8.0)) * rng.choice([-1, 1]),
            "vy":  float(rng.uniform(3.5, 8.0)) * rng.choice([-1, 1]),
            "r":   int(rng.integers(20, 48)),
            # Màu ngẫu nhiên, đủ sáng để thấy rõ
            "hue": int(rng.integers(0, 180)),
        })

    print(f"\n[Video 1] Tạo {total_frames} frames ({width}×{height}, {fps}fps)...")

    for f in range(total_frames):
        t   = f / fps   # thời gian tính bằng giây

        # ── Nền: gradient màu xoay theo thời gian ────────────────────────────
        hue_bg    = int((t * 15) % 180)   # xoay chậm
        hsv_frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Tạo gradient ngang
        for col in range(0, width, 4):   # bước 4 pixel để nhanh hơn
            h = (hue_bg + col * 60 // width) % 180
            hsv_frame[:, col:col+4, 0] = h
        hsv_frame[:, :, 1] = 180   # saturation
        hsv_frame[:, :, 2] = 60    # value (tối để bóng nổi bật)

        frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)

        # Thêm chút Gaussian noise để tăng texture (thực tế hơn, khó nén hơn)
        noise = rng.integers(-12, 12, (height, width, 3), dtype=np.int8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # ── Cập nhật và vẽ các quả bóng ──────────────────────────────────────
        for ball in balls:
            # Di chuyển
            ball["x"] += ball["vx"]
            ball["y"] += ball["vy"]

            # Nảy ở biên
            if ball["x"] - ball["r"] < 0 or ball["x"] + ball["r"] > width:
                ball["vx"] *= -1
                ball["hue"] = (ball["hue"] + 30) % 180   # đổi màu khi nảy
                ball["x"]   = max(ball["r"], min(width  - ball["r"], ball["x"]))
            if ball["y"] - ball["r"] < 0 or ball["y"] + ball["r"] > height:
                ball["vy"] *= -1
                ball["hue"] = (ball["hue"] + 30) % 180
                ball["y"]   = max(ball["r"], min(height - ball["r"], ball["y"]))

            # Màu HSV → BGR
            hsv_color = np.uint8([[[ball["hue"], 255, 255]]])
            bgr_color = tuple(int(c) for c in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0])

            cx, cy, r = int(ball["x"]), int(ball["y"]), ball["r"]

            # Glow effect: vòng ngoài mờ
            cv2.circle(frame, (cx, cy), r + 6, bgr_color, -1, cv2.LINE_AA)
            # Core sáng
            bright = tuple(min(255, int(c * 1.4)) for c in bgr_color)
            cv2.circle(frame, (cx, cy), r, bright, -1, cv2.LINE_AA)
            # Highlight nhỏ trên cùng
            cv2.circle(frame, (cx - r//4, cy - r//4), r//4, (255, 255, 255), -1, cv2.LINE_AA)

        # ── Overlay thông tin ─────────────────────────────────────────────────
        # Thanh thông tin bên trên
        cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 0), -1)
        write_text_centered(
            frame,
            f"[MOTION VIDEO 720p]  Frame: {f:04d}/{total_frames}  "
            f"Time: {t:.2f}s  Balls: {N_BALLS}",
            y=35, scale=0.7, color=(0, 255, 180),
        )

        # Đồng hồ lớn ở dưới
        cv2.rectangle(frame, (0, height - 55), (width, height), (0, 0, 0), -1)
        write_text_centered(
            frame, f"t = {t:06.2f}s",
            y=height - 12, scale=1.1, color=(200, 200, 255),
        )

        writer.write(frame)
        if f % fps == 0:
            progress_bar(f, total_frames, "Motion")

    writer.release()
    progress_bar(total_frames, total_frames, "Motion")
    print(f"  → Đang convert sang H.264...")

    ok = save_video_ffmpeg(tmp_path, out_path, fps)
    os.remove(tmp_path)

    if ok:
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  {out_path}  ({size_mb:.1f} MB)")
    else:
        print(f"   FFmpeg lỗi, dùng file raw: {tmp_path} → {out_path}")
        if os.path.exists(tmp_path):
            os.rename(tmp_path, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO 2: CẢNH TĨNH — 1080p
# ─────────────────────────────────────────────────────────────────────────────

def generate_static_video(
    out_path: str,
    width:    int = 1920,
    height:   int = 1080,
    fps:      int = DEFAULT_FPS,
    duration: int = DEFAULT_DURATION,
) -> None:
    """
    Tạo video 1080p với ÍT CHUYỂN ĐỘNG:
      - Nền gradient màu xanh tím rất mịn, thay đổi cực kỳ chậm
      - Các đường thẳng trang trí, không di chuyển
      - Chỉ có timestamp thay đổi mỗi giây
      - Không có noise
      
    Mục đích: Inter-frame prediction cực kỳ hiệu quả vì hầu hết pixel
    không đổi → encoder tạo ra file nhỏ hơn nhiều → so sánh rõ với video motion.
    
    Đây chính xác là kiểu video khiến preset slow tỏa sáng:
    tốn thêm thời gian encode nhưng tiết kiệm được rất nhiều bitrate.
    """
    tmp_path     = out_path + ".tmp.mp4"
    total_frames = fps * duration
    fourcc        = cv2.VideoWriter_fourcc(*"mp4v")
    writer        = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Không tạo được VideoWriter: {tmp_path}")

    print(f"\n[Video 2] Tạo {total_frames} frames ({width}×{height}, {fps}fps)...")

    # Tạo nền base một lần (không đổi) → rất ít temporal change
    base_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Gradient ngang xanh-tím
    for x in range(width):
        ratio = x / width
        r = int(15  + ratio * 40)
        g = int(10  + ratio * 20)
        b = int(80  + ratio * 120)
        base_frame[:, x] = [b, g, r]

    # Gradient dọc nhẹ (tối hơn ở trên và dưới)
    for y in range(height):
        factor = 0.7 + 0.3 * math.sin(math.pi * y / height)
        base_frame[y] = np.clip(base_frame[y] * factor, 0, 255).astype(np.uint8)

    # Vẽ các đường trang trí cố định (không thay đổi)
    for i in range(1, 6):
        y_line = height * i // 6
        cv2.line(base_frame, (0, y_line), (width, y_line), (60, 60, 120), 1)
    for i in range(1, 10):
        x_line = width * i // 10
        cv2.line(base_frame, (x_line, 0), (x_line, height), (60, 60, 120), 1)

    # Các hình chữ nhật trang trí (cố định)
    rects = [
        ((width//4 - 200, height//2 - 150), (width//4 + 200, height//2 + 150), (100, 80, 180), 2),
        ((3*width//4 - 200, height//2 - 150), (3*width//4 + 200, height//2 + 150), (80, 160, 100), 2),
        ((width//2 - 250, height//3 - 80), (width//2 + 250, height//3 + 80), (200, 120, 60), 3),
    ]
    for (pt1, pt2, color, thick) in rects:
        cv2.rectangle(base_frame, pt1, pt2, color, thick)

    # Text tĩnh
    write_text_centered(base_frame, "STATIC SCENE — 1080p", y=height//2 - 20,
                        scale=2.0, color=(220, 220, 255), thickness=3)
    write_text_centered(base_frame, "Low Temporal Complexity — Encoder Friendly",
                        y=height//2 + 50, scale=0.9, color=(180, 180, 230))

    for f in range(total_frames):
        t = f / fps

        # Copy nền (hầu hết không đổi)
        frame = base_frame.copy()

        # CHỈ thay đổi nhỏ: breathing effect cực nhẹ trên 2 vòng tròn
        alpha_slow = 0.5 + 0.5 * math.sin(2 * math.pi * t / duration)

        # Vòng tròn trung tâm thay đổi kích thước rất chậm
        r_big = int(90 + 20 * alpha_slow)
        cv2.circle(frame, (width // 2, 3 * height // 4),
                   r_big, (100, 180, 255), 3, cv2.LINE_AA)

        # Timestamp (thay đổi mỗi giây)
        cv2.rectangle(frame, (0, 0), (width, 50), (5, 5, 25), -1)
        write_text_centered(frame,
            f"[STATIC VIDEO 1080p]  Frame: {f:04d}/{total_frames}  Time: {t:.2f}s",
            y=35, scale=0.75, color=(120, 220, 255))

        cv2.rectangle(frame, (0, height - 55), (width, height), (5, 5, 25), -1)
        write_text_centered(frame, f"t = {t:06.2f}s",
                            y=height - 12, scale=1.3, color=(180, 255, 220))

        writer.write(frame)
        if f % fps == 0:
            progress_bar(f, total_frames, "Static")

    writer.release()
    progress_bar(total_frames, total_frames, "Static")
    print(f"  → Đang convert sang H.264...")

    ok = save_video_ffmpeg(tmp_path, out_path, fps)
    os.remove(tmp_path)

    if ok:
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  {out_path}  ({size_mb:.1f} MB)")
    else:
        if os.path.exists(tmp_path):
            os.rename(tmp_path, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO 3: CẢNH HỖN HỢP — 1080p
# ─────────────────────────────────────────────────────────────────────────────

def generate_mixed_video(
    out_path: str,
    width:    int = 1920,
    height:   int = 1080,
    fps:      int = DEFAULT_FPS,
    duration: int = DEFAULT_DURATION,
) -> None:
    """
    Tạo video 1080p HỖN HỢP:
      - 1/3 đầu: TĨNH (giống video 2) — encoder thoải mái
      - 1/3 giữa: CHUYỂN TIẾP — encoder bắt đầu phải xử lý nhiều hơn
      - 1/3 cuối: ĐỘNG MẠNH (giống video 1) — encoder stress test

    Mục đích:
      - Test encoder với cả 2 trường hợp trong 1 video
      - Quan sát biến động bitrate theo thời gian (VBR behavior)
      - Phản ánh thực tế: phim có cả cảnh tĩnh lẫn action
      
    Bonus: Có overlay chỉ rõ "PHASE 1/2/3" để dễ nhận biết
    trong demo slide và demo video.
    """
    tmp_path     = out_path + ".tmp.mp4"
    total_frames = fps * duration
    fourcc        = cv2.VideoWriter_fourcc(*"mp4v")
    writer        = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Không tạo được VideoWriter: {tmp_path}")

    print(f"\n[Video 3] Tạo {total_frames} frames ({width}×{height}, {fps}fps)...")

    rng = np.random.default_rng(seed=99)

    # Khởi tạo particles (chỉ dùng trong phase 3)
    N_PARTICLES = 25
    particles   = []
    for _ in range(N_PARTICLES):
        particles.append({
            "x":  float(rng.integers(0, width)),
            "y":  float(rng.integers(0, height)),
            "vx": float(rng.uniform(-10, 10)),
            "vy": float(rng.uniform(-10, 10)),
            "r":  int(rng.integers(12, 40)),
            "hue": int(rng.integers(0, 180)),
            "trail": [],   # lịch sử vị trí để vẽ đuôi
        })

    phase1_end = total_frames // 3
    phase2_end = 2 * total_frames // 3

    for f in range(total_frames):
        t     = f / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # ── PHASE 1: Tĩnh ────────────────────────────────────────────────────
        if f < phase1_end:
            phase_ratio = f / phase1_end  # 0 → 1

            # Gradient tĩnh màu xanh lá
            for x in range(0, width, 2):
                ratio = x / width
                r = int(10  + ratio * 30)
                g = int(60  + ratio * 80 + phase_ratio * 20)
                b = int(20  + ratio * 30)
                frame[:, x:x+2] = [b, g, r]

            # Vòng tròn đồng tâm cố định
            for ring in range(1, 6):
                cv2.circle(frame, (width//2, height//2),
                           ring * 120, (30, 100 + ring*20, 40), 2, cv2.LINE_AA)

            write_text_centered(frame, "PHASE 1 — STATIC",
                                y=height//2 - 20, scale=2.0,
                                color=(100, 255, 130), thickness=3)
            write_text_centered(frame, "Low motion  |  Encoder compresses well",
                                y=height//2 + 50, scale=0.9, color=(150, 230, 160))

            phase_label = "PHASE 1/3 — STATIC"
            label_color = (80, 255, 100)

        # ── PHASE 2: Chuyển tiếp ─────────────────────────────────────────────
        elif f < phase2_end:
            phase_ratio = (f - phase1_end) / (phase2_end - phase1_end)  # 0 → 1

            # Nền chuyển dần từ xanh sang đỏ/cam
            for x in range(0, width, 2):
                ratio = x / width
                g = int((1 - phase_ratio) * (60 + ratio * 80))
                r = int(phase_ratio       * (80 + ratio * 120))
                b = int(30 + ratio * 60)
                frame[:, x:x+2] = [b, g, r]

            # Sóng sin chuyển động
            wave_speed  = phase_ratio * 5
            for x in range(width):
                wave_y = int(height//2 + 80 * math.sin(2 * math.pi * (x/width * 4 + t * wave_speed)))
                wave_y = max(10, min(height - 10, wave_y))
                cv2.circle(frame, (x, wave_y), 3,
                           (int(50 + phase_ratio * 200), 150, int(200 - phase_ratio * 150)), -1)

            # Các quả cầu xoay
            n_spheres = int(2 + phase_ratio * 8)
            for i in range(n_spheres):
                angle = 2 * math.pi * i / n_spheres + t * (1 + phase_ratio * 3)
                cx    = int(width//2  + (300 + phase_ratio * 200) * math.cos(angle))
                cy    = int(height//2 + (200 + phase_ratio * 150) * math.sin(angle))
                r_sp  = int(20 + phase_ratio * 20)
                if 0 < cx < width and 0 < cy < height:
                    hue_sp  = int((i * 20 + t * 50) % 180)
                    hsv_col = np.uint8([[[hue_sp, 240, 255]]])
                    bgr_col = tuple(int(c) for c in cv2.cvtColor(hsv_col, cv2.COLOR_HSV2BGR)[0][0])
                    cv2.circle(frame, (cx, cy), r_sp, bgr_col, -1, cv2.LINE_AA)

            write_text_centered(frame, "PHASE 2 — TRANSITION",
                                y=height//2 - 30, scale=1.8,
                                color=(255, 200, 80), thickness=3)
            write_text_centered(frame, f"Motion increasing... ({phase_ratio*100:.0f}%)",
                                y=height//2 + 40, scale=0.85, color=(255, 230, 150))

            phase_label = "PHASE 2/3 — TRANSITION"
            label_color = (80, 200, 255)

        # ── PHASE 3: Động mạnh ────────────────────────────────────────────────
        else:
            phase_ratio = (f - phase2_end) / (total_frames - phase2_end)  # 0 → 1

            # Nền tối với noise
            noise = rng.integers(0, 30, (height, width, 3), dtype=np.uint8)
            frame = noise.copy()

            # Cập nhật và vẽ particles với đuôi
            for ball in particles:
                # Tăng tốc theo phase_ratio (càng về cuối càng nhanh)
                speed_mult = 1.0 + phase_ratio * 1.5
                ball["x"] += ball["vx"] * speed_mult
                ball["y"] += ball["vy"] * speed_mult

                if ball["x"] < 0 or ball["x"] >= width:
                    ball["vx"] *= -1
                    ball["hue"] = (ball["hue"] + 25) % 180
                    ball["x"]   = max(0, min(width - 1, ball["x"]))
                if ball["y"] < 0 or ball["y"] >= height:
                    ball["vy"] *= -1
                    ball["hue"] = (ball["hue"] + 25) % 180
                    ball["y"]   = max(0, min(height - 1, ball["y"]))

                # Lưu trail
                ball["trail"].append((int(ball["x"]), int(ball["y"])))
                if len(ball["trail"]) > 15:
                    ball["trail"].pop(0)

                # Vẽ trail (đuôi mờ dần)
                for ti, (tx, ty) in enumerate(ball["trail"]):
                    alpha = ti / len(ball["trail"])
                    hsv_c = np.uint8([[[ball["hue"], 200, int(200 * alpha)]]])
                    bgr_c = tuple(int(c) for c in cv2.cvtColor(hsv_c, cv2.COLOR_HSV2BGR)[0][0])
                    tr_r  = max(1, int(ball["r"] * alpha * 0.6))
                    if 0 <= tx < width and 0 <= ty < height:
                        cv2.circle(frame, (tx, ty), tr_r, bgr_c, -1, cv2.LINE_AA)

                # Vẽ ball chính
                cx, cy = int(ball["x"]), int(ball["y"])
                hsv_c  = np.uint8([[[ball["hue"], 255, 255]]])
                bgr_c  = tuple(int(c) for c in cv2.cvtColor(hsv_c, cv2.COLOR_HSV2BGR)[0][0])
                cv2.circle(frame, (cx, cy), ball["r"], bgr_c, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx - ball["r"]//4, cy - ball["r"]//4),
                           ball["r"]//3, (255, 255, 255), -1, cv2.LINE_AA)

            # Flash effect (thay đổi nhanh → harder to encode)
            if f % 8 == 0:
                flash = rng.integers(10, 50, (3,), dtype=np.uint8)
                frame = np.clip(frame.astype(np.int16) + flash, 0, 255).astype(np.uint8)

            write_text_centered(frame, "PHASE 3 — HIGH MOTION",
                                y=height//2, scale=1.8,
                                color=(255, 80, 80), thickness=3)

            phase_label = "PHASE 3/3 — HIGH MOTION"
            label_color = (80, 80, 255)

        # ── Overlay chung ─────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (width, 52), (0, 0, 0), -1)
        write_text_centered(frame,
            f"[MIXED VIDEO 1080p]  {phase_label}  Frame: {f:04d}  t: {t:.2f}s",
            y=35, scale=0.72, color=label_color)

        cv2.rectangle(frame, (0, height - 55), (width, height), (0, 0, 0), -1)
        write_text_centered(frame, f"t = {t:06.2f}s",
                            y=height - 12, scale=1.3, color=(200, 200, 255))

        # Progress bar trong video (ở dưới cùng)
        bar_x = int(width * f / total_frames)
        cv2.rectangle(frame, (0, height - 5), (bar_x, height), label_color, -1)

        writer.write(frame)
        if f % fps == 0:
            progress_bar(f, total_frames, "Mixed ")

    writer.release()
    progress_bar(total_frames, total_frames, "Mixed ")
    print(f"  → Đang convert sang H.264...")

    ok = save_video_ffmpeg(tmp_path, out_path, fps)
    os.remove(tmp_path)

    if ok:
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  {out_path}  ({size_mb:.1f} MB)")
    else:
        if os.path.exists(tmp_path):
            os.rename(tmp_path, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tạo 3 video mẫu synthetic cho benchmark H.264",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python generate_dataset.py                    # Tạo cả 3 video, 15 giây, 30fps
  python generate_dataset.py --duration 10      # Video ngắn hơn (test nhanh)
  python generate_dataset.py --duration 30      # Video dài hơn (kết quả chính xác hơn)
  python generate_dataset.py --fps 25           # 25 fps
  python generate_dataset.py --skip-motion      # Bỏ qua video 1 (đã có rồi)
        """
    )
    parser.add_argument("--duration",     type=int,  default=DEFAULT_DURATION,
                        help=f"Thời lượng mỗi video (giây, mặc định {DEFAULT_DURATION})")
    parser.add_argument("--fps",          type=int,  default=DEFAULT_FPS,
                        help=f"Frames per second (mặc định {DEFAULT_FPS})")
    parser.add_argument("--output-dir",   type=str,  default=OUTPUT_DIR,
                        help="Thư mục lưu video")
    parser.add_argument("--skip-motion",  action="store_true", help="Bỏ qua video 1 (motion)")
    parser.add_argument("--skip-static",  action="store_true", help="Bỏ qua video 2 (static)")
    parser.add_argument("--skip-mixed",   action="store_true", help="Bỏ qua video 3 (mixed)")
    parser.add_argument("--overwrite",    action="store_true", help="Ghi đè file đã có")

    args = parser.parse_args()

    # Tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("🎬 TẠO DATASET VIDEO MẪU — Dự án 2502q")
    print("=" * 60)
    print(f"Thư mục : {os.path.abspath(args.output_dir)}")
    print(f"Thời lượng: {args.duration}s  |  FPS: {args.fps}")
    print(f"Tổng frames mỗi video: {args.duration * args.fps}")
    print()

    # Kiểm tra FFmpeg (cần để convert sang H.264 chuẩn)
    has_ffmpeg = True
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except Exception:
        has_ffmpeg = False
        print("    FFmpeg không tìm thấy — video sẽ lưu dạng mp4v (vẫn dùng được).")
        print("    Cài FFmpeg để có kết quả tốt nhất: xem README.md\n")

    t_start = time.time()
    created = []

    # ── Video 1: Motion ───────────────────────────────────────────────────────
    path1 = os.path.join(args.output_dir, "test_720p_motion.mp4")
    if not args.skip_motion:
        if os.path.exists(path1) and not args.overwrite:
            size_mb = os.path.getsize(path1) / (1024 * 1024)
            print(f" Đã có: test_720p_motion.mp4 ({size_mb:.1f} MB) — dùng --overwrite để tạo lại")
        else:
            generate_motion_video(path1, width=1280, height=720,
                                  fps=args.fps, duration=args.duration)
            created.append("test_720p_motion.mp4")
        print()

    # ── Video 2: Static ───────────────────────────────────────────────────────
    path2 = os.path.join(args.output_dir, "test_1080p_static.mp4")
    if not args.skip_static:
        if os.path.exists(path2) and not args.overwrite:
            size_mb = os.path.getsize(path2) / (1024 * 1024)
            print(f" Đã có: test_1080p_static.mp4 ({size_mb:.1f} MB) — dùng --overwrite để tạo lại")
        else:
            generate_static_video(path2, width=1920, height=1080,
                                  fps=args.fps, duration=args.duration)
            created.append("test_1080p_static.mp4")
        print()

    # ── Video 3: Mixed ────────────────────────────────────────────────────────
    path3 = os.path.join(args.output_dir, "test_1080p_mixed.mp4")
    if not args.skip_mixed:
        if os.path.exists(path3) and not args.overwrite:
            size_mb = os.path.getsize(path3) / (1024 * 1024)
            print(f" Đã có: test_1080p_mixed.mp4 ({size_mb:.1f} MB) — dùng --overwrite để tạo lại")
        else:
            generate_mixed_video(path3, width=1920, height=1080,
                                 fps=args.fps, duration=args.duration)
            created.append("test_1080p_mixed.mp4")
        print()

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"Hoàn thành trong {elapsed:.1f}s")
    print()

    for fname, path in [
        ("test_720p_motion.mp4",  path1),
        ("test_1080p_static.mp4", path2),
        ("test_1080p_mixed.mp4",  path3),
    ]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {fname:30s}  {size_mb:6.1f} MB")

    print()
    print("Bước tiếp theo:")
    print("  streamlit run app.py")
    print("  → Tab Module 1 → Upload một trong các file trên → Benchmark!")
    print("=" * 60)


if __name__ == "__main__":
    main()