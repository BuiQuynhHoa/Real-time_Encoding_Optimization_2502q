"""
download_dataset.py — Tải video mẫu về thư mục datasets/raw/
==============================================================
Tải 3 video mẫu miễn phí từ Blender Foundation (Big Buck Bunny)
và Wikimedia Commons — không cần đăng ký, không vi phạm bản quyền.

Cách dùng:
    python download_dataset.py
"""

import os
import sys
import urllib.request

# ─────────────────────────────────────────────────────────────────────────────
# DANH SÁCH VIDEO MẪU
# Nguồn: Blender Foundation (CC BY 3.0) và Wikimedia Commons (CC)
# ─────────────────────────────────────────────────────────────────────────────

VIDEOS = [
    {
        "filename":    "test_720p_motion.mp4",
        "description": "Big Buck Bunny 720p — cảnh động, nhiều chuyển động",
        "url":         "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        "note":        "Video dài ~10 phút, chỉ dùng 30s đầu để test",
        "fallback":    None,
    },
    {
        "filename":    "test_1080p_static.mp4",
        "description": "Elephant Dream 720p — cảnh chuyển động vừa",
        "url":         "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
        "note":        "Video Blender Foundation, CC BY 2.5",
        "fallback":    None,
    },
    {
        "filename":    "test_1080p_mixed.mp4",
        "description": "Subaru Outback sample — cảnh thực tế đa dạng",
        "url":         "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnStreetAndDirt.mp4",
        "note":        "Video sample Google, dùng cho test",
        "fallback":    None,
    },
]


def download_with_progress(url: str, dest_path: str, desc: str) -> bool:
    """Tải file với thanh tiến độ đơn giản."""
    print(f"\nĐang tải: {desc}")
    print(f"   URL     : {url}")
    print(f"   Lưu vào : {dest_path}")

    downloaded = [0]
    last_pct   = [-1]

    def reporthook(count, block_size, total_size):
        if total_size <= 0:
            return
        downloaded[0] = count * block_size
        pct = min(int(downloaded[0] * 100 / total_size), 100)
        if pct != last_pct[0]:
            bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
            mb  = downloaded[0] / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r   [{bar}] {pct}% ({mb:.1f}/{total_mb:.1f} MB)", end='', flush=True)
            last_pct[0] = pct

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"\n   Hoàn thành!")
        return True
    except Exception as e:
        print(f"\n   Lỗi: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def trim_video_ffmpeg(input_path: str, output_path: str, duration_sec: int = 30) -> bool:
    """Cắt video còn duration_sec giây dùng FFmpeg (để tiết kiệm dung lượng)."""
    import subprocess
    cmd = [
        'ffmpeg', '-y',
        '-i',  input_path,
        '-t',  str(duration_sec),
        '-c',  'copy',           # copy stream, không re-encode
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False


def main():
    out_dir = os.path.join("datasets", "raw")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("TẢI VIDEO MẪU CHO DỰ ÁN 2502q")
    print("=" * 60)
    print(f"Thư mục lưu: {os.path.abspath(out_dir)}")
    print()

    # Kiểm tra FFmpeg để trim video
    import subprocess
    has_ffmpeg = True
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except Exception:
        has_ffmpeg = False
        print(" FFmpeg chưa cài — video sẽ không được cắt ngắn.")

    success_count = 0

    for video in VIDEOS:
        dest = os.path.join(out_dir, video["filename"])

        # Bỏ qua nếu đã tồn tại
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f" Đã có: {video['filename']} ({size_mb:.1f} MB) — bỏ qua")
            success_count += 1
            continue

        # Tải về file tạm
        tmp_path = dest + ".tmp"
        ok = download_with_progress(video["url"], tmp_path, video["description"])

        if ok:
            # Nếu có FFmpeg, trim còn 30s để tiết kiệm dung lượng
            if has_ffmpeg:
                print(f"    Cắt còn 30 giây để tiết kiệm dung lượng...")
                trimmed = trim_video_ffmpeg(tmp_path, dest, duration_sec=30)
                os.remove(tmp_path)
                if not trimmed:
                    # Nếu trim lỗi thì giữ nguyên file gốc
                    os.rename(tmp_path, dest) if os.path.exists(tmp_path) else None
            else:
                os.rename(tmp_path, dest)

            if os.path.exists(dest):
                size_mb = os.path.getsize(dest) / (1024 * 1024)
                print(f"   Kích thước: {size_mb:.1f} MB")
                success_count += 1
        else:
            print(f"       Tải thất bại. Bạn có thể tải thủ công từ:")
            print(f"       {video['url']}")
            print(f"       Đổi tên thành: {video['filename']}")
            print(f"       Đặt vào: {out_dir}/")

    print()
    print("=" * 60)
    if success_count == len(VIDEOS):
        print(f"Đã có đủ {success_count}/{len(VIDEOS)} video mẫu!")
        print(f"   Chạy ứng dụng: streamlit run app.py")
    else:
        print(f" Chỉ có {success_count}/{len(VIDEOS)} video mẫu.")
        print("   Xem README.md → mục Dataset để tải thủ công.")
    print("=" * 60)

    print("\nGhi chú về bản quyền:")
    print("   Big Buck Bunny   — Creative Commons Attribution 3.0")
    print("   Elephants Dream  — Creative Commons Attribution 2.5")
    print("   Subaru sample    — Google sample video")
    print("   Tất cả đều miễn phí cho mục đích học thuật.")


if __name__ == '__main__':
    main()
