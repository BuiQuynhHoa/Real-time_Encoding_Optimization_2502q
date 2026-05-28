"""
run_benchmark_cli.py — Chạy benchmark Module 1 từ command line
===============================================================
Dùng khi muốn test nhanh không cần mở browser Streamlit.

Cách dùng:
    python run_benchmark_cli.py --help
    python run_benchmark_cli.py --input datasets/raw/test_720p_motion.mp4
    python run_benchmark_cli.py --input video.mp4 --presets ultrafast medium slow --crf 23
"""

import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

# Thêm thư mục gốc vào path để import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import H264_PRESETS, check_ffmpeg, get_size_mb, get_video_info, run_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark H.264 encoding với nhiều preset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python run_benchmark_cli.py --input datasets/raw/test_720p_motion.mp4
  python run_benchmark_cli.py --input video.mp4 --presets ultrafast medium slow --crf 18
  python run_benchmark_cli.py --input video.mp4 --presets ultrafast veryfast medium slow veryslow --output results/my_results.csv
        """
    )

    parser.add_argument('--input',   required=True,  help="Đường dẫn file video đầu vào")
    parser.add_argument('--output',  default=None,   help="Đường dẫn file CSV kết quả (mặc định: ./results/benchmark_<timestamp>.csv)")
    parser.add_argument('--presets', nargs='+', default=['ultrafast', 'veryfast', 'medium', 'slow'],
                        choices=H264_PRESETS, help="Danh sách preset H.264 cần test")
    parser.add_argument('--crf',     type=int, default=23, help="CRF value (0–51, mặc định 23)")

    args = parser.parse_args()

    # ── Kiểm tra đầu vào ────────────────────────────────────────────────────
    if not os.path.exists(args.input):
        print(f"File không tồn tại: {args.input}")
        sys.exit(1)

    if not check_ffmpeg():
        print("FFmpeg chưa được cài đặt! Xem README.md để hướng dẫn cài.")
        sys.exit(1)

    # ── Thông tin video gốc ──────────────────────────────────────────────────
    print(f"\n📹 Video đầu vào: {args.input}")
    info = get_video_info(args.input)
    if info:
        print(f"   Độ phân giải : {info['width']}×{info['height']}")
        print(f"   FPS          : {info['fps']}")
        print(f"   Thời lượng   : {info['duration']:.1f} giây")
        print(f"   Kích thước   : {get_size_mb(args.input):.2f} MB")
        print(f"   Codec gốc   : {info['codec']}")

    print(f"\n Preset sẽ test : {args.presets}")
    print(f"   CRF           : {args.crf}")
    print(f"   Số lần encode : {len(args.presets)}")
    print()

    # ── Thư mục output ───────────────────────────────────────────────────────
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="enc_cli_")

    # ── Callback hiển thị tiến độ ─────────────────────────────────────────────
    pbar = tqdm(total=len(args.presets), desc="Encoding", unit="preset")

    def progress_cb(step, total, preset_name):
        pbar.set_description(f"[{preset_name}]")
        pbar.update(0 if step == 0 else 1)

    # ── Chạy benchmark ───────────────────────────────────────────────────────
    df = run_benchmark(
        input_path        = args.input,
        output_dir        = tmp_dir,
        presets           = args.presets,
        crf               = args.crf,
        progress_callback = progress_cb,
    )
    pbar.update(1)  # Cập nhật lần cuối
    pbar.close()

    # ── In kết quả ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("KẾT QUẢ BENCHMARK")
    print("="*70)

    # Các cột hiển thị ra terminal
    display_cols = [
        'Preset', 'Encoding FPS', 'Encoding Time (s)',
        'Encoded Size (MB)', 'Compression Ratio',
        'Bitrate (kbps)', 'PSNR (dB)', 'SSIM',
    ]
    display_df = df[[c for c in display_cols if c in df.columns]]
    print(display_df.to_string(index=False))
    print()

    # Nhận xét nhanh
    if 'Encoding FPS' in df.columns and len(df) >= 2:
        fastest = df.loc[df['Encoding FPS'].idxmax(), 'Preset']
        slowest = df.loc[df['Encoding FPS'].idxmin(), 'Preset']
        print(f"Nhanh nhất : {fastest} ({df['Encoding FPS'].max():.1f} FPS)")
        print(f"Chậm nhất  : {slowest} ({df['Encoding FPS'].min():.1f} FPS)")

    if 'Compression Ratio' in df.columns and len(df) >= 2:
        best_compress = df.loc[df['Compression Ratio'].idxmax(), 'Preset']
        print(f"Nén tốt nhất: {best_compress} (ratio = {df['Compression Ratio'].max():.2f}×)")

    # ── Lưu CSV ──────────────────────────────────────────────────────────────
    if args.output is None:
        import datetime
        os.makedirs("results", exist_ok=True)
        ts         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/benchmark_{ts}_crf{args.crf}.csv"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_cols = [c for c in df.columns if not c.startswith('_')]
    df[save_cols].to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nKết quả đã lưu: {args.output}")


if __name__ == '__main__':
    main()
