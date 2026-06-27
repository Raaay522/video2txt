#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video2txt — 影片語音轉逐字稿(帶時間戳)
使用 faster-whisper(本地 Whisper),預設 large-v3 模型。

用法:
    python video2txt.py 影片.mp4
    python video2txt.py 影片.mp4 -o 輸出.txt
    python video2txt.py 影片.mp4 --model medium --device cuda
    python video2txt.py *.mp4               # 批次處理多檔

輸出:每行帶 [hh:mm:ss] 時間戳的逐字稿 .txt
"""

import argparse
import sys
from pathlib import Path


def fmt_ts(seconds: float) -> str:
    """把秒數轉成 [hh:mm:ss] 字串。"""
    if seconds < 0:
        seconds = 0
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def transcribe_one(model, video_path: Path, language: str, out_path: Path) -> None:
    """轉錄單一影片並寫出帶時間戳的 txt。"""
    print(f"\n[轉錄中] {video_path.name}")

    segments, info = model.transcribe(
        str(video_path),
        language=None if language == "auto" else language,
        vad_filter=True,                       # 過濾無聲段落,減少幻覺
        vad_parameters={"min_silence_duration_ms": 500},
    )

    detected = getattr(info, "language", language)
    duration = getattr(info, "duration", 0) or 0
    print(f"  偵測語言:{detected}  影片長度:{fmt_ts(duration)}")

    lines = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        line = f"{fmt_ts(seg.start)} {text}"
        lines.append(line)
        # 即時印出進度,讓使用者看到正在跑
        print(f"  {line}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[完成] 已輸出 → {out_path}  (共 {len(lines)} 行)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="影片語音轉逐字稿(帶時間戳),使用本地 faster-whisper。"
    )
    parser.add_argument("videos", nargs="+", help="影片檔路徑(可多個)")
    parser.add_argument(
        "-o", "--output",
        help="輸出 txt 路徑(僅單一影片時有效;多檔時自動以原檔名 .txt 輸出)",
    )
    parser.add_argument(
        "--model", default="large-v3",
        help="Whisper 模型:tiny/base/small/medium/large-v3(預設 large-v3)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="運算裝置(預設 auto:有 CUDA 就用 GPU,否則 CPU)",
    )
    parser.add_argument(
        "--compute-type", default="auto",
        help="運算精度:auto/int8/int8_float16/float16/float32"
             "(CPU 建議 int8,GPU 建議 float16)",
    )
    parser.add_argument(
        "--language", default="zh",
        help="語言代碼,如 zh(中文)、en(英文)、auto(自動偵測)。預設 zh",
    )
    args = parser.parse_args()

    # 延遲匯入,讓 --help 不必先載入大套件
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("錯誤:尚未安裝 faster-whisper。請先執行:", file=sys.stderr)
        print("    pip install -r requirements.txt", file=sys.stderr)
        return 1

    # 解析裝置與精度
    device = args.device
    compute_type = args.compute_type
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    print(f"載入模型 {args.model}(device={device}, compute_type={compute_type})…")
    print("※ 首次使用會自動下載模型,large-v3 約 3GB,請耐心等候。")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    # 展開輸入檔案
    video_paths = [Path(v) for v in args.videos]
    missing = [p for p in video_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"錯誤:找不到檔案 {p}", file=sys.stderr)
        return 1

    if args.output and len(video_paths) > 1:
        print("警告:輸入多檔,忽略 -o,改用各自原檔名輸出。", file=sys.stderr)

    for vp in video_paths:
        if args.output and len(video_paths) == 1:
            out_path = Path(args.output)
        else:
            out_path = vp.with_suffix(".txt")
        transcribe_one(model, vp, args.language, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
