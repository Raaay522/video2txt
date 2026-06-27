# video2txt — 影片語音轉逐字稿

## 功能

- 將影片的語音轉成帶 `[hh:mm:ss]` 時間戳的逐字稿 `.txt`。
- 使用本地 [faster-whisper](https://github.com/SYSTRAN/faster-whisper)(Whisper),不需上傳雲端、不需另裝系統 ffmpeg。
- 內建 VAD 過濾無聲段落,減少幻覺字。
- 自動偵測運算裝置:有 NVIDIA GPU(CUDA)就用 GPU,否則用 CPU。
- 支援多語言(預設中文 `zh`,可指定其他語言或自動偵測)。
- 支援一次批次處理多個影片,各自輸出同名 `.txt`。
- 轉錄過程即時印出每一行進度。

## 專案結構

```
video2txt/
├── video2txt.py       # 主程式(CLI)
├── requirements.txt   # 相依套件
├── .gitignore
└── README.md
```

## 安裝指令

```bash
pip install -r requirements.txt
```

> 若要使用 GPU 加速,請另依顯卡到 https://pytorch.org 安裝對應 CUDA 版本的 torch。
> 首次執行會自動下載模型(`large-v3` 約 3GB)。

## 執行指令

```bash
# 最基本:轉一個影片,輸出同名 .txt
python video2txt.py 影片.mp4

# 指定輸出檔名(僅單檔有效)
python video2txt.py 影片.mp4 -o 逐字稿.txt

# 批次處理(各自輸出同名 txt)
python video2txt.py a.mp4 b.mp4 c.mp4

# 換較快的小模型(沒 GPU 時 large-v3 會很慢)
python video2txt.py 影片.mp4 --model medium

# 指定運算裝置與精度
python video2txt.py 影片.mp4 --device cuda --compute-type float16

# 自動偵測語言(預設是 zh 中文)
python video2txt.py 影片.mp4 --language auto
```
