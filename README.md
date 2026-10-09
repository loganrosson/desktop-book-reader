# Desktop Book Reader (AI Text-to-Speech)

A desktop app that reads PDF and Word documents aloud using the [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) neural text-to-speech model, running fully locally on the CPU.

## Features
- Open **PDF** (via PyMuPDF) or **.docx** files and choose a page range
- Read aloud from a selection or from the cursor, with live sentence highlighting
- Pause / resume / stop, adjustable reading speed
- Export audiobooks to **WAV or MP3**, generated in parallel across multiple threads
- Modern dark UI built with CustomTkinter
- Downloads the Kokoro model and voices automatically on first run

## Setup
```bash
pip install -r requirements.txt
python Desktop_Book_App_Complete.py
```
MP3 export requires [FFmpeg](https://ffmpeg.org/) on your PATH (used by `pydub`).

## Tech
Python · CustomTkinter · ONNX Runtime · Kokoro TTS · NumPy · sounddevice · PyMuPDF · python-docx · pydub
