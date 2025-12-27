# Hindi Speech-to-Text Backend (Video Input)

## Overview
This backend module processes a video file, extracts audio using FFmpeg,
and generates a Hindi speech-to-text transcript with timestamps using
OpenAI Whisper.

## Features
- Video to audio extraction (FFmpeg)
- Hindi-only speech transcription
- Timestamped segments
- Output formats: JSON, SRT, TXT
- CPU/GPU auto-detection

## Requirements
- Python 3.11
- FFmpeg (added to PATH)
- Python packages:
  - openai-whisper
  - torch

## Folder Structure
