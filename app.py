import os
import json
import logging
import subprocess
import time
from datetime import datetime
from typing import Dict, Any, Union

import whisper
import torch

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SpeechToText")

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- VIDEO → AUDIO ----------------
def extract_audio_from_video(video_path: str) -> str:
    """
    Extract mono 16kHz WAV audio from video using FFmpeg
    """
    audio_path = os.path.join(
        TEMP_DIR, f"extracted_{int(time.time())}.wav"
    )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            audio_path
        ],
        check=True
    )

    return audio_path


# ---------------- SPEECH TO TEXT ----------------
class SpeechToTextProcessor:
    def __init__(self, model_size: str = "small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model ({model_size}) on {self.device}")
        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        logger.info(f"Transcribing audio: {audio_path}")

        result = self.model.transcribe(
            audio_path,
            language="hi",                       # Force Hindi
            task="transcribe",
            word_timestamps=True,
            initial_prompt="यह एक भ्रष्टाचार से संबंधित बातचीत है।",
            fp16=False
        )

        # Remove silent segments
        result["segments"] = [
            s for s in result["segments"]
            if s.get("no_speech_prob", 1.0) < 0.6
        ]

        return result

    # ---------------- FORMATTERS ----------------
    def format_timestamp(self, seconds: float) -> str:
        ms = int((seconds - int(seconds)) * 1000)
        return f"{int(seconds // 3600):02d}:{int((seconds % 3600)//60):02d}:{int(seconds % 60):02d},{ms:03d}"

    def to_json(self, transcription: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "language": "Hindi",
            "generated_at": datetime.now().isoformat(),
            "segments": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "start_formatted": self.format_timestamp(s["start"]),
                    "end_formatted": self.format_timestamp(s["end"]),
                    "text": s["text"].strip()
                }
                for s in transcription["segments"]
            ],
            "full_text": " ".join(
                s["text"].strip() for s in transcription["segments"]
            )
        }

    def to_srt(self, transcription: Dict[str, Any]) -> str:
        blocks = []
        for i, s in enumerate(transcription["segments"], 1):
            blocks.append(
                f"{i}\n"
                f"{self.format_timestamp(s['start'])} --> {self.format_timestamp(s['end'])}\n"
                f"{s['text'].strip()}\n"
            )
        return "\n".join(blocks)

    def to_txt(self, transcription: Dict[str, Any]) -> str:
        return " ".join(s["text"].strip() for s in transcription["segments"])

    def process_audio(
        self,
        audio_path: str,
        output_format: str
    ) -> Union[str, Dict[str, Any]]:
        transcription = self.transcribe(audio_path)

        if output_format == "srt":
            return self.to_srt(transcription)
        elif output_format == "txt":
            return self.to_txt(transcription)
        else:
            return self.to_json(transcription)


# ---------------- SAVE OUTPUT ----------------
def save_output(result, filename: str, output_format: str):
    output_path = os.path.join(OUTPUT_DIR, filename)

    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)

    print(f"📁 Output saved at: {output_path}")


# ---------------- RUN ----------------
if __name__ == "__main__":
    print("\n🎉 Video → Hindi Speech-to-Text Pipeline Started\n")

    VIDEO_FILE = os.path.join(VIDEO_DIR, "sample.mp4")

    if not os.path.exists(VIDEO_FILE):
        raise FileNotFoundError(f"Video file not found: {VIDEO_FILE}")

    # Step 1: Extract audio from video
    audio_file = extract_audio_from_video(VIDEO_FILE)

    # Step 2: Transcribe audio
    processor = SpeechToTextProcessor(model_size="small")

    json_result = processor.process_audio(audio_file, "json")
    save_output(json_result, "sample_transcript.json", "json")

    srt_result = processor.process_audio(audio_file, "srt")
    save_output(srt_result, "sample_transcript.srt", "srt")

    txt_result = processor.process_audio(audio_file, "txt")
    save_output(txt_result, "sample_transcript.txt", "txt")

    # Cleanup temporary audio
    os.remove(audio_file)

    print("\n✅ Processing completed successfully\n")
