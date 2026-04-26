import whisper
import subprocess
import os

model = whisper.load_model("base")

def transcribe(audio_path):
    wav_path = audio_path.replace(".webm", ".wav")

    # Convert WebM → WAV (required for Whisper)
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    result = model.transcribe(wav_path, fp16=False, word_timestamps=True)

    words = []
    duration = 0

    for seg in result["segments"]:
        duration = max(duration, seg["end"])

        if "words" in seg:
            for w in seg["words"]:
                words.append({
                    "word": w["word"].strip().lower(),
                    "start": float(w["start"]),
                    "end": float(w["end"])
                })

    try:
        os.remove(wav_path)
    except:
        pass

    return {
        "text": result["text"].strip(),
        "words": words,
        "duration": duration
    }
