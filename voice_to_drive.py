#cd C:\Users\YourName\{foldername}
#python -m venv venv
#.\venv\Scripts\activate
#pip install openai sounddevice numpy python-dotenv

# voice_to_drive.py
import os, tempfile, time
from dotenv import load_dotenv
from openai import OpenAI
import sounddevice as sd
import numpy as np
import wave

DRIVE_PROMPT_FILE = r"G:/My Drive/StreamDiffusion/prompt.txt"  # path to Google Drive file
RECORD_DURATION = 10        
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 50   

load_dotenv()  # loads OPENAI_API_KEY from .env if present
print("OPENAI_API_KEY=", os.getenv("OPENAI_API_KEY"))
client = OpenAI()  # reads OPENAI_API_KEY automatically

def record_chunk(duration=RECORD_DURATION, rate=SAMPLE_RATE):
    print("Recording...", end="", flush=True)
    data = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype='int16')
    sd.wait()
    print(" done.")
    # simple energy check (avoid transcribing silence)
    energy = np.mean(np.abs(data))
    return data, energy

def write_wav_from_array(data, path, rate=SAMPLE_RATE):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 -> 2 bytes
        wf.setframerate(rate)
        wf.writeframes(data.tobytes())

def transcribe_with_whisper(wav_path):
    with open(wav_path, "rb") as af:
        resp = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",  # or "whisper-1"
            file=af
        )
    return getattr(resp, "text", "").strip()

def summarize_to_prompt(text):
    # Use OpenAI GPT to make a short art prompt
    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"Summarize this idea into a short, vivid AI art prompt: {text}"
    )
    return response.output_text.strip()

def save_prompt_to_drive(prompt_text):
    os.makedirs(os.path.dirname(DRIVE_PROMPT_FILE), exist_ok=True)
    with open(DRIVE_PROMPT_FILE, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    print("Saved prompt to", DRIVE_PROMPT_FILE)

def main_loop():
    print("Starting loop. Say 'prompt' followed by your prompt. Ctrl+C to stop.")
    while True:
        data, energy = record_chunk()
        if energy < SILENCE_THRESHOLD:
            print("(silence) energy", int(energy))
            time.sleep(0.2)
            continue

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        write_wav_from_array(data, tmp.name)

        try:
            transcript = transcribe_with_whisper(tmp.name)
            print("Transcript:", transcript)

            # Check if transcript starts with "prompt"
            words = transcript.strip().split()
            if len(words) > 1 and words[0][:6].lower() == "prompt":
                raw_text = " ".join(words[1:]).strip()
                if raw_text:
                    summarized = summarize_to_prompt(raw_text)
                    print("Summarized:", summarized)
                    if summarized:
                        save_prompt_to_drive(summarized)
            else:
                print("No 'prompt' keyword detected — skipping update.")

        except Exception as e:
            print("Error:", e)
        finally:
            try:
                os.remove(tmp.name)
            except:
                pass

        time.sleep(0.2)

if __name__ == "__main__":
    main_loop()
