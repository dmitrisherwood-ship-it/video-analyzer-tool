from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess, tempfile, os, openai, json
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

class ScriptResult(BaseModel):
    transcript: str
    script: str
    segments: list

def extract_audio(video_path, audio_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,   # 👈 yahan -y add hua
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ], check=True)


@app.post("/analyze", response_model=ScriptResult)
async def analyze_video(file: UploadFile = File(...)):
    video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video.write(await file.read())
    video.flush()
    video.close()

    audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.close()

    extract_audio(video.name, audio.name)

    transcript_data = openai.Audio.transcribe("whisper-1", open(audio.name, "rb"))
    transcript = transcript_data["text"]

    prompt = f"""
You are a script writing assistant. Below is a transcript:

{transcript}

Your tasks:
1. Rewrite it into a full, clean script.
2. Split it into segments with timestamps and headings.
3. Return JSON with 'script', 'segments': [{{"start": "...", "end": "...", "text": "...", "title": "..."}}]
    """

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )

    response_text = completion["choices"][0]["message"]["content"]
    parsed = json.loads(response_text)

    os.unlink(video.name)
    os.unlink(audio.name)

    return ScriptResult(**parsed)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
