import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Literal, Optional

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


class VideoSourceRequest(BaseModel):
    sourceType: Literal["googleDrive", "youtube", "directUrl"]
    sourceUrl: Optional[str] = None
    fileId: Optional[str] = None


class TranscriptRequest(VideoSourceRequest):
    language: str = "pt-BR"


class AnalyzeRequest(VideoSourceRequest):
    language: str = "pt-BR"
    includeTranscript: bool = True
    includeFrames: bool = False
    frameCount: int = Field(default=4, ge=1, le=12)


def validate_source(body):
    if not body.sourceUrl and not body.fileId:
        raise HTTPException(status_code=400, detail="Informe sourceUrl ou fileId")


def make_workdir() -> Path:
    path = Path(tempfile.gettempdir()) / f"video_job_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_dir(path: Path):
    shutil.rmtree(path, ignore_errors=True)


def run_cmd(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao executar comando: {' '.join(cmd)} | stderr={e.stderr[:500]}"
        )


def extract_drive_file_id(source_url: Optional[str], file_id: Optional[str]) -> str:
    if file_id:
        return file_id
    if not source_url:
        raise HTTPException(status_code=400, detail="Google Drive exige sourceUrl ou fileId")
    marker = "/file/d/"
    if marker in source_url:
        tail = source_url.split(marker, 1)[1]
        return tail.split("/", 1)[0]
    raise HTTPException(status_code=400, detail="Não consegui extrair o fileId do link do Google Drive")


def download_from_google_drive(source_url: Optional[str], file_id: Optional[str], dest: Path) -> None:
    drive_file_id = extract_drive_file_id(source_url, file_id)
    url = f"https://drive.google.com/uc?export=download&id={drive_file_id}"

    session = requests.Session()
    response = session.get(url, stream=True, timeout=120)

    if "text/html" in response.headers.get("Content-Type", ""):
        confirm_token = None
        for cookie_name, cookie_value in response.cookies.items():
            if cookie_name.startswith("download_warning"):
                confirm_token = cookie_value
                break
        if confirm_token:
            response = session.get(f"{url}&confirm={confirm_token}", stream=True, timeout=120)

    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def download_direct_url(source_url: str, dest: Path) -> None:
    with requests.get(source_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_youtube(source_url: str, dest: Path) -> None:
    cmd = ["yt-dlp", "-f", "mp4/best", "-o", str(dest), source_url]
    run_cmd(cmd)

    if not dest.exists():
        candidates = list(dest.parent.glob(dest.stem + ".*"))
        if not candidates:
            raise HTTPException(status_code=500, detail="Falha ao baixar vídeo do YouTube")
        candidates[0].rename(dest)


def resolve_video(body: VideoSourceRequest, workdir: Path) -> Path:
    video_path = workdir / "video.mp4"

    if body.sourceType == "googleDrive":
        download_from_google_drive(body.sourceUrl, body.fileId, video_path)
    elif body.sourceType == "directUrl":
        if not body.sourceUrl:
            raise HTTPException(status_code=400, detail="directUrl exige sourceUrl")
        download_direct_url(body.sourceUrl, video_path)
    elif body.sourceType == "youtube":
        if not body.sourceUrl:
            raise HTTPException(status_code=400, detail="youtube exige sourceUrl")
        download_youtube(body.sourceUrl, video_path)
    else:
        raise HTTPException(status_code=400, detail="sourceType inválido")

    if not video_path.exists():
        raise HTTPException(status_code=500, detail="Falha ao obter vídeo")
    return video_path


def ffprobe_metadata(video_path: Path) -> dict:
    output = run_cmd([
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ])
    data = json.loads(output or "{}")
    fmt = data.get("format", {})
    duration = fmt.get("duration")
    size = fmt.get("size")

    return {
        "title": video_path.name,
        "mimeType": "video/mp4",
        "durationSeconds": int(float(duration)) if duration else None,
        "sizeBytes": int(size) if size else None,
    }


def extract_audio(video_path: Path, workdir: Path) -> Path:
    audio_path = workdir / "audio.mp3"
    run_cmd([
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),
    ])
    if not audio_path.exists():
        raise HTTPException(status_code=500, detail="Falha ao extrair áudio")
    return audio_path


def transcribe_audio(audio_path: Path, language: str) -> dict:
    if not client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY não configurada")

    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language.split("-")[0] if language else "pt",
                response_format="verbose_json"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na transcrição: {str(e)}")

    segments = []
    raw_segments = getattr(transcript, "segments", None) or []
    for seg in raw_segments:
        start = getattr(seg, "start", 0)
        end = getattr(seg, "end", 0)
        text = getattr(seg, "text", "")
        segments.append({
            "startSeconds": start,
            "endSeconds": end,
            "text": text
        })

    full_text = getattr(transcript, "text", "") or ""

    return {
        "available": True,
        "language": language,
        "source": "speech_to_text",
        "fullText": full_text,
        "segments": segments
    }


def analyze_transcript(transcript_text: str, language: str) -> dict:
    if not client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY não configurada")

    prompt = f"""
Analise editorialmente a transcrição abaixo em {language}.
Responda em JSON com as chaves:
summary, strengths, risksBeforePublishing, improvements, suggestedTitle, suggestedDescription, suggestedHook.

Transcrição:
{transcript_text}
"""

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )
        text = response.output_text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na análise: {str(e)}")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {
            "summary": text,
            "strengths": [],
            "risksBeforePublishing": [],
            "improvements": [],
            "suggestedTitle": "",
            "suggestedDescription": "",
            "suggestedHook": ""
        }

    return {
        "summary": data.get("summary", ""),
        "strengths": data.get("strengths", []),
        "risksBeforePublishing": data.get("risksBeforePublishing", []),
        "improvements": data.get("improvements", []),
        "suggestedTitle": data.get("suggestedTitle", ""),
        "suggestedDescription": data.get("suggestedDescription", ""),
        "suggestedHook": data.get("suggestedHook", ""),
    }


@app.get("/health")
def health_check():
    return {"ok": True, "service": "video-backend", "version": "2.0.0"}


@app.post("/video/metadata")
def get_video_metadata(body: VideoSourceRequest):
    validate_source(body)
    workdir = make_workdir()
    try:
        video_path = resolve_video(body, workdir)
        metadata = ffprobe_metadata(video_path)
        return {
            "sourceType": body.sourceType,
            "sourceUrl": body.sourceUrl,
            "fileId": body.fileId,
            **metadata
        }
    finally:
        cleanup_dir(workdir)


@app.post("/video/transcript")
def get_video_transcript(body: TranscriptRequest):
    validate_source(body)
    workdir = make_workdir()
    try:
        video_path = resolve_video(body, workdir)
        audio_path = extract_audio(video_path, workdir)
        transcript = transcribe_audio(audio_path, body.language)
        return {"transcript": transcript}
    except Exception as e:
        print(f"ERROR /video/transcript: {repr(e)}", flush=True)
        raise
    finally:
        cleanup_dir(workdir)


@app.post("/video/analyze")
def analyze_video(body: AnalyzeRequest):
    validate_source(body)
    workdir = make_workdir()
    try:
        video_path = resolve_video(body, workdir)
        metadata = ffprobe_metadata(video_path)

        transcript = None
        if body.includeTranscript:
            audio_path = extract_audio(video_path, workdir)
            transcript = transcribe_audio(audio_path, body.language)

        analysis = analyze_transcript(
            transcript["fullText"] if transcript else "",
            body.language
        )

        return {
            "sourceType": body.sourceType,
            "sourceUrl": body.sourceUrl,
            "fileId": body.fileId,
            "title": metadata["title"],
            "mimeType": metadata["mimeType"],
            "durationSeconds": metadata["durationSeconds"],
            "transcript": transcript,
            "frames": [],
            **analysis
        }
    except Exception as e:
        print(f"ERROR /video/analyze: {repr(e)}", flush=True)
        raise
    finally:
        cleanup_dir(workdir)


@app.post("/video/uploadAnalyze")
async def upload_analyze(
    videoFile: UploadFile = File(...),
    language: str = Form("pt-BR"),
    includeTranscript: bool = Form(True),
    includeFrames: bool = Form(False),
    frameCount: int = Form(4),
    platform: str = Form(""),
    reviewGoal: str = Form(""),
    audience: str = Form(""),
    notes: str = Form("")
):
    workdir = make_workdir()
    try:
        video_path = workdir / videoFile.filename
        with open(video_path, "wb") as f:
            f.write(await videoFile.read())

        metadata = ffprobe_metadata(video_path)

        transcript = None
        if includeTranscript:
            audio_path = extract_audio(video_path, workdir)
            transcript = transcribe_audio(audio_path, language)

        analysis = analyze_transcript(
            transcript["fullText"] if transcript else "",
            language
        )

        return {
            "title": metadata["title"],
            "mimeType": metadata["mimeType"],
            "durationSeconds": metadata["durationSeconds"],
            "transcript": transcript,
            "frames": [],
            **analysis
        }
    except Exception as e:
        print(f"ERROR /video/uploadAnalyze: {repr(e)}", flush=True)
        raise
    finally:
        cleanup_dir(workdir)
