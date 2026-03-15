import base64
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Literal, Optional

import requests
from fastapi import FastAPI, HTTPException
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
    includeFrames: bool = True
    frameCount: int = Field(default=6, ge=1, le=12)

def validate_source(body: VideoSourceRequest):
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
            detail=f"Erro ao executar comando: {' '.join(cmd)} | stderr={e.stderr[:800]}"
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


def validate_downloaded_video(dest: Path):
    if not dest.exists():
        raise HTTPException(status_code=500, detail="Arquivo de vídeo não foi baixado")

    size = dest.stat().st_size
    if size == 0:
        raise HTTPException(status_code=500, detail="Arquivo baixado está vazio")

    header = dest.read_bytes()[:512]

    # Se o download veio como HTML, o Drive não entregou o vídeo real
    if b"<html" in header.lower() or b"<!doctype html" in header.lower():
        raise HTTPException(
            status_code=500,
            detail="Google Drive devolveu HTML em vez do arquivo de vídeo. Verifique compartilhamento/permissão."
        )


def download_from_google_drive(source_url: Optional[str], file_id: Optional[str], dest: Path) -> None:
    drive_file_id = extract_drive_file_id(source_url, file_id)
    url = f"https://drive.google.com/uc?export=download&id={drive_file_id}"

    session = requests.Session()
    response = session.get(url, stream=True, timeout=120)

    content_type = response.headers.get("Content-Type", "")

    if "text/html" in content_type:
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

    validate_downloaded_video(dest)


def download_direct_url(source_url: str, dest: Path) -> None:
    with requests.get(source_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    validate_downloaded_video(dest)


def download_youtube(source_url: str, dest: Path) -> None:
    cmd = ["yt-dlp", "-f", "mp4/best", "-o", str(dest), source_url]
    run_cmd(cmd)

    if not dest.exists():
        candidates = list(dest.parent.glob(dest.stem + ".*"))
        if not candidates:
            raise HTTPException(status_code=500, detail="Falha ao baixar vídeo do YouTube")
        candidates[0].rename(dest)

    validate_downloaded_video(dest)


def resolve_video(body: VideoSourceRequest, workdir: Path) -> Path:
    video_path = workdir / "video.mp4"

    if body.sourceType == "googleDrive":
        download_from_google_drive(body.sourceUrl, body.fileId, video_path)

    elif body.sourceType == "directUrl":
        if not body.sourceUrl:
            raise HTTPException(status_code=400, detail="directUrl exige sourceUrl")

        if body.sourceUrl.startswith("/"):
            local_path = Path(body.sourceUrl)
            if not local_path.exists():
                raise HTTPException(status_code=400, detail=f"Arquivo local não encontrado: {body.sourceUrl}")
            shutil.copy(local_path, video_path)
            validate_downloaded_video(video_path)
        else:
            download_direct_url(body.sourceUrl, video_path)

    elif body.sourceType == "youtube":
        if not body.sourceUrl:
            raise HTTPException(status_code=400, detail="youtube exige sourceUrl")
        download_youtube(body.sourceUrl, video_path)

    else:
        raise HTTPException(status_code=400, detail="sourceType inválido")

    return video_path


def ffprobe_metadata(video_path: Path) -> dict:
    try:
        output = run_cmd([
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ])
    except HTTPException:
        raise HTTPException(
            status_code=500,
            detail="ffprobe não conseguiu ler o MP4. O arquivo pode estar corrompido ou o download do Drive não entregou um vídeo válido."
        )

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

    try:
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
    except HTTPException:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg não conseguiu extrair áudio do vídeo"
        )

    if not audio_path.exists():
        raise HTTPException(status_code=500, detail="Falha ao extrair áudio")

    return audio_path


def extract_frame_files(video_path: Path, workdir: Path, frame_count: int = 6) -> list[dict]:
    output_dir = workdir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = ffprobe_metadata(video_path)
    duration = metadata.get("durationSeconds") or 0
    if duration <= 0:
        return []

    # prioriza abertura, meio e fechamento
    points = set()

    # primeiros segundos
    points.add(1)
    if duration >= 2:
        points.add(2)

    # meio
    points.add(max(1, duration // 2))

    # final
    if duration >= 3:
        points.add(max(1, duration - 2))
    points.add(max(1, duration - 1))

    # completa com intervalos regulares
    if frame_count > len(points):
        step = max(1, duration // (frame_count + 1))
        for i in range(frame_count):
            points.add(max(1, step * (i + 1)))

    ordered_points = sorted(p for p in points if p <= max(1, duration))[:frame_count]

    frames = []

    for idx, sec in enumerate(ordered_points, start=1):
        frame_path = output_dir / f"frame_{idx}.jpg"

        try:
            run_cmd([
                "ffmpeg",
                "-y",
                "-ss", str(sec),
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(frame_path),
            ])
        except HTTPException:
            continue

        if frame_path.exists():
            frames.append({
                "timestampSeconds": sec,
                "path": frame_path
            })

    return frames


def image_to_data_url(image_path: Path) -> str:
    mime = "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def describe_frames(frame_files: list[dict], language: str) -> list[dict]:
    if not client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY não configurada")

    described = []

    for item in frame_files:
        frame_path: Path = item["path"]
        timestamp = item["timestampSeconds"]

        try:
            image_data_url = image_to_data_url(frame_path)

            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    f"Descreva este frame de vídeo culinário em {language}. "
                                    "Seja extremamente conservadora. "
                                    "Descreva apenas o que estiver claramente visível. "
                                    "Não adivinhe ingrediente exato. "
                                    "Não afirme tipo exato de massa se não estiver totalmente evidente. "
                                    "Foque em estrutura editorial e força visual. "
                                    "Responda em JSON com as chaves: "
                                    "description, visualStrength, appetizing, hasDish, hasHuman, hasTextOverlay, likelyMoment, confidence."
                                )
                            },
                            {
                                "type": "input_image",
                                "image_url": image_data_url
                            }
                        ]
                    }
                ]
            )

            text = response.output_text.strip()

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {
                    "description": text,
                    "visualStrength": "unknown",
                    "appetizing": "unknown",
                    "hasDish": None,
                    "hasHuman": None,
                    "hasTextOverlay": None,
                    "likelyMoment": "unknown",
                    "confidence": "low"
                }

            described.append({
                "timestampSeconds": timestamp,
                "description": data.get("description", ""),
                "visualStrength": data.get("visualStrength", "unknown"),
                "appetizing": data.get("appetizing", "unknown"),
                "hasDish": data.get("hasDish"),
                "hasHuman": data.get("hasHuman"),
                "hasTextOverlay": data.get("hasTextOverlay"),
                "likelyMoment": data.get("likelyMoment", "unknown"),
                "confidence": data.get("confidence", "low")
            })

        except Exception as e:
            described.append({
                "timestampSeconds": timestamp,
                "description": f"Falha ao descrever frame: {str(e)}",
                "visualStrength": "unknown",
                "appetizing": "unknown",
                "hasDish": None,
                "hasHuman": None,
                "hasTextOverlay": None,
                "likelyMoment": "unknown",
                "confidence": "low"
            })

    return described


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


def summarize_frame_confidence(frames: list[dict]) -> dict:
    if not frames:
        return {
            "usable": False,
            "highConfidenceCount": 0,
            "mediumConfidenceCount": 0,
            "lowConfidenceCount": 0
        }

    high_count = sum(1 for f in frames if f.get("confidence") == "high")
    medium_count = sum(1 for f in frames if f.get("confidence") == "medium")
    low_count = sum(1 for f in frames if f.get("confidence") == "low")

    usable = (high_count + medium_count) >= max(1, len(frames) // 2)

    return {
        "usable": usable,
        "highConfidenceCount": high_count,
        "mediumConfidenceCount": medium_count,
        "lowConfidenceCount": low_count
    }


def analyze_video_editorially(transcript_text: str, frames: list[dict], language: str) -> dict:
    if not client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY não configurada")

    frames_text = json.dumps(frames, ensure_ascii=False, indent=2)

    prompt = f"""
Faça uma análise editorial de um vídeo curto de gastronomia em {language}.

Use os frames apenas para avaliar:
- abertura visual
- presença de prato
- presença humana
- texto na tela
- força visual geral
- apetite visual
- estrutura de começo, meio e fim

Não use frames para afirmar:
- ingrediente exato
- tipo exato de massa
- composição culinária detalhada

Responda em JSON com as chaves:
summary, strengths, risksBeforePublishing, improvements, suggestedTitle, suggestedDescription, suggestedHook.

Transcrição:
{transcript_text}

Frames descritos:
{frames_text}
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
    return {"ok": True, "service": "video-backend", "version": "4.2.0"}


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
    except Exception as e:
        print(f"ERROR /video/metadata: {repr(e)}", flush=True)
        raise
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

        frames = []
        if body.includeFrames:
            frame_files = extract_frame_files(video_path, workdir, body.frameCount)
            frames = describe_frames(frame_files, body.language)

        if body.includeFrames and len(frames) == 0:
            raise HTTPException(
                status_code=422,
                detail="Não foi possível extrair frames suficientes para avaliação visual"
            )

        confidence_summary = summarize_frame_confidence(frames)

        if body.includeFrames and not confidence_summary["usable"]:
            raise HTTPException(
                status_code=422,
                detail="Frames extraídos com baixa confiabilidade para avaliação visual completa"
            )

        analysis = analyze_video_editorially(
            transcript["fullText"] if transcript else "",
            frames,
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
            "frames": frames,
            "frameConfidence": confidence_summary,
            **analysis
        }

    except Exception as e:
        print(f"ERROR /video/analyze: {repr(e)}", flush=True)
        raise
    finally:
        cleanup_dir(workdir)
