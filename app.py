from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


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
    frameCount: int = 4


def validate_source(body):
    if not body.sourceUrl and not body.fileId:
        raise HTTPException(status_code=400, detail="Informe sourceUrl ou fileId")


@app.get("/health")
def health_check():
    return {"ok": True, "service": "video-backend", "version": "1.0.0"}


@app.post("/video/metadata")
def get_video_metadata(body: VideoSourceRequest):
    validate_source(body)
    return {
        "sourceType": body.sourceType,
        "sourceUrl": body.sourceUrl,
        "fileId": body.fileId,
        "title": "Vídeo de teste",
        "mimeType": "video/mp4",
        "durationSeconds": 120,
        "sizeBytes": 12345678
    }


@app.post("/video/transcript")
def get_video_transcript(body: TranscriptRequest):
    validate_source(body)
    return {
        "transcript": {
            "available": True,
            "language": body.language,
            "source": "mock",
            "fullText": "Transcrição de teste do vídeo.",
            "segments": [
                {
                    "startSeconds": 0,
                    "endSeconds": 5,
                    "text": "Transcrição de teste do vídeo."
                }
            ]
        }
    }


@app.post("/video/analyze")
def analyze_video(body: AnalyzeRequest):
    validate_source(body)
    return {
        "sourceType": body.sourceType,
        "sourceUrl": body.sourceUrl,
        "fileId": body.fileId,
        "title": "Vídeo de teste",
        "mimeType": "video/mp4",
        "durationSeconds": 120,
        "transcript": {
            "available": True,
            "language": body.language,
            "source": "mock",
            "fullText": "Transcrição de teste do vídeo.",
            "segments": [
                {
                    "startSeconds": 0,
                    "endSeconds": 5,
                    "text": "Transcrição de teste do vídeo."
                }
            ]
        },
        "frames": [],
        "summary": "Análise de teste concluída.",
        "strengths": [
            "Estrutura básica funcionando"
        ],
        "risksBeforePublishing": [
            "Backend ainda está em modo de teste"
        ],
        "improvements": [
            "Trocar respostas mock por análise real"
        ],
        "suggestedTitle": "Título sugerido",
        "suggestedDescription": "Descrição sugerida",
        "suggestedHook": "Gancho sugerido"
    }
