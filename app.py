from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"ok": True, "service": "video-backend", "version": "1.0.0"}
