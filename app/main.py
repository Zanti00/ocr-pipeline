from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config import settings
from app.api.routes import ocr, duplicate, jobs, health, metrics
from app.db.mongodb import MongoDBClient

@asynccontextmanager
async def lifespan(app: FastAPI):
    MongoDBClient.connect()
    yield
    MongoDBClient.close()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="AI OCR Pipeline for SERMS",
    lifespan=lifespan
)

app.include_router(health.router)
app.include_router(ocr.router)
app.include_router(duplicate.router)
app.include_router(jobs.router)
app.include_router(metrics.router)

@app.get("/")
def root():
    return {"message": "OCR Pipeline API is running."}
