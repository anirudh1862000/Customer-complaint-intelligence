from __future__ import annotations

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.rag import LocalRAG

app = FastAPI(title='Document RAG Question Answering API', version='1.0.0',
              description='Local document ingestion, FAISS retrieval, and cited question answering.')
rag = LocalRAG(index_dir=os.getenv('RAG_INDEX_DIR', 'storage'))

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)

class IngestRequest(BaseModel):
    directory: str = Field(default_factory=lambda: os.getenv('RAG_DATA_DIR', 'data/documents'))

@app.get('/health')
def health():
    loaded = rag.load() if rag.index is None else True
    return {'status': 'ok', 'index_loaded': loaded, 'indexed_chunks': len(rag.chunks)}

@app.post('/ingest')
def ingest(request: IngestRequest):
    try:
        count = rag.ingest(request.directory)
        return {'status': 'indexed', 'chunks': count, 'directory': str(Path(request.directory))}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post('/ask')
def ask(request: AskRequest):
    try:
        if rag.index is None and not rag.load():
            raise HTTPException(status_code=409, detail='No index found. Call POST /ingest first.')
        return rag.answer(request.question, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
