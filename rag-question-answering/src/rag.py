from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

SUPPORTED = {'.txt', '.md', '.pdf'}

@dataclass
class Chunk:
    chunk_id: int
    source: str
    text: str

class LocalRAG:
    """Local vector retrieval with optional OpenAI answer generation."""
    def __init__(self, index_dir: str = 'storage', model_name: str | None = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.encoder = SentenceTransformer(self.model_name)
        self.index = None
        self.chunks: list[Chunk] = []

    @staticmethod
    def read_file(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == '.pdf':
            reader = PdfReader(str(path))
            return '\n'.join(page.extract_text() or '' for page in reader.pages)
        return path.read_text(encoding='utf-8', errors='ignore')

    @staticmethod
    def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return []
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError('Require chunk_size > overlap >= 0')
        words = text.split(' ')
        chunks = []
        step = chunk_size - overlap
        # Chunk by characters while preserving word boundaries.
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(' ', start, end)
                if boundary > start:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def ingest(self, directory: str, chunk_size: int = 900, overlap: int = 150) -> int:
        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(f'Document directory does not exist: {root}')
        new_chunks: list[Chunk] = []
        for path in sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED):
            text = self.read_file(path)
            for piece in self.split_text(text, chunk_size, overlap):
                new_chunks.append(Chunk(len(new_chunks), str(path.relative_to(root)), piece))
        if not new_chunks:
            raise ValueError(f'No readable PDF, TXT, or Markdown content found in {root}')
        vectors = self.encoder.encode([c.text for c in new_chunks], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        vectors = np.asarray(vectors, dtype='float32')
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index, self.chunks = index, new_chunks
        self.save()
        return len(new_chunks)

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError('Nothing to save; ingest documents first')
        faiss.write_index(self.index, str(self.index_dir / 'documents.faiss'))
        payload = {'model_name': self.model_name, 'chunks': [asdict(c) for c in self.chunks]}
        (self.index_dir / 'chunks.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def load(self) -> bool:
        index_path, metadata_path = self.index_dir / 'documents.faiss', self.index_dir / 'chunks.json'
        if not index_path.exists() or not metadata_path.exists():
            return False
        payload = json.loads(metadata_path.read_text(encoding='utf-8'))
        if payload.get('model_name') != self.model_name:
            raise ValueError('Saved index uses a different embedding model; re-ingest documents')
        self.index = faiss.read_index(str(index_path))
        self.chunks = [Chunk(**item) for item in payload['chunks']]
        return True

    def retrieve(self, question: str, top_k: int = 4) -> list[dict]:
        if self.index is None:
            raise RuntimeError('Index is empty. Call ingest() or load() first.')
        if not question.strip():
            raise ValueError('Question cannot be empty')
        query = self.encoder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(query, dtype='float32'), min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append({'source': chunk.source, 'chunk_id': chunk.chunk_id, 'score': float(score), 'text': chunk.text})
        return results

    @staticmethod
    def extractive_answer(question: str, contexts: list[dict]) -> str:
        if not contexts:
            return 'I could not find relevant information in the indexed documents.'
        # The fallback deliberately returns retrieved evidence rather than inventing an answer.
        excerpts = '\n'.join(f"[{i}] {item['text']}" for i, item in enumerate(contexts, 1))
        return 'I do not have an LLM key configured, so here are the most relevant passages. Use the source references below to verify the answer.\n\n' + excerpts

    def answer(self, question: str, top_k: int = 4) -> dict:
        contexts = self.retrieve(question, top_k)
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and contexts:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            evidence = '\n\n'.join(f"[Source: {c['source']} | chunk {c['chunk_id']}] {c['text']}" for c in contexts)
            response = client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                temperature=0,
                messages=[
                    {'role': 'system', 'content': 'Answer only from the supplied context. If the context does not contain the answer, say so. Cite sources inline using [source: filename, chunk: id]. Never invent facts.'},
                    {'role': 'user', 'content': f'Question: {question}\n\nRetrieved context:\n{evidence}'}
                ]
            )
            answer = response.choices[0].message.content or 'No answer was generated.'
            mode = 'retrieval_augmented_generation'
        else:
            answer = self.extractive_answer(question, contexts)
            mode = 'extractive_retrieval_fallback'
        return {'question': question, 'answer': answer, 'mode': mode,
                'sources': [{'source': c['source'], 'chunk_id': c['chunk_id'], 'score': c['score']} for c in contexts]}
