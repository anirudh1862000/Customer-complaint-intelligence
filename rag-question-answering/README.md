# Document RAG Question Answering Pipeline

A local-first Retrieval-Augmented Generation (RAG) portfolio project. It loads PDF, TXT, and Markdown documents, splits them into overlapping chunks, creates sentence-transformer embeddings, indexes them with FAISS, retrieves relevant passages, and answers questions with source references. If `OPENAI_API_KEY` is configured, it uses an OpenAI chat model for grounded answer generation. Without a key, it returns the retrieved passages in a transparent extractive fallback mode.

## Architecture

1. **Ingestion:** read PDF/TXT/Markdown files from a local folder.
2. **Chunking:** create overlapping text chunks while retaining source filenames and chunk IDs.
3. **Embeddings:** encode chunks with `sentence-transformers/all-MiniLM-L6-v2`.
4. **Vector search:** store normalized vectors in a FAISS inner-product index (cosine similarity equivalent for normalized vectors).
5. **Generation:** optionally send retrieved context to an OpenAI chat model with instructions to answer only from evidence and cite source chunks.
6. **API:** FastAPI exposes health, ingestion, and question-answering endpoints.

## Quick start (Python 3.10+)

```bash
cd rag-question-answering
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The embedding model downloads on first run. Add your own permitted documents under `data/documents/`; a fictional sample policy is included so the pipeline can be tried immediately.

Index documents and start the API:

```bash
python -c "from src.rag import LocalRAG; r=LocalRAG(); print('Indexed chunks:', r.ingest('data/documents'))"
uvicorn src.api:app --reload
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ask \\
  -H "Content-Type: application/json" \\
  -d '{"question":"How many annual leave days are provided?","top_k":3}'
```

Expected evidence: the answer should identify 20 annual leave days and cite `sample_company_policy.md`. Exact similarity scores depend on the embedding model version.

## Enable generated answers

Set `OPENAI_API_KEY` in your environment before starting the API. Optionally set `OPENAI_MODEL` (default `gpt-4o-mini`). The application sends only retrieved document excerpts to the configured API. Do not index confidential, personal, regulated, or otherwise restricted documents unless you are authorized to do so.

Without an API key, the system remains runnable and returns relevant retrieved passages rather than pretending to generate an LLM answer.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check whether a saved FAISS index is loaded |
| POST | `/ingest` | Build and persist an index from a document directory |
| POST | `/ask` | Retrieve passages and answer a question |

Interactive API documentation is available at `/docs` after starting the server.

## Run tests

```bash
pytest -q
```

## Evaluation and limitations

- Retrieval quality should be evaluated on a held-out set of questions with known supporting passages; this repository does not claim a measured retrieval or answer-quality score yet.
- The sample policy is fictional and only tests basic end-to-end retrieval; replace it with documents you are permitted to use for a meaningful portfolio evaluation.
- LLM output can still be wrong. Check cited source passages and add retrieval/answer faithfulness evaluation before production use.
- FAISS index files and downloaded embedding weights are generated locally and excluded from Git.
- This is a portfolio prototype, not a deployed production service.

## Suggested evaluation checklist

1. Create 30–50 questions answerable from your document set, with gold source passages.
2. Measure Recall@k and Mean Reciprocal Rank for retrieval.
3. Review answer correctness, citation correctness, and unsupported-claim rate.
4. Compare generated answers against extractive fallback.
5. Record model version, dataset version, and test split in a results report before publishing metrics.
