# GitLab Handbook RAG Chatbot

A Retrieval-Augmented Generation chatbot built with **Streamlit**, **ChromaDB**, and **Gemini**.

## Project Structure

```
.
├── app.py                  # Streamlit UI
├── rag_pipeline.py         # Ingestion, retrieval, and generation logic
├── evaluate.py             # Golden dataset evaluation script
├── golden_dataset.json     # 10 Q&A pairs (1 negative example)
├── requirements.txt
├── Gitlab/                 # ← place your markdown files here
└── chroma_db/              # ← auto-created after first ingest
```

## Setup & Run

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Place your GitLab handbook .md files in the Gitlab/ folder

# 3. Run the app
streamlit run app.py
```

## First-time use

1. Enter your **Gemini API key** in the sidebar.
2. Click **"🔨 Build / Refresh DB"** — this parses all markdown files and builds ChromaDB (~5 min).
3. Start chatting! The DB is persisted to `chroma_db/` so you only need to build it once.

## Evaluation

```bash
export GEMINI_API_KEY=your_key
python evaluate.py
# Results saved to test_results.json
```

## Key Design Decisions

| Choice | Rationale |
|--------|-----------|
| ChromaDB `all-MiniLM-L6-v2` | Fast local embeddings, no API cost |
| Cosine similarity threshold = 0.35 | Filters weak matches; prevents hallucination |
| Chunk size 800 chars / 150 overlap | Balances context richness vs retrieval precision |
| Gemini 1.5 Flash | Fast, cheap, generous context window |
| Streamlit | Simple to run locally and deploy |

## Deployment (Hugging Face Spaces)

1. Push all files including the `chroma_db/` folder to a HF Space (Streamlit SDK).
2. Set `GEMINI_API_KEY` as a Space secret.
3. The sidebar key field can then auto-read from `os.environ`.
