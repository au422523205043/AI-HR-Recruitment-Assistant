# AI HR Recruitment Assistant

A runnable Agent + Tools + RAG project for:
1. PDF resume ingestion
2. FAISS semantic retrieval
3. Job-description matching
4. Candidate scoring/ranking
5. Evidence-backed strengths and gaps
6. Tailored interview-question generation

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add OPENAI_API_KEY
streamlit run app/main.py
```

Human review is required; don't use protected characteristics or proxies in employment decisions.
