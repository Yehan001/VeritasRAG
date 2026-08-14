# VeritasRAG

VeritasRAG is a guarded RAG chatbot prototype. It filters unsafe user input before sending it to an LLM, answers using a local knowledge base, and checks the generated answer for faithfulness before returning it to the user.

## Main Components

- `input_filtering/veritasrag_input_filter.py` - reusable input filtering guardrail.
- `faithfulness_guardrail/rag_faithfulness_checker.py` - reusable output faithfulness checker.
- `chatbot/chatbot.py` - example chatbot integration using both guardrails.
- `chatbot/chat_app.py` - Streamlit demo UI.
- `knowledge_base/sample_kb.txt` - sample RAG knowledge base.
- `download_models.py` - optional pre-download script for local input-filtering models.
- `docs/DEPLOYMENT_GUIDE.md` - step-by-step setup and handover guide.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python download_models.py
streamlit run chatbot/chat_app.py
```

Add real `OPENROUTER_API_KEY` and `GROQ_API_KEY` values to `.env` before running the chatbot.

`download_models.py` is optional but recommended when local Hugging Face and SentenceTransformer filtering models are enabled, because it avoids a slow first request.
