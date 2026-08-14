# Project Overview

## What This Project Is

VeritasRAG is a RAG-style chatbot prototype with two safety layers:

1. Input filtering guardrail before the LLM call.
2. Faithfulness guardrail after the LLM generates an answer.

The chatbot uses `knowledge_base/sample_kb.txt` as its source context. For non-chit-chat questions, the chatbot sends that context to the LLM and instructs it to answer only from the provided information.

## Reusable Files

The main handover files are:

- `input_filtering/veritasrag_input_filter.py`
- `faithfulness_guardrail/rag_faithfulness_checker.py`
- `download_models.py`

The supervisor can replace the demo chatbot with another application by calling these two modules in the same order:

1. Run user input through `InputGuardrail.check()`.
2. Send the sanitized input to the RAG/LLM application if it is not blocked.
3. Run the generated answer through `check_faithfulness()`.
4. Return only the grounded final answer.

## Current Demo Flow

```text
User message
  -> input filtering guardrail
  -> sanitized message
  -> chatbot + knowledge base context
  -> LLM answer
  -> faithfulness checker
  -> final safe answer
```

## Notes

This is a prototype RAG application. The demo uses a local text file as the knowledge base. A production application can replace this with a database, document store, vector database, or enterprise retrieval system.

`download_models.py` is included because the original project experienced high first-run latency when local models were downloaded lazily. Running it during setup caches the configured input-filtering models before the application starts.
