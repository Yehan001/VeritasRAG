# RAGAS-Style Faithfulness Filtering System with Question Guardrail

## Overview

This project is a complete document Question Answering (QA) system that combines:

- Input Guardrail Filtering
- FAISS Vector Retrieval
- Groq Llama Models
- RAGAS-Style Faithfulness Verification
- Hallucination Detection
- Answer Rebuilding
- Streamlit User Interface

The system allows users to upload a document and ask questions from that document. Before the question reaches the Large Language Model (LLM), the system performs multiple security, safety, and validation checks.

After retrieval and answer generation, the system performs statement-level faithfulness verification to detect hallucinated content and remove unsupported statements.

---

# System Architecture

```text
User Question
      ↓
Question Guardrail
      ↓
Question Sanitization
      ↓
FAISS Retrieval
      ↓
LLM Answer Generation
      ↓
Atomic Statement Generation
      ↓
Statement Verification
      ↓
Hallucination Detection
      ↓
Answer Rebuilding
      ↓
Final Clean Answer
```

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application frontend |
| FAISS | Vector similarity search |
| LangChain | Text splitting and vector pipeline utilities |
| Sentence Transformers | Embedding generation |
| HuggingFace Embeddings | Semantic vector embeddings |
| Groq API | LLM inference |
| Llama Models | Answer generation and verification |
| RAGAS-style Verification | Statement-level faithfulness checking |
| dotenv | Environment variable management |
| pypdf | PDF text extraction |
| python-docx | DOCX extraction |
| Pandas | CSV export and tabular handling |

---

# Main Features

## 1. Question Guardrail System

The guardrail filters user questions before retrieval or LLM usage.

### Purpose

The purpose of the guardrail is to:

- Prevent malicious prompts
- Block prompt injection attacks
- Detect jailbreak attempts
- Remove noisy formatting
- Detect harmful content
- Detect suspicious inputs
- Improve retrieval quality
- Protect the RAG pipeline

---

# Guardrail Techniques Used

## A. Sanitization

The sanitizer cleans formatting issues without changing user meaning.

### Sanitization Operations

- Unicode normalization
- Remove invisible characters
- Collapse extra spaces
- Remove repeated punctuation
- Remove noisy repeated symbols
- Trim unnecessary punctuation

### Example

Input:

```text
what   is solar energy????%%%
```

Sanitized:

```text
what is solar energy?
```

---

## B. Validation Checks

### Checks Performed

- Empty input detection
- Too short input
- Too long input
- No real words
- Character spam detection
- Excessive punctuation detection
- Excessive capital letters
- Repeated phrase detection
- Gibberish detection
- Question structure validation

### Example Blocked Inputs

```text
aaaaaaaaaaaaa
```

```text
!!!!!!!!!
```

---

## C. Security Filtering

### Security Techniques Used

- Prompt injection detection
- Jailbreak detection
- Script injection detection
- SQL injection detection
- Template injection detection
- Path traversal detection
- Unicode bypass detection
- System prompt extraction detection

### Example Prompt Injection

```text
Ignore all previous instructions
```

### Example Jailbreak

```text
Act as DAN
```

### Example Script Injection

```html
<script>alert(1)</script>
```

### Example SQL Injection

```sql
DROP TABLE users;
```

---

## D. Safety Filtering

### Safety Checks

- Harmful intent detection
- Hacking-related prompts
- Weapon generation prompts
- Hate speech detection
- Profanity detection
- Self-harm related content
- PII detection

### Example Blocked Input

```text
How to make a bomb
```

---

# Important Design Decision

## No Automatic Spell Correction

The system intentionally DOES NOT auto-correct spelling.

### Reason

Automatic correction can corrupt:

- Technical terms
- Dataset names
- Product names
- Acronyms
- Variable names
- Medical terms
- Legal terms
- Programming syntax

### Example

```text
FAISS
PySpark
GraphFrames
LlamaIndex
```

could be incorrectly modified by spell correction systems.

Therefore the system only warns about spelling quality and preserves the original wording.

---

# Retrieval System

## FAISS Vector Retrieval

The system uses FAISS for semantic similarity search.

### Process

1. Document is split into chunks
2. Chunks are converted into embeddings
3. FAISS indexes the embeddings
4. User question embedding is generated
5. Top-k similar chunks are retrieved

---

# Embedding Technique

## Sentence Transformers

Model used:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Purpose

Converts text into semantic vectors for similarity search.

### Why Used

- Lightweight
- Fast
- Good semantic understanding
- Free local embedding generation

---

# Text Splitting Technique

## RecursiveCharacterTextSplitter

The system splits documents into overlapping chunks.

### Parameters

- Chunk size
- Chunk overlap

### Why Overlap is Used

Overlap preserves context between chunks.

Without overlap:

```text
Sentence may split across chunks.
```

With overlap:

```text
Context continuity is preserved.
```

---

# Large Language Models Used

## Answer Generation Model

Main model:

```text
llama-3.3-70b-versatile
```

### Purpose

Used for:

- Final answer generation

### Why Used

- Better reasoning
- Better answer quality
- Stronger contextual understanding

---

## Verification Model

Main model:

```text
llama-3.1-8b-instant
```

### Purpose

Used for:

- Statement generation
- Statement verification
- Answer rebuilding

### Why Used

- Lower cost
- Faster inference
- Good enough for verification tasks

---

# RAGAS-Style Faithfulness Verification

## Core Idea

The system checks whether every statement in the answer is supported by the retrieved context.

---

# Verification Pipeline

## Step 1 — Generate Raw Answer

The LLM generates an answer strictly from retrieved context.

---

## Step 2 — Atomic Statement Generation

The answer is broken into atomic statements.

### Example

Answer:

```text
Solar energy uses photovoltaic panels. Wind energy uses turbines.
```

Statements:

```text
1. Solar energy uses photovoltaic panels.
2. Wind energy uses turbines.
```

---

## Step 3 — Statement Verification

Each statement is verified against the retrieved context.

### Verification Rules

- verdict = 1 → supported
- verdict = 0 → unsupported
- verdict = 0 → contradictory
- verdict = 0 → missing from context

---

# Hallucination Detection

Hallucinated statements are statements not supported by the retrieved context.

### Example

Retrieved context:

```text
Solar energy uses photovoltaic panels.
```

Generated answer:

```text
Solar panels are 99% efficient.
```

Result:

```text
Unsupported → Hallucination
```

---

# Faithfulness Score

The system calculates:

```text
Faithfulness Score = Grounded Statements / Total Statements
```

### Example

```text
Grounded = 4
Hallucinated = 1

Score = 4 / 5 = 0.80
```

---

# Verdict System

| Score | Verdict |
|---|---|
| >= 0.85 | PASS |
| 0.50 – 0.84 | PARTIAL |
| < 0.50 | FAIL |

---

# Answer Rebuilding

If hallucinated statements exist:

- Unsupported statements are removed
- Remaining grounded statements are rebuilt into a clean answer

### Purpose

This ensures the final answer only contains context-supported information.

---

# Cost Optimization Techniques

## 1. Model Separation

Expensive 70B model only used for:

- Answer generation

Cheaper 8B model used for:

- Statement generation
- Verification
- Rebuilding

---

## 2. Batch Verification

Statements are verified together in one LLM call when possible.

This reduces:

- API cost
- latency
- repeated requests

---

## 3. Rebuild Skipping

If no hallucinated statements are found:

- rebuild stage is skipped
- raw answer becomes final answer

This saves additional LLM calls.

---

## 4. In-Memory Cache

The system supports caching repeated questions.

### Purpose

Avoid repeated:

- retrieval
- LLM calls
- verification costs

---

# Files in the Project

| File | Purpose |
|---|---|
| app_integrated_guardrail_option_b_FINAL.py | Main Streamlit application |
| input_guardrail_question_only_FIXED.py | Question guardrail system |
| final_option_b_faithfulness_filter_FINAL.py | Option B faithfulness backend |
| requirements_integrated_FINAL.txt | Required dependencies |
| .env | Stores Groq API key |
| README.md | Project documentation |

---

# Installation Guide

## Step 1 — Create Virtual Environment

Windows:

```bash
python -m venv venv
```

---

## Step 2 — Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements_integrated_FINAL.txt
```

---

# Groq API Setup

Create:

```text
.env
```

Inside `.env`:

```env
GROQ_API_KEY=your_api_key_here
```

---

# Running the Project

Run:

```bash
streamlit run app_integrated_guardrail_option_b_FINAL.py
```

Then open:

```text
http://localhost:8501
```

---

# How the System Works

## User Flow

1. User uploads document
2. User asks question
3. Guardrail checks question
4. Question is sanitized
5. FAISS retrieves relevant chunks
6. LLM generates answer
7. Statements are extracted
8. Statements are verified
9. Hallucinated statements are removed
10. Final answer is rebuilt
11. User receives grounded answer

---

# Advantages of This System

## Security Advantages

- Prompt injection protection
- Jailbreak protection
- Harmful content filtering
- Script injection protection

---

## RAG Advantages

- Semantic retrieval
- Context-based answering
- Reduced hallucination
- Statement-level verification
- Grounded answer generation

---

## Cost Advantages

- 8B verifier model
- Batch verification
- Rebuild skipping
- Cache support

---

# Limitations

- Depends on retrieval quality
- Limited by document extraction quality
- OCR/scanned PDFs may fail
- Verification still depends on LLM reasoning
- Large documents increase processing time

---

# Future Improvements

Potential future improvements:

- OCR support for scanned PDFs
- Multi-document retrieval
- Hybrid BM25 + FAISS retrieval
- Pre-generation hallucination prevention
- Retrieval confidence scoring
- Context sufficiency checks
- Advanced semantic caching
- Multi-language support
- Database-backed persistent caching
- User authentication

---

# Recommended Python Version

Recommended:

```text
Python 3.10 or Python 3.11
```

Avoid:

```text
Python 3.13+
```

because some ML libraries may not yet fully support newer versions.

---

# Conclusion

This project demonstrates a production-style Retrieval-Augmented Generation (RAG) system with:

- Secure question filtering
- Semantic retrieval
- LLM answer generation
- RAGAS-style faithfulness verification
- Hallucination removal
- Cost optimization
- Streamlit deployment

The system focuses on generating grounded, document-supported answers while reducing hallucinations and protecting the pipeline from unsafe or malicious inputs.

