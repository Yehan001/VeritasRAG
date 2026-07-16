# VeritasRAG Input Filtering 4.0 - Clean Safety-Only Guardrail

This version removes knowledge-base relevance and implements only input filtering, as requested.

## Final pipeline

The system uses **rules + three model layers** with short-circuit execution:

1. **Deterministic rules**
   - script/HTML injection
   - command/path/SQL/template injection
   - prompt-injection patterns
   - cyber abuse
   - violence/self-harm/dangerous content
   - Base64 hidden payloads
   - obvious fraud, extremism, child-safety, hate patterns

2. **Model 1: Prompt-injection classifier**
   - default: `protectai/deberta-v3-small-prompt-injection-v2`
   - detects prompt injection and jailbreak attempts.

3. **Model 2: Moderation classifier**
   - default: `oxyapi/albert-moderation-001`
   - detects harmful content such as violence, self-harm, hate, sexual unsafe content, and dangerous content.

4. **Model 3: SentenceTransformer semantic risk matcher**
   - default: `sentence-transformers/all-MiniLM-L6-v2`
   - compares the user input with curated harmful-risk prototypes using semantic similarity.
   - if SentenceTransformers is not installed, it uses a TF-IDF fallback.

5. **PII masking**
   - Microsoft Presidio if installed.
   - Regex fallback for email, phone, card-like numbers, IP, Sri Lankan NIC, and IBAN.

## Short-circuit behaviour

If one stage blocks, the system immediately returns `BLOCKED` and skips later models.

Example:

```text
<script>alert(1)</script>
```

Blocked by deterministic rules, so the Hugging Face models and SentenceTransformer matcher are skipped.

## Outputs

- `PASSED`
- `PASSED_WITH_WARNING`
- `BLOCKED`

## Run light version

```powershell
pip install -r requirements_light.txt
streamlit run app.py
```

## Run full model version

```powershell
pip install -r requirements_full.txt
python download_models.py
streamlit run app.py
```

## Run tests

```powershell
python tests/test_guardrail.py
```

## Run evaluation

```powershell
python evaluate_guardrail.py
```

## Important note about accuracy

Accuracy depends on the test set, selected thresholds, and whether Hugging Face/SentenceTransformer models are enabled. Use `evaluate_guardrail.py` to report accuracy on the included curated test cases, and keep adding real failed cases to improve coverage.
