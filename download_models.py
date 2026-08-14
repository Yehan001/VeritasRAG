"""Pre-download local models used by the VeritasRAG input filter.

Run this once after installing requirements to reduce first-request latency.
The chatbot still works without this script, but enabled local model stages will
download models lazily the first time they are used.
"""

from input_filtering.veritasrag_input_filter import (
    GuardrailSettings,
    download_configured_models,
)


def main() -> None:
    settings = GuardrailSettings(
        enable_hf_models=True,
        enable_semantic_matcher=True,
        use_presidio=False,
        enable_audit_log=False,
    )
    download_configured_models(settings)
    print("Configured input-filtering models downloaded successfully.")


if __name__ == "__main__":
    main()
