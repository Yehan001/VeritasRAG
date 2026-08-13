import csv
from pathlib import Path
from collections import Counter

from guardrail import GuardrailSettings, InputGuardrail

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "eval_cases.csv"

def main():
    g = InputGuardrail(GuardrailSettings(enable_hf_models=False, enable_sentence_transformer=True, use_presidio=False, enable_audit_log=False))
    rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
    total = 0
    correct_decision = 0
    correct_label = 0
    conf = Counter()
    print("Evaluation results")
    print("=" * 80)
    for row in rows:
        r = g.check(row["text"])
        total += 1
        ok_dec = r.decision == row["expected_decision"]
        ok_lab = (row["expected_label"] == "safe" and r.safety_label == "safe") or (row["expected_label"] != "safe" and r.safety_label == row["expected_label"])
        correct_decision += int(ok_dec)
        correct_label += int(ok_lab)
        conf[(row["expected_decision"], r.decision)] += 1
        status = "OK" if ok_dec else "FAIL"
        print(f"{status:4} expected={row['expected_decision']:20} got={r.decision:20} label={r.safety_label:30} text={row['text']}")
    print("=" * 80)
    print(f"Decision accuracy: {correct_decision}/{total} = {correct_decision/total:.3f}")
    print(f"Exact label accuracy: {correct_label}/{total} = {correct_label/total:.3f}")
    print("Decision confusion counts:")
    for k, v in sorted(conf.items()):
        print(k, v)

if __name__ == "__main__":
    main()
