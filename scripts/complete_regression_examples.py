"""Collect actual regression losses without training or altering frozen artifacts."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "src"))


def compare(records, base_preds, ft_preds, score):
    if not (len(records) == len(base_preds) == len(ft_preds)):
        raise ValueError("Prediction counts differ")
    rows = []
    for i, (r, b, c) in enumerate(zip(records, base_preds, ft_preds)):
        bs, cs = score(b, r["keywords"]), score(c, r["keywords"])
        rows.append({"group": "regression", "i": i, "instruction": r["instruction"],
                     "keywords": r["keywords"], "base_pred": b, "ft_pred": c,
                     "base_score": bs, "ft_score": cs, "delta": cs-bs,
                     "outcome": "FT thua" if cs < bs else "FT thắng" if cs > bs else "Hòa"})
    return rows


def main():
    from labkit import generate, evaluate as ev
    from labkit.config import get_tier
    from peft import PeftModel

    frozen = json.loads((ROOT / "results/baselines_frozen.json").read_text(encoding="utf-8"))
    verdict = json.loads((ROOT / "results/verdict.json").read_text(encoding="utf-8"))
    tier = get_tier(frozen["tier"])
    if tier.model_id != frozen["model"]:
        raise RuntimeError("Restore the original model configuration")
    source = ROOT / "data/eval_regression.jsonl"
    checksums = json.loads((ROOT / "data/checksums.json").read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest()[:16] != checksums[source.name]:
        raise RuntimeError("Regression corpus differs from the original")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != frozen["n_regression"] or frozen.get("smoke_mode"):
        raise RuntimeError("Full original regression set required")
    prompts = [r["instruction"] for r in records]
    model, tok = generate.load_base(tier)
    model.eval()
    base_preds, _ = generate.generate_batch(model, tok, prompts, system=None,
        max_new_tokens=96, label="supplement/base-regression")
    del model, tok
    generate.free_memory()
    model, tok = generate.load_base(tier)
    model = PeftModel.from_pretrained(model, str(ROOT / "adapters/correct"))
    model.eval()
    ft_preds, _ = generate.generate_batch(model, tok, prompts, system=None,
        max_new_tokens=96, label="supplement/ft-regression")
    del model, tok
    generate.free_memory()
    rows = compare(records, base_preds, ft_preds, ev.keyword_recall)
    b = sum(r["base_score"] for r in rows)/len(rows)
    c = sum(r["ft_score"] for r in rows)/len(rows)
    matches = abs(b-frozen["baseline_b"]["regression"]) < .00005 and abs(c-verdict["comparison"][2]["regression"]) < .00005
    losses = [r for r in rows if r["delta"] < 0]
    result = {"provenance": "Retrospective generation on the unchanged original regression set; system=None and max_new_tokens=96 for both models, matching NB2/NB5",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "model": tier.model_id, "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "baseline_regression": b, "ft_regression": c,
              "original_baseline_regression": frozen["baseline_b"]["regression"],
              "original_ft_regression": verdict["comparison"][2]["regression"],
              "aggregate_scores_match": matches, "n_losses": len(losses),
              "loss_examples": losses, "all_examples": rows,
              "scope_note": "These are general-capability examples, not additional target-ticket losses. Instructor interpretation determines whether they satisfy rubric 3.4."}
    path = ROOT / "results/regression_comparison.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}; baseline={b:.6f}, FT={c:.6f}; matches original={matches}")
    print(f"Actual regression losses: {len(losses)}")
    print(json.dumps(losses, ensure_ascii=False, indent=2))
    if not matches:
        print("Supplemental scores differ: investigate before treating these as evidence for the original verdict.")


if __name__ == "__main__":
    main()
