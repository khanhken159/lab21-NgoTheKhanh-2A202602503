"""Recover full per-ticket evidence without training or replacing frozen results.

Run in the existing Colab checkout. This is a retrospective generation pass,
not a new baseline chosen after training. Prompts and eval data stay unchanged.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "src"))


def comparison_rows(target, base_preds, ft_preds, score):
    if not (len(target) == len(base_preds) == len(ft_preds)):
        raise ValueError("Prediction counts must match the full target set")
    rows = []
    for i, (rec, base, ft) in enumerate(zip(target, base_preds, ft_preds)):
        b, c = score(base, rec["label"]), score(ft, rec["label"])
        rows.append({"i": i, "ticket": rec["input"], "label": rec["label"],
                     "base_pred": base, "ft_pred": ft, "base_score": b,
                     "ft_score": c, "delta": c-b,
                     "outcome": "FT thua" if c < b else "FT thắng" if c > b else "Hòa"})
    return rows


def main():
    from labkit import evaluate as ev, generate
    from labkit.config import get_tier
    from peft import PeftModel

    frozen = json.loads((ROOT / "results/baselines_frozen.json").read_text(encoding="utf-8"))
    verdict = json.loads((ROOT / "results/verdict.json").read_text(encoding="utf-8"))
    tier = get_tier(frozen["tier"])
    if tier.model_id != frozen["model"]:
        raise RuntimeError("Model differs from the frozen experiment")
    if hashlib.sha256(generate.OPTIMIZED_PROMPT.encode()).hexdigest()[:16] != frozen["optimized_prompt_sha"]:
        raise RuntimeError("Optimized prompt changed; restore the original prompt")
    eval_path = ROOT / "data/eval_target.jsonl"
    checksums = json.loads((ROOT / "data/checksums.json").read_text(encoding="utf-8"))
    if hashlib.sha256(eval_path.read_bytes()).hexdigest()[:16] != checksums["eval_target.jsonl"]:
        raise RuntimeError("Target dataset checksum does not match the original corpus")
    target = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if frozen.get("smoke_mode") or len(target) != frozen["n_target"]:
        raise RuntimeError("This supplement requires the full, unchanged eval set")
    prompts = [rec["input"] for rec in target]

    print("Generate baseline (b) with the unchanged frozen prompt; no training.", flush=True)
    model, tok = generate.load_base(tier)
    model.eval()
    base_preds, _ = generate.generate_batch(model, tok, prompts,
        system=generate.OPTIMIZED_PROMPT, label="supplement/baseline-b")
    del model, tok
    generate.free_memory()

    print("Generate the saved correct adapter; no training.", flush=True)
    model, tok = generate.load_base(tier)
    model = PeftModel.from_pretrained(model, str(ROOT / "adapters/correct"))
    model.eval()
    ft_preds, _ = generate.generate_batch(model, tok, prompts,
        system=generate.NAIVE_PROMPT, label="supplement/correct")
    del model, tok
    generate.free_memory()

    rows = comparison_rows(target, base_preds, ft_preds, ev.triage_field_accuracy)
    baseline_score = sum(r["base_score"] for r in rows)/len(rows)
    ft_score = sum(r["ft_score"] for r in rows)/len(rows)
    original_ft = verdict["comparison"][2]["target"]
    matches = abs(baseline_score-frozen["baseline_b"]["target"]) < 0.00005 and abs(ft_score-original_ft) < 0.00005
    losses = sorted((r for r in rows if r["delta"] < 0), key=lambda r: r["delta"])
    wins = sorted((r for r in rows if r["delta"] > 0), key=lambda r: -r["delta"])
    selected = losses[:2] + wins[:2]
    selected += [r for r in rows if r not in selected][:max(0, 5-len(selected))]
    output = {"provenance": "Retrospective greedy generation with unchanged prompts, full eval and saved adapter; original frozen artifacts preserved",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "model": tier.model_id, "target_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
              "optimized_prompt_sha": frozen["optimized_prompt_sha"],
              "baseline_target": baseline_score, "ft_target": ft_score,
              "original_baseline_target": frozen["baseline_b"]["target"],
              "original_ft_target": original_ft, "aggregate_scores_match": matches,
              "n_losses": len(losses), "n_wins": len(wins),
              "qualitative_requirement_met": matches and len(losses) >= 2 and len(selected) >= 5,
              "selected": selected, "all_examples": rows}
    path = ROOT / "results/qualitative_comparison.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}")
    print(f"Baseline={baseline_score:.4f}, FT={ft_score:.4f}, matches original scores={matches}")
    print(f"FT loses on {len(losses)} tickets; wins on {len(wins)} tickets.")
    if not matches:
        print("Scores differ from the frozen run: retain this as supplementary evidence and investigate; do not overwrite original scores.")
    if len(losses) < 2:
        print("Fewer than two actual losses: disclose this limitation. Do not fabricate examples or alter eval labels.")
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
