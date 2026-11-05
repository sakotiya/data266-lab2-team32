"""Results tables for the report, built only from the metrics files (C2.2: every number comes
from a notebook output). Writes metrics/results_table.md and metrics/training_table.md.
"""

from __future__ import annotations

from .common import MODELS, RUNS, read_json, repo_dir
from .figures import LABEL


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def training_table() -> str:
    rows = []
    for r in RUNS:
        t = read_json(repo_dir("metrics") / f"train_{r}.json")
        rows.append([LABEL[r], f"{t['trainable_params']:,} ({t['trainable_pct']}%)",
                     t["learning_rate"], t["steps"], t["batch_size"], t["epochs"],
                     f"{t['peak_gpu_mem_allocated_gb']} GB", f"{t['train_loop_s'] / 60:.1f} min",
                     f"{t['gpu_hours']:.3f}", f"{t['final_loss_last100']:.4f}"])
    md = _table(["Run", "Trainable params", "LR", "Steps", "Batch", "Epochs", "Peak GPU mem",
                 "Training time", "GPU-hours (incl. load)", "Loss, last 100 steps"], rows)
    (repo_dir("metrics") / "training_table.md").write_text(md + "\n")
    return md


def results_table() -> str:
    s = read_json(repo_dir("metrics") / "summary_auto.json")
    hp = repo_dir("metrics") / "human_eval_summary.json"
    h = read_json(hp) if hp.exists() else {}
    final = s["final_run"]

    def human(m):
        if not h.get("raters"):
            return "pending"
        if m == final:
            return (f"style {h['style']['win_rate_all_votes']:.0%}, prompt "
                    f"{h['prompt']['win_rate_all_votes']:.0%} (vs base, {h['raters']} raters)")
        return "reference" if m == "base" else "not rated"

    rows = []
    for m in MODELS:
        v = s[m]
        eff = ("- (no training)" if m == "base" else
               f"{v['gpu_hours']:.3f} GPU-h, {v['peak_gpu_mem_gb']} GB peak")
        name = LABEL[m] + (" **(final)**" if m == final else "") + (" (baseline)" if m == "base" else "")
        rows.append([name, f"{v['kid_mean']:.4f} ± {v['kid_std']:.4f}", f"{v['clip_all']:.2f}",
                     f"{v.get('clip_in_domain', float('nan')):.2f}", f"{v['lpips_mean']:.3f}",
                     human(m), eff, f"{v['median_s_per_image']:.2f} s"])
    md = _table(["Model", "KID in-domain ↓ (primary)", "CLIP all ↑ (primary)", "CLIP in-domain",
                 "LPIPS seeds (secondary)", "Human win rate", "Efficiency (training)",
                 "Median s/image"], rows)
    md += (f"\n\nFinal run: **{LABEL[final]}** - rule: {s['final_run_rule']}. KID uses "
           f"{s[final]['kid_n_fake']} generated vs {s[final]['kid_n_real']} test images "
           f"(subset size {s[final]['kid_subset_size']}), so it is noisy.")
    (repo_dir("metrics") / "results_table.md").write_text(md + "\n")
    return md
