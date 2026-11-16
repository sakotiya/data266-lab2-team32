# Commit IDs before and after the history rewrite (2026-11-16)

On 2026-11-16 we rewrote the git history to remove `Co-Authored-By` lines from commit messages. Only the commit
messages changed: every file, author and date is the same. Because a commit's ID includes its message, the IDs changed.

Notebook outputs printed the commit they ran on, so they show the old IDs. This table maps each one to the new ID. The
documentation files use the new IDs.

| Old ID (printed in the notebook output) | New ID | Where it appears |
|---|---|---|
| `dd73b67` | `57b061c` | notebooks/p1_eval.ipynb |
| `2dd936d` | `403065d` | notebooks/p1_rag.ipynb (Kaggle run) |
| `18e9f8a` | `4b09535` | notebooks/p3_agents.ipynb |
| `4bd8bce` | `d7b7cbe` | notebooks/p3_simple_shreya.ipynb |
| `285e5f3` | `370b7e6` | notebooks/p2_baseline_zoheb.ipynb (Zoheb's run at p2-v1.0) |

The full old-to-new map for all commits is in `docs/commit-map.txt`.
