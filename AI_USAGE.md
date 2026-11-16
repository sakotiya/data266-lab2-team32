# AI Usage — Pair_Programming_Team_32

AI coding assistants we used while building the lab. The language and vision models that are part of the systems
themselves (Claude `claude-sonnet-5-5`, Gemini for figure transcriptions) are hosted APIs, listed with their
versions in `README.md`. Commits made with an assistant used to carry a `Co-Authored-By: Claude` line. We removed
those lines from the history on 2026-11-16 ([COMMIT_MAP.md](COMMIT_MAP.md)); this file is the record of what the assistant did.

| Tool | Used by | What for |
|---|---|---|
| Claude Code (Anthropic) | Shreya | Reviewing the repository against the lab brief. Merging branches. Building the blind HTML rating sheets for the Part 2 human evaluation and the Part 1 groundedness ratings. Rewriting the Part 1 golden set from the three PDFs (the assistant checked each expected answer against the PDF pages; we reviewed them). The golden-set evaluation code and notebook. Fixing the Kaggle notebook (commit printed first, tagged code, pinned versions). Pinning `requirements.txt`. The notebooks for the non-leading partners' baselines (`p2_baseline_zoheb.ipynb`, `p3_simple_shreya.ipynb`) and the import fix in `simple_pipeline.py`. Drafting the README, CONTRIBUTIONS.md and this file. Drafting the report and the architecture diagrams, and converting the report to IEEE layout; we reworded the report ourselves. |
| Claude Code (Anthropic) | Zoheb | Rebuilding the Part 1 pipeline (figure crops, hybrid search, multimodal answers, conversation memory) and its notebooks. The Part 3 agents, orchestration, Gradio app, evaluation and README. |
| <!-- TODO: any other assistant (ChatGPT, Copilot, Gemini chat, ...) --> | | |

We reviewed and ran all generated code ourselves. The results come from the executed notebooks, and the
human-evaluation scores are our own (and our outside rater's) judgements.
<!-- TODO both: confirm this paragraph is true before submitting -->

