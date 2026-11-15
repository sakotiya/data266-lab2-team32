# AI Usage — Pair_Programming_Team_32

AI coding assistants we used while building the lab. The language and vision models that are part of the systems
themselves (Claude `claude-sonnet-5-5`, Gemini for figure transcriptions) are hosted APIs, listed with their
versions in `README.md`. Commits written with an assistant carry a `Co-Authored-By: Claude` line in git.

| Tool | Used by | What for |
|---|---|---|
| Claude Code (Anthropic) | Shreya | Reviewing the repository against the lab brief. Merging branches. Building the blind HTML rating sheets for the Part 2 human evaluation and the Part 1 groundedness ratings. Rewriting the Part 1 golden set from the three PDFs (the assistant checked each expected answer against the PDF pages; we reviewed them). The golden-set evaluation code and notebook. Fixing the Kaggle notebook (commit printed first, tagged code, pinned versions). Pinning `requirements.txt`. |
| Claude Code (Anthropic) | Zoheb | Rebuilding the Part 1 pipeline (figure crops, hybrid search, multimodal answers, conversation memory) and its notebooks. The Part 3 agents, orchestration, Gradio app, evaluation and README. <!-- TODO Zoheb: confirm and correct --> |
| <!-- TODO: any other assistant (ChatGPT, Copilot, Gemini chat, ...) --> | | |

We reviewed and ran all generated code ourselves. The results come from the executed notebooks, and the
human-evaluation scores are our own (and our outside rater's) judgements.
<!-- TODO both: confirm this paragraph is true before submitting -->

