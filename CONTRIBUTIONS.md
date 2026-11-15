# Contributions — Pair_Programming_Team_32

Members: Shreya Akotiya (GitHub `sakotiya`), Zoheb Waghu. Both members worked on all three parts; the git history
(`git log --format='%an %ad %s' --date=short`) shows who committed what.

## Part 1 — Multimodal RAG (Kaggle)
- **Shreya:**
  - Knowledge-base config (`kb_manifest.json`) and submission validator in the repo.
  - First pipeline: PDF parsing, ChromaDB indexing and hybrid (dense + BM25) search.
  - First answer generation with Gemini, the first Kaggle notebooks, and vision extraction of figures and tables (public score 0.632).
  - The golden set: first version on Nov 8; rewritten on Nov 14 because most of it overlapped with `Questions.csv`.
  - The golden-set evaluation: `src/evaluate.py`, `notebooks/p1_eval.ipynb`, and the groundedness rating sheet.
- **Zoheb:**
  - Rebuilt the pipeline: figure crops with vision-model transcriptions, hybrid retrieval with exact figure lookup, multimodal JSON answers with Claude, and per-conversation memory (public score 0.916).
  - The Kaggle notebook that loads the code from a `repo.bundle` dataset (`notebooks/p1_rag.ipynb`).
  - Ran the golden-set evaluation. <!-- TODO: confirm once the run is pushed -->
- **Both:** groundedness ratings of the 25 final answers. <!-- TODO: confirm once rated -->

## Part 2 — Diffusion fine-tuning (LoRA)
- **Shreya (lead):**
  - Pipeline code and configs: data cleaning and split, training, generation, evaluation, human-evaluation pairs, Hugging Face upload.
  - The 20 fixed prompts.
  - Both training runs (LoRA rank 4 and rank 32) on a Colab T4.
  - KID, CLIP, LPIPS and the memorization check.
  - The model card and the uploads to Hugging Face.
  - The executed notebook.
- **Zoheb:** <!-- TODO: the brief requires the non-leading partner to implement and run the complete pipeline at least as a working baseline. Describe Zoheb's baseline run (where it is: notebook or HF link). It is not in this repository yet. -->
- **Both:** human evaluation raters, with Punit Chittora as the rater from outside the team (25 blinded pairs).

## Part 3 — Multimodal multi-agent system
- **Zoheb (lead):**
  - The five agents, the shared typed state, and orchestration with the critique loop (at most 2 revisions) and JSON traces.
  - Failure handling and the Gradio app.
  - The 15 scenarios and 10 red-team cases (committed before running).
  - The Colab T4 runs, the results table, the safety notes and the system card.
- **Shreya:**
  - A simple sequential version that calls the five agents in order (`part3_agents/simple_pipeline.py`), as the working baseline of the non-leading partner.
  - Model and LoRA wiring (Hugging Face repo id for the Part 2 LoRA).
  - Independent pass/fail marks on every scenario; disagreements resolved with Zoheb (`part3_agents/MARKING.md`).

## Meetings

We met every two weeks from the start of the lab, on Zoom or by phone, to review progress, plan the next steps and
discuss design choices and trade-offs.

| Date (week of) | Format | Topic |
|---|---|---|
| Oct 6, 2026 | Zoom | Kickoff: repository setup, work split across the three parts, Part 1 first milestone |
| Oct 20, 2026 | Zoom / phone | Part 1 progress (parsing, retrieval), Part 2 data preparation and training plan |
| Nov 3, 2026 | Zoom / phone | Part 2 training results, Part 3 agent design, Part 1 Kaggle approach and rebuild, evaluation and report plan |
<!-- TODO: check these dates and topics against your Zoom history and call logs -->

