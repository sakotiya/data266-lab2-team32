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
  - Rebuilt the pipeline (`c2956fe`):
    - figure and table crops with vision-model transcriptions;
    - hybrid retrieval with exact figure lookup and corrections kept next to originals;
    - multimodal JSON answers with Claude `claude-sonnet-5-5`, with a post-check of labels and citations;
    - per-conversation memory.

    Public score 0.916, team rank 7 on the public leaderboard.
  - Moved answering from Gemini to Claude and kept Gemini only for the cached figure transcriptions (`d03e000`).
  - The Colab and Kaggle notebooks:
    - code loaded from an uploaded `repo.bundle` instead of a GitHub token;
    - Kaggle secrets and competition input;
    - the executed Kaggle run at tag `p1-v1.0`, with 40 valid rows and a median of 3.26 s/question (`f4c857a`).
  - Ran the golden-set evaluation, baseline vs final (`403065d`).
  - The Part 1 safety analysis of DOC3 Appendix C (`part1_rag/SAFETY.md`), plus a probe that forces Appendix C into
    the evidence: 0 of 12 answers fooled (`src/probe_appendix_c.py`, `ef1ea4b`).
- **Both:** groundedness ratings of the 25 final answers, each rated independently.

## Part 2 — Diffusion fine-tuning (LoRA)
- **Shreya (lead):**
  - Pipeline code and configs: data cleaning and split, training, generation, evaluation, human-evaluation pairs, Hugging Face upload.
  - The 20 fixed prompts.
  - Both training runs (LoRA rank 4 and rank 32) on a Colab T4.
  - KID, CLIP, LPIPS and the memorization check.
  - The model card and the uploads to Hugging Face.
  - The executed notebook.
- **Zoheb (baseline, non-leading partner):**
  - Ran the complete Part 2 pipeline end to end on a Colab T4 as a working baseline (`notebooks/p2_baseline_zoheb.ipynb`):
    - data cleaning and the same split;
    - a LoRA rank 4 run for 500 steps;
    - generation for the 20 fixed prompts;
    - KID, CLIP score and LPIPS against the base model.
  - Results, from the executed notebook (commit `aefe087`, run from tag `p2-v1.0`):
    - KID 0.070 ± 0.023, against 0.124 ± 0.018 for the base model;
    - CLIP score 30.71 (base 31.88);
    - LPIPS between seeds 0.823 (base 0.806);
    - 797,184 trainable parameters, 14.3 training minutes (0.238 GPU-hours), 5.57 GB peak GPU memory;
    - 1 of 80 images flagged by the NSFW filter.

    With a third of the steps, the baseline already lowers KID against the base model. The final rank-4 run
    (1,500 steps) reaches 0.053.
- **Both:** human evaluation raters, with Punit Chittora as the rater from outside the team (25 blinded pairs).

## Part 3 — Multimodal multi-agent system
- **Zoheb (lead):**
  - The 15 test scenarios and 10 red-team cases with rubrics and test images, committed before running (`5189062`).
  - The five agents, with orchestration and a critique loop (`1bdd2b2`):
    - the agents are Planner, Vision, Prompt Engineering, Image Generation and Critique;
    - they share a typed `State`;
    - the critique loop allows at most 2 revisions, and every request writes a JSON trace.
  - Failure handling, the Gradio app, and an offline check of the orchestration (`test_pipeline.py`).
  - Switched the agents to Claude `claude-sonnet-5-5`, the same model as Part 1, and set the default LoRA to Part 2 run 2 (rank 32) from
    Hugging Face (`360fd71`).
  - The Colab T4 runs: 15 scenarios × 2 conditions plus 10 red-team cases, with traces, the results table and the
    executed notebook. Run 1 is `eaf8232`; the clean run 2 is `3d209ef` and `02a03d0`.
  - Output images on Hugging Face (`ZohebW/data266-team32-p3-outputs`).
  - The marking guide, his independent marks, and resolving disagreements with Shreya (`550dc08`).
  - The Part 3 README: results, safety notes and system card (`1d05e37`).
- **Shreya:**
  - The working baseline of the non-leading partner: a simple sequential version that calls the five agents once
    each, in order, with no revise loop (`part3_agents/simple_pipeline.py`).
  - Ran it on a Colab T4 on one scenario of each type: S01, S04, S07, S10 and S15
    (`notebooks/p3_simple_shreya.ipynb`, `metrics/simple_pipeline_results.csv`).
    - All 5 routes were correct.
    - The critic accepted S07 and asked for revisions on S04 and S10.
    - S15 (corrupted file) stopped with a clear message.
    - S04 and S07 include the first-time model downloads, so their times (110.9 s and 82.2 s) are not comparable to
      the team's timings.
  - Model and LoRA wiring (Hugging Face repo id for the Part 2 LoRA).
  - Independent pass/fail marks on every scenario; disagreements resolved with Zoheb (`part3_agents/MARKING.md`).

## Meetings

We met every two weeks from the start of the lab, and more often in November, on Zoom or by phone, to review
progress, plan the next steps and discuss design choices and trade-offs.

| Date | Format | Topic |
|---|---|---|
| Week of Oct 6, 2026 | Zoom | Kickoff: repository setup, work split across the three parts, Part 1 first milestone |
| Week of Oct 20, 2026 | Zoom / phone | Part 1 progress (parsing, retrieval), Part 2 data preparation and training plan |
| Nov 3, 2026 | Zoom / phone | Part 2 pipeline and prompts, Part 3 agent design, Part 1 plan for answers and Kaggle |
| Nov 8, 2026 | Zoom / phone | Part 2 training results (rank 4 vs rank 32), Part 1 parsing, indexing and hybrid search, split of answer generation, first golden set |
| Nov 12, 2026 | Zoom / phone | Part 1 rebuild with Claude, Part 3 Colab run and independent human marking, which Part 2 LoRA Part 3 uses |
| Nov 14, 2026 | Zoom / phone | Merging branches, Part 2 human evaluation results, golden-set rewrite and evaluation, Kaggle notebook, report and submission plan |

