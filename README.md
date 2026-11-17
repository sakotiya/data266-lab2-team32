# DATA 266 — Lab 2 · Team 32

Multimodal RAG · Diffusion fine-tuning with LoRA · Multimodal multi-agent system

**Members:** Shreya Akotiya (GitHub `sakotiya`) · Zoheb Waghu (Hugging Face `ZohebW`)
**Final report:** [report/Lab2_Report_IEEE_final.pdf](report/Lab2_Report_IEEE_final.pdf)

Both members worked on all three parts. Zoheb built Part 1's final pipeline and led Part 3; Shreya led Part 2. In
Parts 2 and 3, the partner who did not lead also ran the complete pipeline as a working baseline. Every part has:
- a test set written before tuning;
- a baseline and a final system;
- a primary metric, a secondary metric, a human evaluation and a cost measure;
- a safety analysis.

Who did what: [CONTRIBUTIONS.md](CONTRIBUTIONS.md) · AI assistants used: [AI_USAGE.md](AI_USAGE.md)

## Results at a glance

| Part | Baseline | Final system |
|---|---|---|
| **1 · Multimodal RAG** (25-question golden set) | Dense retrieval only: 23/25 correct · Recall@5 0.833 · citation accuracy 0.684 · 4.11 s/question | Hybrid dense + BM25: **24/25 correct** · Recall@5 **0.958** · citation accuracy 0.700 · groundedness **4.60/5** (2 raters, 22/25 within 1 point) · 3.69 s/question · **Kaggle public score 0.916** |
| **2 · LoRA fine-tuning** (SD 1.5, Naruto style) | Base SD 1.5: KID 0.124 ± 0.018 · CLIP 31.88 · LPIPS 0.806 | Run 1, LoRA rank 4: **KID 0.053 ± 0.025** · CLIP 29.90 · LPIPS 0.762 · human style win **82.7%** · prompt-match win 29.3% (3 raters, 25 pairs) |
| **3 · Multi-agent system** (15 scenarios, 10 red-team cases) | Single pass, no critique: **14/15 task success** · 14.1 s median | Critique loop (≤ 2 revisions): 9/15 · critic agreement 22/22 · **red-team attacks 0/10** · 18.0 s median |

Other runs:
- **Part 2, run 2 (LoRA rank 32):** KID 0.076 ± 0.048 · CLIP 29.33 · LPIPS 0.722.
- **Part 2, Zoheb's baseline (rank 4, 500 steps):** KID 0.070 ± 0.023 · CLIP 30.71 · LPIPS 0.823.
- **Part 3, Shreya's simple sequential baseline:** 5/5 routes correct on S01, S04, S07, S10 and S15.

## Hardware and training time

| Part | Hardware | Time |
|---|---|---|
| 1 · RAG (Kaggle run) | Kaggle CPU (no GPU needed: the answer model is a hosted API) · Python 3.13 | 40 public questions in 335 s · median 3.26 s/question |
| 1 · Golden-set evaluation | Local CPU · Python 3.11 | Median 4.11 s (baseline) / 3.69 s (final) per question |
| 2 · LoRA training | NVIDIA Tesla T4 16 GB (Google Colab) · Python 3.13, PyTorch 2.14.1 | Run 1: 42.3 min (0.705 GPU-h, 3.54 GB peak) · Run 2: 42.9 min (0.716 GPU-h, 3.60 GB) · Zoheb's baseline: 14.3 min (0.238 GPU-h) |
| 3 · Agents | NVIDIA Tesla T4 16 GB (Google Colab) · Python 3.13, PyTorch 2.14.1 | Median 14.1 s/request (single pass), 18.0 s (critique loop) |

All required work runs on one 16 GB GPU. The seed is 266 throughout. The first cell of every notebook prints the
commit, library versions, GPU and seed.

## What we did

### Part 1 — Multimodal RAG over three economics PDFs (Kaggle)
- **Ingestion:** PyMuPDF gives 51 page chunks and 36 figure/table crops at 200 dpi. Gemini transcribed each crop
  once, and the transcriptions are cached in `part1_rag/outputs/figure_descriptions.json`. Trust levels come from
  `kb_manifest.json`.
- **Retrieval:**
  - ChromaDB (`BAAI/bge-small-en-v1.5`) plus BM25, fused with reciprocal-rank fusion (k = 60).
  - A figure or table named in the question is looked up exactly and put first.
  - A corrected figure travels with its original.
  - The two best hits of every document are always included, so conflicts stay visible.
- **Answers:** Claude `claude-sonnet-5-5` reads the page text *and the figure images* and returns structured JSON
  (answer, status, citations, figure references). A post-check validates labels and drops untrusted citations.
- **Memory:** one conversation per `conversation_id`. User-supplied values never change the knowledge base.
- **Development history:** the public score went from 0.632 (text plus figure transcriptions) to 0.916 (figure
  images and exact figure lookup).
- **Golden set:** 25 questions written from the PDFs, none taken from `Questions.csv`. They cover facts,
  near-identical figures, two sources, conflicts, false premises, injection, and two multi-turn conversations. Both
  members rated groundedness independently.
- **Safety:** the untrusted Appendix C is filtered out at retrieval, ruled out in the prompt and stripped from
  citations. A probe that forces it into the evidence fooled the model 0 of 12 times.

### Part 2 — Fine-tuning Stable Diffusion 1.5 with LoRA
- **Data:** `lambdalabs/naruto-blip-captions`, 1,221 images. Corrupted images, exact duplicates and near duplicates
  were checked for (none found). The 90/10 split with seed 266 gives 1,099 train and 122 test images at 512×512.
- **Training:** two runs that differ only in LoRA rank (4 vs 32). Both train on the UNet attention layers with
  1,500 steps, batch 4 and learning rate 1e-4.
- **Generation:** 20 fixed prompts (10 in-domain, 5 out-of-domain, 5 compositional) × seeds 0 and 1, for the base
  model and both runs, with the same sampler.
- **Evaluation:**
  - KID and CLIP score (primary), LPIPS between seeds (secondary);
  - a blinded human evaluation of 25 pairs by 3 raters (both members and Punit Chittora, from outside the team);
  - a memorization check, and failure examples.
- **Final model:** run 1 (rank 4), chosen for lower KID, the human style vote and fewer NSFW false positives.
- **Safety:** the dataset license is for non-commercial use (copyrighted characters), the SD safety checker stays on,
  and we ran a memorization check. 8 of 80 LoRA images were false NSFW positives; we report them rather than turn the
  filter off.

### Part 3 — Multimodal multi-agent system
- **Agents:** Planner → Vision → Prompt Engineering → Image Generation → Critique. They share one typed `State`, the
  critique loop allows at most 2 revisions, and every request writes a JSON trace.
- **Models:**
  - Claude `claude-sonnet-5-5` for planning, vision, prompts and critique;
  - SD 1.5 img2img for style changes, with the Part 2 LoRA for Naruto style;
  - instruct-pix2pix for edits;
  - CLIP for scoring.
- **Evaluation:** 15 scenarios and 10 red-team cases, committed before running. Each was run with and without the
  critique loop, and both members marked every output independently.
- **Finding:** the critic judged correctly (22/22), but its revisions drifted from the input, so the single pass is
  better (14/15 vs 9/15) and faster.
- **Safety and demo:**
  - five safety layers, and 0 of 10 red-team attacks succeeded;
  - a system card in [part3_agents/README.md](part3_agents/README.md);
  - a Gradio app for the demo.

## Quick start

```bash
git clone https://github.com/sakotiya/data266-lab2-team32.git
cd data266-lab2-team32
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the package versions printed by the executed notebooks. Hosted APIs need keys, given as
secrets or environment variables, never committed:
- `ANTHROPIC_API_KEY` for Parts 1 and 3;
- `GEMINI_API_KEY` for Part 1, only if a figure transcription is missing from the cache.

Offline check of the Part 3 orchestration (no API or GPU needed):

```bash
python -m part3_agents.test_pipeline
```

## Reproducing a run

Datasets, PDFs, weights and generated images are not committed. Each part's notebook says where they come from.

- **Part 1 (Kaggle, the graded path)**, `notebooks/p1_rag.ipynb`:
  1. Make the code bundle: `git bundle create repo.bundle p1-v1.0 main`.
  2. Upload `repo.bundle` as a private Kaggle dataset.
  3. In the notebook, attach the competition data and that dataset, turn Internet on and add the secret
     `ANTHROPIC_API_KEY`.
  4. Run All. It clones tag `p1-v1.0`, answers `Questions.csv` (or the file in `QUESTIONS_PATH`) and writes
     `/kaggle/working/submission.csv`.
- **Part 1 golden-set evaluation:** `notebooks/p1_eval.ipynb` (same setup). It writes
  `part1_rag/metrics/results_table.csv`. The groundedness rating sheet is built with
  `python -m part1_rag.src.groundedness_sheet`.
- **Part 2 (Colab T4):**
  - `notebooks/p2_finetune.ipynb`: set `TEST_RUN = True` for a 15-minute check, then `False` for the full run
    (about 3 hours, resumable).
  - Setup and the human evaluation: [part2_diffusion/README.md](part2_diffusion/README.md).
  - Zoheb's baseline: `notebooks/p2_baseline_zoheb.ipynb`.
- **Part 3 (Colab T4):**
  - `notebooks/p3_agents.ipynb`: add the secret `ANTHROPIC_API_KEY`, then Run All. The Gradio app is the last
    cell, or run `python -m part3_agents.app.app` locally.
  - Shreya's baseline: `notebooks/p3_simple_shreya.ipynb`.

## Repository structure

```
data266-lab2-team32/
├── README.md
├── CONTRIBUTIONS.md          who did what in each part, meetings
├── AI_USAGE.md               AI coding assistants used, and for what
├── COMMIT_MAP.md             old → new commit IDs (history rewritten on 2026-11-16)
├── requirements.txt          pinned versions
├── report/
│   └── Lab2_Report_IEEE_final.pdf
├── notebooks/                executed notebooks (outputs saved)
│   ├── p1_rag.ipynb          Part 1 Kaggle submission notebook
│   ├── p1_eval.ipynb         Part 1 golden-set evaluation
│   ├── p2_finetune.ipynb     Part 2 data, training, generation, evaluation, human eval
│   ├── p2_baseline_zoheb.ipynb
│   ├── p3_agents.ipynb       Part 3 runs, evaluation and Gradio app
│   └── p3_simple_shreya.ipynb
├── part1_rag/
│   ├── src/                  ingest, index, search, answer, pipeline, evaluate, probe
│   ├── eval/                 test_questions.csv (golden set)
│   ├── metrics/              golden_*.csv, results_table.csv, groundedness/, appendix_c_probe.csv
│   ├── outputs/              figure_descriptions.json (cache), submission.csv
│   ├── data/                 kb_manifest.json, validate_submission.py (PDFs not committed)
│   ├── README.md
│   └── SAFETY.md             Appendix C analysis
├── part2_diffusion/
│   ├── src/                  data, train, generate, evaluate, human_eval, upload
│   ├── configs/config.yaml
│   ├── eval/prompts.csv      the 20 fixed prompts
│   ├── metrics/              results_table.csv, auto_metrics.csv, train_*.json, human_eval/, memorization.csv
│   ├── outputs/              loss curves, failure examples, memorization pairs, split manifest
│   └── README.md
├── part3_agents/
│   ├── src/                  agents, pipeline, models, evaluate
│   ├── app/app.py            Gradio demo
│   ├── eval/                 scenarios.csv, redteam.csv (written before running)
│   ├── metrics/              results_table.csv, scenario_results.csv, redteam_results.csv, human_marks.csv
│   ├── outputs/traces/       JSON trace of every request
│   ├── test_images/
│   ├── simple_pipeline.py    Shreya's sequential baseline
│   ├── test_pipeline.py      offline check (no API or GPU)
│   ├── MARKING.md            marking guide and how disagreements were resolved
│   └── README.md             results, safety notes, system card
└── docs/commit-map.txt
```

## Where to find what

| Looking for | Where |
|---|---|
| Final report | `report/Lab2_Report_IEEE_final.pdf` |
| Results table of each part | `part1_rag/metrics/results_table.csv` · `part2_diffusion/metrics/results_table.csv` · `part3_agents/metrics/results_table.csv` |
| Test sets written before tuning | `part1_rag/eval/test_questions.csv` · `part2_diffusion/eval/prompts.csv` · `part3_agents/eval/scenarios.csv`, `redteam.csv` |
| Human evaluation | Part 1: `part1_rag/metrics/groundedness/` · Part 2: `part2_diffusion/metrics/human_eval/` · Part 3: `part3_agents/metrics/human_marks.csv`, `MARKING.md` |
| Safety analysis | `part1_rag/SAFETY.md` · Part 2 section of the report · `part3_agents/README.md` (safety notes, system card) |
| Kaggle submission | `part1_rag/outputs/submission.csv` |
| Agent traces | `part3_agents/outputs/traces/` |
| Baselines by the non-leading partner | `notebooks/p2_baseline_zoheb.ipynb` · `notebooks/p3_simple_shreya.ipynb`, `part3_agents/metrics/simple_pipeline_results.csv` |
| Old commit IDs printed in notebook outputs | `COMMIT_MAP.md` |

## Links

| What | Link |
|---|---|
| Repository | https://github.com/sakotiya/data266-lab2-team32 |
| Kaggle competition | https://www.kaggle.com/competitions/data-266-fall-2026-lab-2-part-1-multimodal-rag |
| Kaggle notebook (final) | https://www.kaggle.com/code/zohebwaghu28/p1-rag?scriptVersionId=369796001 |
| Part 2 model, run 1 (LoRA rank 4, final) | https://huggingface.co/sakotiya/data266-team32-naruto-run1-lora-r4 |
| Part 2 model, run 2 (LoRA rank 32) | https://huggingface.co/sakotiya/data266-team32-naruto-run2-lora-r32 |
| Part 2 generated images | https://huggingface.co/datasets/sakotiya/data266-team32-p2-images |
| Part 3 output images | https://huggingface.co/datasets/ZohebW/data266-team32-p3-outputs |

Tags: `p1-v1.0`, `p2-v1.0`, `p3-v1.0`.

## Notes

- **Hosted APIs.**

  | Part | API | Model | Used for |
  |---|---|---|---|
  | 1 | Anthropic Claude | `claude-sonnet-5-5` | Answers (page text plus figure images) |
  | 1 | Google Gemini | `gemini-3.8-flash`, `gemini-3.6-flash` | Figure and table transcriptions, made once and cached in the repo |
  | 3 | Anthropic Claude | `claude-sonnet-5-5` | Planner, Vision, Prompt Engineering and Critique |

  All other models (embeddings, BM25, SD 1.5, LoRA, instruct-pix2pix, CLIP, and the KID and LPIPS networks) run
  locally.
- **Part 3 LoRA.** Part 3 was built and evaluated with the rank-32 adapter (run 2). Part 2's final model is rank 4.
  `P3_LORA=sakotiya/data266-team32-naruto-run1-lora-r4` switches it.
- **Golden set.** The first version (Nov 8) overlapped with `Questions.csv`, so it was rewritten on Nov 14, before
  the evaluation ran. Both versions are in the git history.
- **Commit IDs.** On 2026-11-16 the history was rewritten to remove co-author lines from commit messages; the files
  did not change. Notebook outputs show the old IDs. [COMMIT_MAP.md](COMMIT_MAP.md) maps them to the new ones.
- **Weights and data.** Weights, datasets and generated images are on Hugging Face, not in git. Secrets are never
  committed.
- **License.** The Naruto images are fan-collected frames of a copyrighted franchise, so the LoRA adapters and their
  outputs are for coursework and non-commercial research only.

## References

- Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Faysse et al. (2024), ColPali: Efficient Document Retrieval with Vision Language Models
- Greshake et al. (2023), Not What You've Signed Up For: Indirect Prompt Injection
- Cormack, Clarke & Büttcher (2009), Reciprocal Rank Fusion
- Robertson & Zaragoza (2009), The Probabilistic Relevance Framework: BM25 and Beyond
- Rombach et al. (2022), High-Resolution Image Synthesis with Latent Diffusion Models
- Hu et al. (2021), LoRA: Low-Rank Adaptation of Large Language Models
- Carlini et al. (2023), Extracting Training Data from Diffusion Models
- Bińkowski et al. (2018), Demystifying MMD GANs (KID)
- Hessel et al. (2021), CLIPScore
- Zhang et al. (2018), The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)
- Yao et al. (2022), ReAct: Synergizing Reasoning and Acting in Language Models
- Madaan et al. (2023), Self-Refine: Iterative Refinement with Self-Feedback
- Brooks et al. (2023), InstructPix2Pix: Learning to Follow Image Editing Instructions
