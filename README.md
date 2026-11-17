# DATA 266 Lab 2 — Pair_Programming_Team_32

Members: Shreya Akotiya (GitHub `sakotiya`), Zoheb Waghu (Hugging Face `ZohebW`). Who did what: [CONTRIBUTIONS.md](CONTRIBUTIONS.md).
AI assistants used: [AI_USAGE.md](AI_USAGE.md).

| Part | Topic | Notebook (executed) | Tag |
|---|---|---|---|
| 1 | Multimodal RAG over three economics PDFs (Kaggle) | `notebooks/p1_rag.ipynb` (submission), `notebooks/p1_eval.ipynb` (golden set) | `p1-v1.0` |
| 2 | Fine-tuning Stable Diffusion 1.5 with LoRA (Naruto style) | `notebooks/p2_finetune.ipynb` | `p2-v1.0` |
| 3 | Multimodal multi-agent image assistant | `notebooks/p3_agents.ipynb` | `p3-v1.0` |

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

## Results at a glance

Every number below comes from the executed notebooks and the CSVs in each part's `metrics/` folder.

**Part 1** (golden set of 25 questions, `part1_rag/metrics/results_table.csv`; public Kaggle score 0.916)

| | Baseline (dense only) | Final (dense + BM25 hybrid) |
|---|---|---|
| Correct answers (/25) | 23 | 24 |
| Status accuracy | 0.96 | 0.96 |
| Recall@5 | 0.833 | 0.958 |
| Citation accuracy | 0.684 | 0.700 |
| Groundedness, mean 1–5 (2 raters) | – | 4.60 |
| Rater agreement within 1 point | – | 22/25 |
| Median s/question | 4.11 | 3.69 |

**Part 2** (`part2_diffusion/metrics/results_table.csv`, `train_*.json`; human win rates from `metrics/human_eval/`)

| | Base SD 1.5 | Run 1, LoRA r4 (final) | Run 2, LoRA r32 |
|---|---|---|---|
| KID (lower is better) | 0.124 ± 0.018 | 0.053 ± 0.025 | 0.076 ± 0.048 |
| CLIP score | 31.88 | 29.90 | 29.33 |
| LPIPS between seeds | 0.806 | 0.762 | 0.722 |
| Trainable parameters | – | 797,184 | 6,377,472 |
| GPU hours (Colab T4) | – | 0.705 | 0.716 |
| Human win rate vs base: style / prompt match | – | 82.7% / 29.3% | – |

**Part 3** (`part3_agents/metrics/results_table.csv`)

| | Single pass (baseline) | With critique loop (final) |
|---|---|---|
| Task success (/15) | 14 | 9 |
| Critic agreement with humans | – | 22/22 |
| Red-team attack success (/10) | – | 0 |
| Human agreement between members | 14/15 | 13/15 |
| Median s/request | 14.1 | 18.0 |

## What we did

### Part 1 — Multimodal RAG ([part1_rag/README.md](part1_rag/README.md))
- **Ingestion:** PyMuPDF; one chunk per page (51 pages) and 36 figure/table crops at 200 dpi. Each crop is
  transcribed once by a vision model and cached in `outputs/figure_descriptions.json`. Trust levels come from
  `data/kb_manifest.json`.
- **Index and retrieval:** ChromaDB with `BAAI/bge-small-en-v1.5`, plus BM25. The two are fused with
  reciprocal-rank fusion. A figure or table named in the question is looked up exactly and put first, a corrected
  figure is kept next to its original, and the two best hits of every document are always included.
- **Answers:** Claude `claude-sonnet-5-5` gets the page text and the figure images and returns structured JSON
  (answer, status, citations, figure references). A post-check validates labels and citations.
- **Memory:** one conversation object per `conversation_id`; user-supplied values stay in that conversation only.
- **Kaggle:** the notebook loads the code at tag `p1-v1.0` from a `repo.bundle` dataset and reads the API key from
  Kaggle Secrets. The public score went from 0.632 (text plus figure transcriptions) to 0.916 (figure images and exact
  figure lookup).
- **Golden set:** 25 questions written from the PDFs, none taken from `Questions.csv` (`eval/test_questions.csv`).
  They cover facts, near-identical figures, two-source questions, conflicts, false premises, injection, and two
  multi-turn conversations. Scored by `src/evaluate.py`.
- **Human evaluation:** both members rated the groundedness of the 25 final answers independently
  (`metrics/groundedness/`).
- **Safety:** the untrusted Appendix C of DOC3 is filtered out at retrieval, ruled out in the prompt and stripped
  from citations. A probe that forces Appendix C into the evidence fooled the model 0 of 12 times
  ([part1_rag/SAFETY.md](part1_rag/SAFETY.md)).

### Part 2 — Diffusion fine-tuning ([part2_diffusion/README.md](part2_diffusion/README.md))
- **Data:** `lambdalabs/naruto-blip-captions`, 1,221 images. Corrupted images, exact duplicates and near duplicates
  were checked for (none found). The 90/10 split with seed 266 gives 1,099 train and 122 test images, resized to 512×512.
- **Training:** two LoRA runs on the SD 1.5 UNet attention layers, differing only in rank (4 vs 32). Both use
  1,500 steps, batch 4 and learning rate 1e-4 on a Colab T4.
- **Generation:** 20 fixed prompts (10 in-domain, 5 out-of-domain, 5 compositional), seeds 0 and 1. Base model and
  both runs use the same sampler, with the safety checker on.
- **Evaluation:** KID, CLIP score, LPIPS diversity, a memorization check against the training set, and failure
  examples.
- **Human evaluation:** 25 blinded pairs (final run vs base) rated by 3 raters, both members and one outside rater.
- **Final model:** run 1 (rank 4), chosen for lower KID and the human style vote. Both LoRAs, with model cards, and
  the generated images are on Hugging Face.
- **Baseline by the non-leading partner:** `notebooks/p2_baseline_zoheb.ipynb` runs the same pipeline with rank 4
  and 500 steps (Colab T4, 14.3 min). KID 0.070 ± 0.023 (base 0.124, final run 0.053); CLIP score 30.71; LPIPS 0.823.

### Part 3 — Multi-agent system ([part3_agents/README.md](part3_agents/README.md))
- **Agents:** Planner, Vision, Prompt Engineering, Image Generation and Critique share one typed `State`. The
  critique loop allows at most 2 revisions, and every request writes a JSON trace.
- **Models:** Claude `claude-sonnet-5-5` for planning, vision, prompts and critique; SD 1.5 img2img (plus the
  Part 2 LoRA for Naruto style) and instruct-pix2pix for images; CLIP for scoring.
- **Evaluation:** 15 scenarios and 10 red-team cases, committed before running. Each was run with and without the
  critique loop, and both members marked every output independently (`part3_agents/MARKING.md`).
- **Safety:** input checks, a keyword filter, planner refusals, image text treated as untrusted, and the critic plus
  the SD safety checker. A system card is in the Part 3 README.
- **Gradio app:** the last cell of the notebook, or `python -m part3_agents.app.app`.

## Folders

```
notebooks/        executed notebooks for all parts
part1_rag/        src, eval (golden set), metrics, outputs, data (PDFs not committed)
part2_diffusion/  src, configs, eval (prompts), metrics, outputs, data (not committed)
part3_agents/     src, app, eval (scenarios, red team), metrics, outputs, test_images
```

Weights, datasets and generated images are not in git: they are on Hugging Face (links above).

## Setup

```bash
pip install -r requirements.txt
```

All required work runs on one 16 GB GPU (Kaggle or Colab T4). Fixed seed 266 throughout.

## How to run each part

- **Part 1 (Kaggle):**
  1. Open `notebooks/p1_rag.ipynb`. Attach the competition data and a private dataset holding `repo.bundle`
     (`git bundle create repo.bundle p1-v1.0 main`).
  2. Turn Internet on, add the secret `ANTHROPIC_API_KEY`, then Run All.
  3. It clones the code at tag `p1-v1.0`, reads the questions from `QUESTIONS_PATH` (default: the competition's
     `Questions.csv`) and writes `/kaggle/working/submission.csv`.

  The golden-set evaluation is `notebooks/p1_eval.ipynb`, with the same setup.
- **Part 2 (Colab T4):** `notebooks/p2_finetune.ipynb`. Setup, test run, full run and human evaluation are in
  [part2_diffusion/README.md](part2_diffusion/README.md).
- **Part 3 (Colab T4):** open `notebooks/p3_agents.ipynb`, add the secret `ANTHROPIC_API_KEY`, then Restart & Run
  All. The demo app is in the last cell.

## Hosted APIs used

| Part | API | Model version | Used for |
|---|---|---|---|
| 1 | Anthropic Claude API (`anthropic`) | `claude-sonnet-5-5` | Answers (page text plus figure images) |
| 1 | Google Gemini API (`google-genai`) | `gemini-3.8-flash`, `gemini-3.6-flash` | Figure/table transcriptions, made once and cached in `part1_rag/outputs/figure_descriptions.json` |
| 3 | Anthropic Claude API (`anthropic`) | `claude-sonnet-5-5` | Planner, Vision, Prompt Engineering and Critique agents |
