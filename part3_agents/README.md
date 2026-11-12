# Part 3 — Multimodal multi-agent system

| Folder | What goes here |
|---|---|
| `src/` | The 5 agents, shared state, orchestration |
| `app/` | Gradio demo app |
| `eval/` | 15 test scenarios + 10 red-team cases with rubrics, committed before running |
| `metrics/` | Success rate, critic agreement, red-team and timing CSVs |
| `outputs/` | JSON traces and small result images |
| `test_images/` | Input images used by the scenarios |

Notebook: `notebooks/p3_agents.ipynb`

## Flow

`src/pipeline.py:run()` passes one `State` dataclass (`src/agents.py`) through:

1. **Planner**: checks the file can be opened and is at least 64×64 px, runs a keyword filter, then Gemini picks the route (`caption`, `style_change`, `edit`, `domain_style`, `clarify`, `unsupported`, or `refuse`). Caption/question requests stop after Vision; `refuse`, `clarify`, `unsupported` and unreadable images stop here.
2. **Vision** (Gemini): caption, objects, verbatim visible text, answers to 2 fixed questions plus the user's question.
3. **Prompt Engineering** (Gemini): SD prompt or pix2pix instruction. Text from the image is passed in as untrusted data; a prompt containing a blocked term stops the request.
4. **Image Generation**: SD 1.5 img2img for style change, the same with the Part 2 LoRA for Naruto style, instruct-pix2pix for edits. Seed 266 + attempt, 30 steps; model, prompt, seed and settings are recorded.
5. **Critique**: CLIP score (ViT-B/32, ≥ 22) plus a Gemini judgement of the original vs. generated image and a safety check (Gemini + SD safety checker). On `revise` it sends one instruction back to Prompt Engineering and makes the image change more or less; at most 2 revisions, then the best safe attempt is shown.

Every request writes `outputs/traces/<run_id>.json` (each agent's input, output, seconds, error). Generated images go to `outputs/images/` (not committed).

Settings: `GEMINI_API_KEY`, `P3_VLM` (default `gemini-3.5-flash-lite`, same as Part 1), `P3_LORA` (Part 2 LoRA folder or Hugging Face repo id).

Offline check of the orchestration and metrics (no API or GPU needed): `python -m part3_agents.test_pipeline`.
