"""Orchestration: Planner -> Vision -> (Prompt -> Generate -> Critique, revise at most twice). One JSON trace per request."""
import json
import time
import uuid

from . import models
from .agents import SETTINGS, State, UnsafePrompt, critique, generate, planner, prompt_engineer, vision

MAX_REVISIONS = 2
TRACES = models.OUTPUTS / "traces"
IMAGES = models.OUTPUTS / "images"
STOP_AFTER_PLANNER = {"refuse", "clarify", "unsupported", "error"}


def step(s, agent, fn, inputs, *args):
    """Run one agent, timing it and recording its input/output (or error) in the trace."""
    entry = {"agent": agent, "input": inputs}
    t = time.perf_counter()
    try:
        out = fn(s, *args)
        entry["output"] = out[1] if isinstance(out, tuple) else out     # generate returns (image, meta)
        return out
    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        entry["seconds"] = round(time.perf_counter() - t, 2)
        s.trace.append(entry)


def adjust(settings, how):
    """Critic's `adjust`: change the image more or less on the next try."""
    out, d = dict(settings), {"more_change": 1, "less_change": -1}.get(how, 0)
    if "strength" in out:
        out["strength"] = round(min(max(out["strength"] + 0.1 * d, 0.3), 0.9), 2)
    else:                   # pix2pix: lower image guidance = further from the input
        out["image_guidance_scale"] = round(min(max(out["image_guidance_scale"] - 0.3 * d, 1.0), 2.5), 2)
    return out


def run(image_path, request, use_critic=True, max_revisions=MAX_REVISIONS, save=True):
    s = State(request=request, image_path=str(image_path), use_critic=use_critic)
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    t0 = time.perf_counter()
    try:
        step(s, "planner", planner, {"request": request, "image": str(image_path)})
        if s.route in STOP_AFTER_PLANNER:
            return s
        step(s, "vision", vision, {"request": request, "route": s.route})
        if s.route == "caption":
            return s

        settings, feedback = dict(SETTINGS[s.route]), ""
        for attempt in range(max_revisions + 1):
            step(s, "prompt_engineering", prompt_engineer, {"route": s.route, "critic_feedback": feedback}, feedback)
            image, meta = step(s, "image_generation", generate, {"prompt": s.prompt, "settings": settings}, settings)
            if save:
                IMAGES.mkdir(parents=True, exist_ok=True)
                meta["output_image"] = str((IMAGES / f"{run_id}_a{attempt}.png").relative_to(models.PART))
                image.save(models.PART / meta["output_image"])
            a = {"attempt": attempt, "image": image, "generation": meta}
            s.attempts.append(a)
            if not use_critic:
                if meta["sd_safety_checker_nsfw"]:
                    s.message = "The generated image was flagged as unsafe by the safety checker and is not shown."
                else:
                    s.final_image, s.message = image, "Done (no critique loop)."
                break
            a["critique"] = step(s, "critique", critique, {"attempt": attempt}, image, meta)
            if a["critique"]["decision"] == "accept":
                s.final_image = image
                s.message = f"Done after {attempt} revision(s). {s.prompt['target_description']}"
                break
            feedback, settings = a["critique"]["revision_instruction"], adjust(settings, a["critique"]["adjust"])
        else:
            safe = [a for a in s.attempts if a["critique"]["safe"]]
            if safe:
                best = max(safe, key=lambda a: (a["critique"]["score"], a["critique"]["clip_score"]))
                s.final_image = best["image"]
                s.message = (f"Best of {len(s.attempts)} attempts (attempt {best['attempt']}); the critic was not "
                             f"fully satisfied: {best['critique']['revision_instruction']}")
            else:
                s.message = "Every generated image failed the safety check, so nothing is shown."
    except UnsafePrompt as e:
        s.route, s.message = "refuse", f"I stopped before generating: the {e}, which looks like an unsafe instruction."
    except Exception as e:
        agent = s.trace[-1]["agent"] if s.trace else "setup"
        s.message = f"Sorry, the {agent} step failed ({type(e).__name__}: {e}). Please try again or change the request."
    finally:
        s.seconds = round(time.perf_counter() - t0, 2)
        if save:
            save_trace(s, run_id)
    return s


def save_trace(s, run_id):
    TRACES.mkdir(parents=True, exist_ok=True)
    final = next((a["generation"].get("output_image") for a in s.attempts if a["image"] is s.final_image), None)
    trace = {"run_id": run_id, "request": s.request, "image": s.image_path, "use_critic": s.use_critic,
             "route": s.route, "message": s.message, "final_image": final, "seconds": s.seconds,
             "revisions": max(len(s.attempts) - 1, 0), "steps": s.trace}
    s.trace_path = TRACES / f"{run_id}.json"
    s.trace_path.write_text(json.dumps(trace, indent=2, default=str))
