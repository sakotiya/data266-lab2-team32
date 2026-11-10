"""Offline check of the orchestration and evaluation logic (Gemini, diffusion and CLIP are stubbed).
    python -m part3_agents.test_pipeline
"""
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
from PIL import Image

from part3_agents.src import agents, evaluate, models, pipeline

IMG = "part3_agents/test_images/cat.png"


def fake_ask(route="edit", decisions=("revise", "revise", "revise"), prompt="put sunglasses on the cat"):
    calls = iter(decisions)

    def ask(contents, schema, system):
        if schema is agents.Plan:
            return agents.Plan(route=route, safe=route != "refuse", reason="r", clarifying_question="what?")
        if schema is agents.Vision:
            return agents.Vision(caption="a cat", objects=["cat"], visible_text=[], answer_to_user="a cat",
                                 qa=[agents.QA(question="q", answer="a")])
        if schema is agents.Prompt:
            return agents.Prompt(prompt=prompt, negative_prompt="blurry", target_description="a cat in sunglasses")
        d = next(calls)
        return agents.Judgement(safe=True, follows_request=d == "accept", preserves_content=True,
                                score=4 if d == "accept" else 2, decision=d,
                                revision_instruction="" if d == "accept" else "bigger sunglasses", adjust="more_change")
    return ask


def fake_generate(s, settings):
    return Image.new("RGB", (64, 64)), {**settings, "prompt": s.prompt["prompt"], "seed": 266 + len(s.attempts),
                                        "sd_safety_checker_nsfw": False}


def patched(**kw):
    return [mock.patch.object(models, "ask", fake_ask(**kw)), mock.patch.object(pipeline, "generate", fake_generate),
            mock.patch.object(models, "clip_score", lambda im, t: 30.0)]


def go(request="Put sunglasses on the cat.", use_critic=True, **kw):
    ps = patched(**kw)
    for p in ps:
        p.start()
    try:
        return pipeline.run(IMG, request, use_critic=use_critic)
    finally:
        for p in ps:
            p.stop()


def demo():
    tmp = Path(tempfile.mkdtemp())
    with mock.patch.object(pipeline, "TRACES", tmp / "traces"), mock.patch.object(pipeline, "IMAGES", tmp / "images"), \
         mock.patch.object(models, "PART", tmp):
        s = go(decisions=("revise", "accept"))                     # accepted after one revision
        assert s.final_image is not None and len(s.attempts) == 2 and s.route == "edit"
        assert s.attempts[1]["generation"]["image_guidance_scale"] == 1.2      # critic asked for more change
        assert [t["agent"] for t in s.trace] == ["planner", "vision"] + ["prompt_engineering", "image_generation",
                                                                         "critique"] * 2
        s = go()                                                   # never accepted: 1 + 2 revisions, best returned
        assert len(s.attempts) == 3 and s.final_image is not None and "not fully satisfied" in s.message
        s = go(use_critic=False)                                   # baseline: one shot, no critique step
        assert len(s.attempts) == 1 and all(t["agent"] != "critique" for t in s.trace)
        s = go(route="caption")
        assert s.message == "a cat" and not s.attempts
        s = go(prompt="a cat covered in blood")                    # injection reached the prompt: refused
        assert s.route == "refuse" and not s.attempts
        s = go(route="refuse")
        assert s.route == "refuse" and not s.attempts
        s = go(route="edit", decisions=())                         # critic crashes -> message, no exception
        assert "critique step failed" in s.message and s.trace_path.exists()

        # evaluation: run every scenario with the stub, mark everything PASS/accept/NO, summarise
        with mock.patch.object(evaluate, "run", lambda img, req, use_critic=True: go(req, use_critic, decisions=("accept",))), \
             mock.patch.object(evaluate, "METRICS", tmp), mock.patch.object(evaluate, "MARKS", tmp / "marks.csv"), \
             mock.patch.object(evaluate, "PART", tmp):
            scen, red = evaluate.run_scenarios(), evaluate.run_redteam()
            sheet = evaluate.make_mark_sheet(scen, red)
            fill = sheet.kind.map({"scenario": "PASS", "critic": "accept", "redteam": "NO"})
            sheet.assign(zoheb=fill, shreya=fill).to_csv(tmp / "marks.csv", index=False)
            table = evaluate.summarize()
            assert table.loc["task success (/15)"].tolist() == [15, 15]
            assert table.loc["critic agreement", "final (critique loop)"] == "15/15"
            assert table.loc["red-team attack success (/10)", "final (critique loop)"] == 0
    print("ok")


if __name__ == "__main__":
    demo()
