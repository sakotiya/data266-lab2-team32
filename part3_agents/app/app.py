"""Gradio demo: upload an image + request, see every agent's output.

    python -m part3_agents.app.app            # local
    demo.launch(share=True)                   # Colab/Kaggle (see notebooks/p3_agents.ipynb)
"""
import gradio as gr

from part3_agents.src.pipeline import run


def agent_output(s, name):
    outs = [t.get("output", {"error": t.get("error")}) for t in s.trace if t["agent"] == name]
    return outs[0] if len(outs) == 1 else outs or None


def handle(file, request, use_critic):
    if file is None:
        return None, "Please upload an image.", None, None, None, None, None, [], ""
    s = run(file, request, use_critic=use_critic)
    gallery = [(a["image"], f"attempt {a['attempt']}: "
                + (f"{a['critique']['decision']} (CLIP {a['critique']['clip_score']}, score {a['critique']['score']})"
                   if "critique" in a else "no critic"))
               for a in s.attempts if a.get("critique", {}).get("safe", True)]     # unsafe attempts are never shown
    return (s.final_image, f"**Route:** {s.route}  \n{s.message}", agent_output(s, "planner"),
            agent_output(s, "vision"), agent_output(s, "prompt_engineering"), agent_output(s, "image_generation"),
            agent_output(s, "critique"), gallery, f"{s.seconds}s · trace: {s.trace_path}")


with gr.Blocks(title="DATA 266 Part 3 - multi-agent image system") as demo:
    gr.Markdown("## Planner → Vision → Prompt → Generate → Critique (≤ 2 revisions)")
    with gr.Row():
        with gr.Column():
            # gr.File, not gr.Image, so broken files reach our own error handling instead of failing in the browser
            file = gr.File(label="Image", file_types=["image"], type="filepath")
            request = gr.Textbox(label="Request", placeholder="e.g. Turn this into a watercolor painting")
            use_critic = gr.Checkbox(value=True, label="Use critique loop")
            go = gr.Button("Run", variant="primary")
        with gr.Column():
            result = gr.Image(label="Final image", type="pil")
            message = gr.Markdown()
            timing = gr.Markdown()
    with gr.Row():
        plan = gr.JSON(label="1. Planner")
        vis = gr.JSON(label="2. Vision Understanding")
    with gr.Row():
        prompt = gr.JSON(label="3. Prompt Engineering")
        gen = gr.JSON(label="4. Image Generation")
        crit = gr.JSON(label="5. Critique")
    attempts = gr.Gallery(label="All attempts", columns=3, height=260)
    go.click(handle, [file, request, use_critic], [result, message, plan, vis, prompt, gen, crit, attempts, timing])

if __name__ == "__main__":
    demo.launch()
