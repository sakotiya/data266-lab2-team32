"""
Shreya's simple version: calls the 5 agents in a straight line with no retry loop.
This demonstrates understanding of the agent pipeline.
"""
import time
from pathlib import Path

from src.agents import State, planner, vision, prompt_engineer, generate, critique, SETTINGS


def run_simple_pipeline(image_path: str, request: str) -> dict:
    """
    Run the 5 agents sequentially on one image.
    No retry loop - just one pass through all agents.

    Args:
        image_path: Path to input image
        request: User's text request

    Returns:
        dict with results from each agent
    """
    start = time.time()

    # Initialize state
    state = State(request=request, image_path=image_path, use_critic=True)
    results = {"request": request, "image_path": image_path}

    # 1. PLANNER - Understand request and check safety
    print("Agent 1: Planner...")
    plan = planner(state)
    results["plan"] = plan
    print(f"  Route: {state.route}")

    # Stop early if refused, error, clarify, or unsupported
    if state.route in ("refuse", "error", "clarify", "unsupported"):
        results["message"] = state.message
        results["final_image"] = None
        results["seconds"] = time.time() - start
        print(f"  Stopping: {state.message}")
        return results

    # 2. VISION - Describe and understand the image
    print("Agent 2: Vision...")
    vis = vision(state)
    results["vision"] = vis
    print(f"  Caption: {vis['caption'][:80]}...")

    # For caption route, we're done (no image generation)
    if state.route == "caption":
        results["message"] = state.message
        results["final_image"] = None
        results["seconds"] = time.time() - start
        return results

    # 3. PROMPT WRITER - Create generation prompt
    print("Agent 3: Prompt Writer...")
    prompt = prompt_engineer(state)
    results["prompt"] = prompt
    print(f"  Prompt: {prompt['prompt'][:80]}...")

    # 4. IMAGE MAKER - Generate the image
    print("Agent 4: Image Maker...")
    settings = SETTINGS[state.route].copy()
    image, gen_meta = generate(state, settings)
    results["generation"] = gen_meta
    print(f"  Generated image: {image.size}")

    # 5. CRITIC - Judge the result
    print("Agent 5: Critic...")
    judgement = critique(state, image, gen_meta)
    results["critique"] = judgement
    print(f"  Decision: {judgement['decision']}, Score: {judgement['score']}/5")

    # Store final result
    state.final_image = image
    results["final_image"] = image
    results["message"] = "Image generated successfully." if judgement["decision"] == "accept" else \
                         f"Generated but could be improved: {judgement['revision_instruction']}"
    results["seconds"] = time.time() - start

    print(f"\nCompleted in {results['seconds']:.1f}s")
    return results


def main():
    """Run simple pipeline on a test image."""
    import sys

    # Default test case
    test_images = Path(__file__).parent / "test_images"
    image_path = str(test_images / "cat.png")
    request = "Convert this to watercolor style"

    # Allow command line args
    if len(sys.argv) >= 3:
        image_path = sys.argv[1]
        request = sys.argv[2]

    print(f"Image: {image_path}")
    print(f"Request: {request}")
    print("-" * 50)

    results = run_simple_pipeline(image_path, request)

    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"  Route: {results.get('plan', {}).get('route')}")
    print(f"  Message: {results.get('message')}")
    if results.get("final_image"):
        # Save output
        output_path = Path(__file__).parent / "outputs" / "simple_output.png"
        output_path.parent.mkdir(exist_ok=True)
        results["final_image"].save(output_path)
        print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
