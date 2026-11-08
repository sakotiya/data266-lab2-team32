"""Main pipeline: read Questions.csv, search, answer, write submission.csv."""
import pandas as pd
from pathlib import Path

from .common import DATA, METRICS, OUTPUTS
from .search import Searcher
from .answer import generate_answer, format_for_submission, clear_memory, configure_gemini


def run_pipeline(questions_path: Path = None, output_path: Path = None, api_key: str = None):
    """
    Process all questions and generate submission.csv.

    Args:
        questions_path: Path to Questions.csv (default: data/Questions.csv)
        output_path: Path for submission.csv (default: outputs/submission.csv)
        api_key: Gemini API key (or set GOOGLE_API_KEY env var)
    """
    questions_path = questions_path or DATA / "Questions.csv"
    output_path = output_path or OUTPUTS / "submission.csv"

    # Configure API
    configure_gemini(api_key)

    # Load questions
    questions = pd.read_csv(questions_path)
    print(f"Loaded {len(questions)} questions")

    # Initialize searcher
    searcher = Searcher()

    # Clear conversation memory
    clear_memory()

    # Process each question
    results = []
    for idx, row in questions.iterrows():
        qid = row["question_id"]
        cid = row["conversation_id"]
        question = row["question"]

        print(f"[{idx+1}/{len(questions)}] {qid}: {question[:50]}...")

        # Search for relevant content
        search_results = searcher.search(question)

        # Generate answer
        answer = generate_answer(question, search_results, conversation_id=cid)
        formatted = format_for_submission(answer)

        results.append({
            "id": row["id"],
            "conversation_id": cid,
            "question_id": qid,
            "answer": formatted["answer"],
            "figure_references": formatted["figure_references"],
            "citations": formatted["citations"],
            "status": formatted["status"]
        })

    # Write submission
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nSaved submission to {output_path}")

    return df


def validate_submission(submission_path: Path = None, questions_path: Path = None):
    """Run the validation script on submission."""
    import subprocess
    submission_path = submission_path or OUTPUTS / "submission.csv"
    questions_path = questions_path or DATA / "Questions.csv"
    validator = DATA / "validate_submission.py"

    result = subprocess.run(
        ["python", str(validator), str(submission_path), str(questions_path)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("VALIDATION FAILED:")
        print(result.stderr)
    return result.returncode == 0


if __name__ == "__main__":
    import sys
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    df = run_pipeline(api_key=api_key)
    print("\nValidating submission...")
    validate_submission()
