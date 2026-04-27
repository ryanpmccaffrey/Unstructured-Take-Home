"""
Evaluation script: run the RAG agent over all questions for the indexed subset
and print aggregate metrics.
Run with: make eval
"""

import pathlib
from datetime import datetime

from src.eval.evaluate import run_evaluation
from src.ingest.loader import load_subset

RESULTS_DIR = pathlib.Path("results")


def main() -> None:
    print("Loading subset metadata for evaluation...")
    doc_metadata_list, _, _ = load_subset()
    total_q = sum(len(d["questions"]) for d in doc_metadata_list)
    print(f"Evaluating {total_q} questions across {len(doc_metadata_list)} documents...\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = str(RESULTS_DIR / f"eval_{ts}.csv")
    output_summary = str(RESULTS_DIR / f"eval_{ts}.json")

    metrics = run_evaluation(doc_metadata_list, output_csv=output_csv, output_summary=output_summary)

    print("\n── Evaluation Results ──────────────────")
    print(f"  Documents evaluated   : {metrics['num_documents']}")
    print(f"  Questions evaluated   : {metrics['num_questions']}")
    print(f"  Page Recall@5         : {metrics['page_recall@5']:.1%}")
    print(f"  Layout Recall@5       : {metrics['layout_recall@5']:.1%}")
    print(f"  Cited Page Grounded   : {metrics['cited_page_grounded']:.1%}")
    print(f"  Cited Layout Grounded : {metrics['cited_layout_grounded']:.1%}")
    print(f"  Token F1              : {metrics['token_f1']:.1%}")
    print(f"  Fuzzy Match           : {metrics['fuzzy_match']:.1%}")
    print(f"  LLM Judge             : {metrics['llm_judge']:.1%}")
    print(f"  Exact Containment     : {metrics['exact_containment']:.1%}")

    breakdown = metrics.get("modality_breakdown", {})
    if breakdown:
        print("\n── By Question Type ────────────────────")
        header = f"  {'Type':<32} {'N':>4}  {'PageR@5':>7}  {'LayR@5':>7}  {'CiteP':>6}  {'CiteL':>6}  {'Judge':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for qtype, b in breakdown.items():
            print(
                f"  {qtype:<32} {b['n']:>4}  {b['page_recall@5']:>7.1%}  "
                f"{b['layout_recall@5']:>7.1%}  {b['cited_page_grounded']:>6.1%}  "
                f"{b['cited_layout_grounded']:>6.1%}  {b['llm_judge']:>6.1%}"
            )

    print("────────────────────────────────────────")
    print(f"Results written to {output_csv} and {output_summary}")


if __name__ == "__main__":
    main()
