"""
Evaluation script: run the RAG agent over all questions for the indexed subset
and print aggregate metrics.
Run with: make eval
"""

from src.eval.evaluate import run_evaluation
from src.ingest.loader import load_subset


def main() -> None:
    print("Loading subset metadata for evaluation...")
    doc_metadata_list, _, _ = load_subset()
    total_q = sum(len(d["questions"]) for d in doc_metadata_list)
    print(f"Evaluating {total_q} questions across {len(doc_metadata_list)} documents...\n")

    metrics = run_evaluation(doc_metadata_list, output_csv="eval_results.csv")

    print("\n── Evaluation Results ──────────────────")
    print(f"  Documents evaluated : {metrics['num_documents']}")
    print(f"  Questions evaluated : {metrics['num_questions']}")
    print(f"  Page Recall@5       : {metrics['page_recall@5']:.1%}")
    print(f"  Layout Recall@5     : {metrics['layout_recall@5']:.1%}")
    print(f"  Token F1            : {metrics['token_f1']:.1%}")
    print(f"  Fuzzy Match         : {metrics['fuzzy_match']:.1%}")
    print(f"  LLM Judge           : {metrics['llm_judge']:.1%}")
    print(f"  Exact Containment   : {metrics['exact_containment']:.1%}")
    print("────────────────────────────────────────")
    print("Results written to eval_results.csv and eval_summary.json")


if __name__ == "__main__":
    main()
