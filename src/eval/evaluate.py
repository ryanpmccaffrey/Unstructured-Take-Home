"""
Evaluation metrics for retrieved chunks and generated answers.

page_recall_at_k:   checks if any ground-truth page_id is in the top-k retrieved chunks.
layout_recall_at_k: expands layout_ids to look up bboxes from the parquet, checks IoU.
answer scoring (3 methods):
  - token_f1:    word-overlap F1 between generated and ground truth (standard QA metric)
  - fuzzy_match: difflib SequenceMatcher ratio (character-level soft similarity, 0–1)
  - llm_judge:   Claude rates whether the generated answer is semantically correct (0 or 1)
"""

import csv
import json
import re
from datetime import datetime
from difflib import SequenceMatcher

import anthropic
import pandas as pd

from src.agent.graph import graph
from src.agent.state import RAGState
from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, LAYOUTS_PATH


def _build_layout_bbox_lookup(doc_names: list[str]) -> dict[tuple[str, int], list[float]]:
    df = pd.read_parquet(LAYOUTS_PATH, columns=["doc_name", "layout_id", "page_id", "bbox"])
    df = df[df["doc_name"].isin(doc_names)]
    return {
        (row["doc_name"], int(row["layout_id"])): list(row["bbox"])
        for _, row in df.iterrows()
        if row["bbox"] is not None
    }


def _iou(b1: list[float], b2: list[float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter)


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


# ── Retrieval metrics ──────────────────────────────────────────────────────────

def page_recall_at_k(
    retrieved_chunks: list[dict],
    ground_truth_page_ids: list[int],
    k: int,
) -> bool:
    top_page_ids = {
        int(c["page_id"]) for c in retrieved_chunks[:k] if c.get("page_id") not in ("", None)
    }
    return bool(top_page_ids & set(ground_truth_page_ids))


def layout_recall_at_k(
    retrieved_chunks: list[dict],
    ground_truth_layout_mappings: list[dict],
    doc_name: str,
    bbox_lookup: dict[tuple[str, int], list[float]],
    k: int,
    iou_threshold: float = 0.5,
) -> bool:
    for chunk in retrieved_chunks[:k]:
        if chunk.get("doc_name") != doc_name:
            continue
        chunk_page_id = int(chunk.get("page_id", -1))
        layout_ids_raw = chunk.get("layout_ids", "[]")
        try:
            layout_ids = (
                json.loads(layout_ids_raw) if isinstance(layout_ids_raw, str) else layout_ids_raw
            )
        except (json.JSONDecodeError, TypeError):
            layout_ids = []
        for lid in layout_ids:
            chunk_bbox = bbox_lookup.get((doc_name, int(lid)))
            if not chunk_bbox or len(chunk_bbox) < 4:
                continue
            for gt in ground_truth_layout_mappings:
                if gt.get("page") != chunk_page_id:
                    continue
                gt_bbox = gt.get("bbox", [])
                if len(gt_bbox) == 4 and _iou(chunk_bbox, gt_bbox) >= iou_threshold:
                    return True
    return False


# ── Answer scoring ─────────────────────────────────────────────────────────────

def token_f1(generated: str, ground_truth: str | list) -> float:
    """Word-overlap F1, averaged over multiple ground-truth answers if provided."""
    if isinstance(ground_truth, list):
        scores = [token_f1(generated, str(a)) for a in ground_truth]
        return sum(scores) / len(scores) if scores else 0.0

    pred_tokens = _tokenize(generated)
    gt_tokens = _tokenize(str(ground_truth))
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    gt_counts: dict[str, int] = {}
    for t in gt_tokens:
        gt_counts[t] = gt_counts.get(t, 0) + 1

    overlap = sum(min(pred_counts.get(t, 0), gt_counts[t]) for t in gt_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def fuzzy_match(generated: str, ground_truth: str | list) -> float:
    """difflib SequenceMatcher ratio — character-level soft similarity (0–1)."""
    if isinstance(ground_truth, list):
        scores = [fuzzy_match(generated, str(a)) for a in ground_truth]
        return sum(scores) / len(scores) if scores else 0.0
    return SequenceMatcher(None, _normalize(generated), _normalize(str(ground_truth))).ratio()


def exact_containment(generated: str, ground_truth: str | list) -> bool:
    """Case-insensitive substring containment of ground truth in generated answer."""
    gen_lower = generated.lower()
    if isinstance(ground_truth, list):
        return all(str(a).lower() in gen_lower for a in ground_truth)
    return str(ground_truth).lower() in gen_lower


def llm_judge(
    question: str,
    generated: str,
    ground_truth: str | list,
    client: anthropic.Anthropic,
) -> int:
    """Ask Claude whether the generated answer is semantically correct. Returns 0 or 1."""
    gt_str = ", ".join(str(a) for a in ground_truth) if isinstance(ground_truth, list) else str(ground_truth)
    prompt = (
        f"Question: {question}\n"
        f"Ground truth answer: {gt_str}\n"
        f"Generated answer: {generated}\n\n"
        "Does the generated answer correctly answer the question, consistent with the ground truth? "
        "Reply with only YES or NO."
    )
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    reply = response.content[0].text.strip().upper()
    return 1 if reply.startswith("YES") else 0


# ── Main evaluation loop ───────────────────────────────────────────────────────

def run_evaluation(
    doc_metadata_list: list[dict],
    output_csv: str = "eval_results.csv",
    output_summary: str = "eval_summary.json",
) -> dict:
    doc_names = [d["doc_name"] for d in doc_metadata_list]
    print("Building layout bbox lookup from parquet...")
    bbox_lookup = _build_layout_bbox_lookup(doc_names)

    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    rows = []
    page_hits: list[bool] = []
    layout_hits: list[bool] = []
    f1_scores: list[float] = []
    fuzzy_scores: list[float] = []
    llm_scores: list[int] = []
    containment_hits: list[bool] = []

    for doc in doc_metadata_list:
        doc_name = doc["doc_name"]
        for qa in doc["questions"]:
            question = qa["Q"]
            gt_answer = qa["A"]
            gt_page_ids = qa.get("page_id", [])
            gt_layout_mappings = qa.get("layout_mapping", [])

            print(f"  Q: {question[:80]}...")
            initial_state: RAGState = {
                "question": question,
                "modality": "",
                "is_multi_hop": False,
                "retrieval_strategy": "hierarchical",
                "rewritten_query": "",
                "candidate_page_ids": [],
                "retrieved_chunks": [],
                "is_sufficient": False,
                "answer": "",
                "cited_chunk_ids": [],
                "retry_count": 0,
            }
            result = graph.invoke(initial_state)
            chunks = result.get("retrieved_chunks", [])
            generated = result.get("answer", "")

            p_hit = page_recall_at_k(chunks, gt_page_ids, k=5)
            l_hit = layout_recall_at_k(chunks, gt_layout_mappings, doc_name, bbox_lookup, k=5)
            f1 = token_f1(generated, gt_answer)
            fuzz = fuzzy_match(generated, gt_answer)
            judge = llm_judge(question, generated, gt_answer, anthropic_client)
            contains = exact_containment(generated, gt_answer)

            page_hits.append(p_hit)
            layout_hits.append(l_hit)
            f1_scores.append(f1)
            fuzzy_scores.append(fuzz)
            llm_scores.append(judge)
            containment_hits.append(contains)

            rows.append(
                {
                    "doc_name": doc_name,
                    "question": question,
                    "gt_answer": str(gt_answer),
                    "generated_answer": generated,
                    "page_recall@5": p_hit,
                    "layout_recall@5": l_hit,
                    "token_f1": round(f1, 4),
                    "fuzzy_match": round(fuzz, 4),
                    "llm_judge": judge,
                    "exact_containment": contains,
                    "cited_chunks": "|".join(result.get("cited_chunk_ids", [])),
                }
            )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    def _avg(vals: list) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_questions": len(rows),
        "num_documents": len(doc_metadata_list),
        "page_recall@5": _avg(page_hits),
        "layout_recall@5": _avg(layout_hits),
        "token_f1": _avg(f1_scores),
        "fuzzy_match": _avg(fuzzy_scores),
        "llm_judge": _avg(llm_scores),
        "exact_containment": _avg(containment_hits),
        "docs_evaluated": doc_names,
    }

    with open(output_summary, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
