"""
Evaluation metrics for retrieved chunks and generated answers.

page_recall_at_k:      checks if any ground-truth page_id is in the top-k retrieved chunks.
layout_recall_at_k:    expands layout_ids to look up bboxes from the parquet, checks IoU.
citation_grounding:    verifies that the chunks Claude *cited* (not just retrieved) include
                       the ground-truth page and layout element.
answer scoring (3 methods):
  - token_f1:    word-overlap F1 between generated and ground truth (standard QA metric)
  - fuzzy_match: difflib SequenceMatcher ratio (character-level soft similarity, 0–1)
  - llm_judge:   Claude rates whether the generated answer is semantically correct (0 or 1)
"""

import ast
import csv
import json
import re
from datetime import datetime
from difflib import SequenceMatcher

import anthropic
import pandas as pd

from src.agent.graph_react import graph
from src.agent.state import RAGState
from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, LAYOUTS_PATH


def _build_layout_bbox_lookup(doc_names: list[str]) -> dict[tuple[str, int], list[float]]:
    """Build a (doc_name, layout_id) → bbox lookup from the layouts parquet for IoU scoring."""
    df = pd.read_parquet(LAYOUTS_PATH, columns=["doc_name", "layout_id", "page_id", "bbox"])
    df = df[df["doc_name"].isin(doc_names)]
    return {
        (row["doc_name"], int(row["layout_id"])): list(row["bbox"])
        for _, row in df.iterrows()
        if row["bbox"] is not None
    }


def _parse_question_type(raw: str | list) -> str:
    """Normalize the question type field to a single primary label."""
    if isinstance(raw, list):
        types = raw
    elif isinstance(raw, str) and raw.startswith("["):
        try:
            types = ast.literal_eval(raw)
        except Exception:
            return raw
    else:
        return str(raw) if raw else "unknown"
    return types[0] if types else "unknown"


def _iou(b1: list[float], b2: list[float]) -> float:
    """Compute intersection-over-union for two [x1, y1, x2, y2] bounding boxes."""
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
    """Lowercase and strip punctuation for token-level comparison."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _tokenize(text: str) -> list[str]:
    """Normalise and split text into whitespace-delimited tokens."""
    return _normalize(text).split()


# ── Retrieval metrics ──────────────────────────────────────────────────────────

def page_recall_at_k(
    retrieved_chunks: list[dict],
    ground_truth_page_ids: list[int],
    k: int,
) -> bool:
    """Return True if any ground-truth page_id appears in the top-k retrieved chunks."""
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
    """Return True if any top-k chunk bbox overlaps a ground-truth layout bbox at IoU >= threshold."""
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


def citation_grounding(
    cited_chunk_ids: list[str],
    retrieved_chunks: list[dict],
    ground_truth_page_ids: list[int],
    ground_truth_layout_mappings: list[dict],
    doc_name: str,
    bbox_lookup: dict[tuple[str, int], list[float]],
) -> tuple[bool, bool]:
    """
    Checks whether the chunks Claude actually cited (not just retrieved) contain
    the ground-truth evidence.

    Returns (page_grounded, layout_grounded):
      page_grounded:   at least one cited chunk is on a ground-truth page
      layout_grounded: at least one cited chunk overlaps the ground-truth layout bbox
    """
    chunk_by_id = {c["chunk_id"]: c for c in retrieved_chunks}
    cited = [chunk_by_id[cid] for cid in cited_chunk_ids if cid in chunk_by_id]
    if not cited:
        return False, False
    page_grounded = page_recall_at_k(cited, ground_truth_page_ids, k=len(cited))
    layout_grounded = layout_recall_at_k(
        cited, ground_truth_layout_mappings, doc_name, bbox_lookup, k=len(cited)
    )
    return page_grounded, layout_grounded


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
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            return 0
        reply = response.content[0].text.strip().upper()
        return 1 if reply.startswith("YES") else 0
    except Exception:
        return 0


# ── Main evaluation loop ───────────────────────────────────────────────────────

def run_evaluation(
    doc_metadata_list: list[dict],
    output_csv: str = "eval_results.csv",
    output_summary: str = "eval_summary.json",
) -> dict:
    """Run the full RAG agent over all questions and write per-question and aggregate metrics."""
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
    cite_page_hits: list[bool] = []
    cite_layout_hits: list[bool] = []

    # per-modality accumulators: {type: {metric: [values]}}
    modality_stats: dict[str, dict[str, list]] = {}

    total_questions = sum(len(d["questions"]) for d in doc_metadata_list)
    q_idx = 0

    for doc in doc_metadata_list:
        doc_name = doc["doc_name"]
        for qa in doc["questions"]:
            q_idx += 1
            question = qa["Q"]
            gt_answer = qa["A"]
            gt_page_ids = qa.get("page_id", [])
            gt_layout_mappings = qa.get("layout_mapping", [])
            q_type = _parse_question_type(qa.get("type", "unknown"))

            print(f"  [{q_idx}/{total_questions}] {question[:75]}...")
            initial_state: RAGState = {
                "question": question,
                # Shared outputs
                "answer": "",
                "cited_chunk_ids": [],
                "retrieved_chunks": [],
                # Planner-Executor fields
                "modality": "",
                "granularity": "layout",
                "rewritten_query": "",
                "top_k": 10,
                "candidate_chunks": [],
                "reranked_chunks": [],
                "is_sufficient": False,
                "insufficiency_reason": "",
                "retry_count": 0,
                "is_validated": False,
                "validation_feedback": "",
                "validation_attempts": 0,
                # ReAct fields
                "messages": [],
            }
            result = graph.invoke(initial_state)
            chunks = result.get("retrieved_chunks", [])
            generated = result.get("answer", "")
            cited_ids = result.get("cited_chunk_ids", [])

            p_hit = page_recall_at_k(chunks, gt_page_ids, k=5)
            l_hit = layout_recall_at_k(chunks, gt_layout_mappings, doc_name, bbox_lookup, k=5)
            f1 = token_f1(generated, gt_answer)
            fuzz = fuzzy_match(generated, gt_answer)
            judge = llm_judge(question, generated, gt_answer, anthropic_client)
            contains = exact_containment(generated, gt_answer)
            cite_page, cite_layout = citation_grounding(
                cited_ids, chunks, gt_page_ids, gt_layout_mappings, doc_name, bbox_lookup
            )

            page_hits.append(p_hit)
            layout_hits.append(l_hit)
            f1_scores.append(f1)
            fuzzy_scores.append(fuzz)
            llm_scores.append(judge)
            containment_hits.append(contains)
            cite_page_hits.append(cite_page)
            cite_layout_hits.append(cite_layout)

            # per-modality accumulation
            if q_type not in modality_stats:
                modality_stats[q_type] = {
                    "page_recall": [], "layout_recall": [],
                    "llm_judge": [], "cite_page": [], "cite_layout": [],
                }
            modality_stats[q_type]["page_recall"].append(p_hit)
            modality_stats[q_type]["layout_recall"].append(l_hit)
            modality_stats[q_type]["llm_judge"].append(judge)
            modality_stats[q_type]["cite_page"].append(cite_page)
            modality_stats[q_type]["cite_layout"].append(cite_layout)

            rows.append(
                {
                    "doc_name": doc_name,
                    "question_type": q_type,
                    "question": question,
                    "gt_answer": str(gt_answer),
                    "generated_answer": generated,
                    "page_recall@5": p_hit,
                    "layout_recall@5": l_hit,
                    "cited_page_grounded": cite_page,
                    "cited_layout_grounded": cite_layout,
                    "token_f1": round(f1, 4),
                    "fuzzy_match": round(fuzz, 4),
                    "llm_judge": judge,
                    "exact_containment": contains,
                    "cited_chunks": "|".join(cited_ids),
                }
            )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    def _avg(vals: list) -> float:
        """Return the mean of a list, or 0.0 if empty."""
        return sum(vals) / len(vals) if vals else 0.0

    modality_breakdown = {
        qtype: {
            "n": len(stats["page_recall"]),
            "page_recall@5": round(_avg(stats["page_recall"]), 4),
            "layout_recall@5": round(_avg(stats["layout_recall"]), 4),
            "llm_judge": round(_avg(stats["llm_judge"]), 4),
            "cited_page_grounded": round(_avg(stats["cite_page"]), 4),
            "cited_layout_grounded": round(_avg(stats["cite_layout"]), 4),
        }
        for qtype, stats in sorted(modality_stats.items())
    }

    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_questions": len(rows),
        "num_documents": len(doc_metadata_list),
        "page_recall@5": _avg(page_hits),
        "layout_recall@5": _avg(layout_hits),
        "cited_page_grounded": _avg(cite_page_hits),
        "cited_layout_grounded": _avg(cite_layout_hits),
        "token_f1": _avg(f1_scores),
        "fuzzy_match": _avg(fuzzy_scores),
        "llm_judge": _avg(llm_scores),
        "exact_containment": _avg(containment_hits),
        "modality_breakdown": modality_breakdown,
        "docs_evaluated": doc_names,
    }

    with open(output_summary, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
