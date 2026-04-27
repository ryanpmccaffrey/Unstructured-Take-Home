import random

import pandas as pd

from src.config import ANNOTATIONS_PATH, LAYOUTS_PATH, PAGES_PATH, SUBSET_SIZE


EXCLUDED_DOCS: set[str] = {"news_combined"}

# When non-empty, load_subset uses exactly these documents instead of random sampling.
# Set to the original 5 baseline documents for reproducible comparisons.
INCLUDED_DOCS: set[str] = {
    "PH_2016.06.08_Economy-Final",
    "0e94b4197b10096b1f4c699701570fbf",
    "2310.05634v2",
    "honor_watch_gs_pro",
    "2024.ug.eprospectus",
}


def load_subset(seed: int = 42) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """
    Returns (doc_metadata_list, pages_df, layouts_df) for the document subset.

    If INCLUDED_DOCS is non-empty, those exact documents are loaded regardless of
    SUBSET_SIZE or the random seed — use this for reproducible baseline comparisons.

    Otherwise, selection is stratified random sampling: one document guaranteed per
    domain, remaining slots filled randomly. EXCLUDED_DOCS are always skipped.
    """
    annotations = pd.read_json(ANNOTATIONS_PATH, lines=True)

    if INCLUDED_DOCS:
        selected = [
            row for _, row in annotations.iterrows()
            if row["doc_name"].removesuffix(".pdf") in INCLUDED_DOCS
        ]
    else:
        rng = random.Random(seed)
        domain_groups: dict[str, list] = {}
        for _, row in annotations.iterrows():
            doc_name = row["doc_name"].removesuffix(".pdf")
            if doc_name in EXCLUDED_DOCS:
                continue
            domain_groups.setdefault(row["domain"], []).append(row)
        for rows in domain_groups.values():
            rng.shuffle(rows)

        guaranteed: list = []
        leftover: list = []
        for rows in domain_groups.values():
            guaranteed.append(rows[0])
            leftover.extend(rows[1:])
        rng.shuffle(leftover)
        selected = (guaranteed + leftover)[: SUBSET_SIZE]

    selected_docs: list[dict] = []
    for row in selected:
        doc_name = row["doc_name"].removesuffix(".pdf")
        selected_docs.append(
            {
                "doc_name": doc_name,
                "domain": row["domain"],
                "page_range": row["page_indices"],
                "layout_range": row["layout_indices"],
                "num_pages": row["page_indices"][1] - row["page_indices"][0] + 1,
                "questions": row["questions"],
            }
        )

    pages_df = pd.read_parquet(PAGES_PATH)
    layouts_df = pd.read_parquet(LAYOUTS_PATH)

    selected_names = {d["doc_name"] for d in selected_docs}
    pages_df = pages_df[pages_df["doc_name"].isin(selected_names)].reset_index(drop=True)
    layouts_df = layouts_df[layouts_df["doc_name"].isin(selected_names)].reset_index(drop=True)

    return selected_docs, pages_df, layouts_df
