import pandas as pd
from src.config import ANNOTATIONS_PATH, LAYOUTS_PATH, PAGES_PATH, SUBSET_SIZE


def load_subset() -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """
    Returns (doc_metadata_list, pages_df, layouts_df) for the document subset.

    doc_metadata_list entries match the Tier 3 schema: doc_name, domain,
    page_range, layout_range, num_pages, questions.

    Subset selection: one document per domain (cycling through 10 domains),
    picking the first document encountered for each domain up to SUBSET_SIZE.
    """
    annotations = pd.read_json(ANNOTATIONS_PATH, lines=True)

    selected_docs: list[dict] = []
    seen_domains: set[str] = set()
    for _, row in annotations.iterrows():
        if len(selected_docs) >= SUBSET_SIZE:
            break
        domain = row["domain"]
        if domain not in seen_domains:
            seen_domains.add(domain)
            # Parquet files use doc names without the .pdf extension
            doc_name = row["doc_name"].removesuffix(".pdf")
            selected_docs.append(
                {
                    "doc_name": doc_name,
                    "domain": domain,
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
    layouts_df = layouts_df[layouts_df["doc_name"].isin(selected_names)].reset_index(
        drop=True
    )

    return selected_docs, pages_df, layouts_df
