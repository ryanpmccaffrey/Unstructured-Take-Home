"""
Ingestion pipeline: load subset → chunk → embed → upsert to Chroma.
Run with: make ingest
"""

from src.ingest.chunker import chunk_document
from src.ingest.indexer import build_index
from src.ingest.loader import load_subset


def main() -> None:
    print("Loading dataset subset...")
    doc_metadata_list, pages_df, layouts_df = load_subset()
    print(f"Selected {len(doc_metadata_list)} documents:")
    for d in doc_metadata_list:
        print(f"  {d['doc_name']} ({d['domain']}, {d['num_pages']} pages)")

    all_tier1 = []
    all_tier2 = []
    for doc in doc_metadata_list:
        print(f"Chunking {doc['doc_name']}...")
        t1, t2 = chunk_document(doc, pages_df, layouts_df)
        all_tier1.extend(t1)
        all_tier2.extend(t2)
        print(f"  → {len(t1)} layout chunks, {len(t2)} page chunks")

    print(f"\nTotal: {len(all_tier1)} layout chunks, {len(all_tier2)} page chunks")
    print("Embedding and indexing (this may take a minute)...")
    build_index(all_tier1, all_tier2)
    print("Done.")


if __name__ == "__main__":
    main()
