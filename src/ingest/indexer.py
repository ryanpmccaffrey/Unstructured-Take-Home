import chromadb

from src.config import CHROMA_PATH, COLLECTION_LAYOUTS, COLLECTION_PAGES
from src.embedder import get_embedder
from src.ingest.chunker import Chunk

_BATCH_SIZE = 64


def _upsert_chunks(
    collection: chromadb.Collection,
    chunks: list[dict],
    text_key: str = "text_for_embedding",
) -> int:
    """Embed chunks in batches and upsert embeddings + metadata into the collection."""
    embedder = get_embedder()
    ids = [c["chunk_id"] for c in chunks]
    texts = [c[text_key] or " " for c in chunks]
    metadatas = [{k: v for k, v in c.items() if k != text_key} for c in chunks]

    total = 0
    for i in range(0, len(chunks), _BATCH_SIZE):
        batch_ids = ids[i : i + _BATCH_SIZE]
        batch_texts = texts[i : i + _BATCH_SIZE]
        batch_meta = metadatas[i : i + _BATCH_SIZE]
        batch_embeds = embedder.embed_documents(batch_texts)
        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeds,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        total += len(batch_ids)
    return total


def build_index(tier1_chunks: list[Chunk], tier2_chunks: list[dict]) -> None:
    """Initialise ChromaDB collections and upsert all Tier 1 and Tier 2 chunks."""
    print(f"Loading BGE embedder...")
    get_embedder()  # warm up

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)

    # Collections store raw embeddings (no built-in embedding function)
    layouts_col = chroma.get_or_create_collection(COLLECTION_LAYOUTS)
    pages_col = chroma.get_or_create_collection(COLLECTION_PAGES)

    tier1_dicts = [c.to_dict() for c in tier1_chunks]

    n_layouts = _upsert_chunks(layouts_col, tier1_dicts)
    n_pages = _upsert_chunks(pages_col, tier2_chunks)

    print(f"Indexed {n_layouts} layout chunks and {n_pages} page chunks.")
