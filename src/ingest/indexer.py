import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.config import CHROMA_PATH, COLLECTION_LAYOUTS, COLLECTION_PAGES
from src.ingest.chunker import Chunk

_BATCH_SIZE = 128
_ef = DefaultEmbeddingFunction()


def _upsert_chunks(
    collection: chromadb.Collection,
    chunks: list[dict],
    text_key: str = "text_for_embedding",
) -> int:
    ids = [c["chunk_id"] for c in chunks]
    texts = [c[text_key] or " " for c in chunks]
    metadatas = [{k: v for k, v in c.items() if k != text_key} for c in chunks]

    total = 0
    for i in range(0, len(chunks), _BATCH_SIZE):
        batch_ids = ids[i : i + _BATCH_SIZE]
        batch_texts = texts[i : i + _BATCH_SIZE]
        batch_meta = metadatas[i : i + _BATCH_SIZE]
        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        total += len(batch_ids)
    return total


def build_index(tier1_chunks: list[Chunk], tier2_chunks: list[dict]) -> None:
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)

    # Collections use DefaultEmbeddingFunction (onnxruntime-backed all-MiniLM-L6-v2)
    layouts_col = chroma.get_or_create_collection(
        COLLECTION_LAYOUTS, embedding_function=_ef
    )
    pages_col = chroma.get_or_create_collection(
        COLLECTION_PAGES, embedding_function=_ef
    )

    tier1_dicts = [c.to_dict() for c in tier1_chunks]

    n_layouts = _upsert_chunks(layouts_col, tier1_dicts)
    n_pages = _upsert_chunks(pages_col, tier2_chunks)

    print(f"Indexed {n_layouts} layout chunks and {n_pages} page chunks.")
