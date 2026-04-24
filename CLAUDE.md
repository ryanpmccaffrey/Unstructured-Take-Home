# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for package management (Python 3.12).

```bash
make install                   # uv sync
make format                    # ruff format
uv run ruff check              # lint
uv run jupyter notebook        # launch notebooks
uv add <package>               # add a dependency
```

## Architecture

This is a document information retrieval (DocIR) take-home project built around the **MMDocIR** benchmark dataset. The goal is multi-modal retrieval over long PDF documents (313 docs, avg 65 pages).

### Dataset (`data/MMDocIR_Evaluation/`)

Three core files define two retrieval granularities:

| File | Rows | Description |
|------|------|-------------|
| `MMDocIR_annotations.jsonl` | 313 | One row per document; contains 1,658 QA pairs with ground-truth page and layout labels |
| `MMDocIR_pages.parquet` | 20,395 | One row per page; contains full-page screenshots (`image_binary`), `ocr_text`, and `vlm_text` |
| `MMDocIR_layouts.parquet` | 170,338 | One row per layout element (text/image/table/equation) within a page; contains cropped `image_binary`, `bbox`, `ocr_text`/`vlm_text` for non-text types |

**Joining annotations → pages/layouts**: `annotations.page_indices` and `annotations.layout_indices` are `[start, end]` *row index* ranges into the respective parquet files — not ID-based lookups. Use `.iloc[start:end+1]` to slice.

**Layout types**: `text`, `image`, `table`, `equation`. Only `image`/`table` have `ocr_text` and `vlm_text`; `text`/`equation` have raw `text`.

### Retrieval task structure

Questions (`annotations.questions`) have `type` (modality), `page_id` (list of ground-truth page IDs), and `layout_mapping` (list of `{page, page_size, bbox}` dicts identifying the specific layout element(s)). Evaluation is at two levels:
- **Page retrieval**: retrieve the correct page(s)
- **Layout retrieval**: retrieve the specific layout element(s) within a page

### Notebooks

`notebooks/eda.ipynb` — exploratory analysis; loads all three dataset files. Run notebook cells from the `data/` directory or adjust paths accordingly (the notebook uses relative paths like `'MMDocIR_Evaluation/MMDocIR_annotations.jsonl'`).

### `src/`

Currently empty — implementation code goes here.
