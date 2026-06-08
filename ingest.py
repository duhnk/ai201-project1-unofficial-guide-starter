"""Document ingestion + chunking + embedding + vector store.

Pipeline stages 1-3 from planning.md:
    Document Ingestion -> Chunking -> Embedding + Vector Store

Run this once (or whenever documents/ changes) to (re)build the ChromaDB
collection that retriever.py reads from:

    python ingest.py
"""

import json
import shutil

import chromadb
from sentence_transformers import SentenceTransformer

import config


def parse_document(path):
    """Parse one source file into its parts.

    Expected format (see documents/):
        Source: <source name>

        Metadata: {<json>}

        <body paragraph(s) of facts>

    Returns a dict with the source name, parsed metadata, and the body text.
    Lines that don't match are tolerated: anything that isn't a Source/Metadata
    label is treated as body text.
    """
    raw = path.read_text(encoding="utf-8").strip()

    source = path.stem
    metadata = {}
    body_lines = []

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("Source:"):
            source = stripped[len("Source:"):].strip()
        elif stripped.startswith("Metadata:"):
            try:
                metadata = json.loads(stripped[len("Metadata:"):].strip())
            except json.JSONDecodeError:
                metadata = {}
        elif stripped:
            body_lines.append(stripped)

    body = " ".join(body_lines)
    return {"source": source, "metadata": metadata, "body": body, "file": path.name}


def chunk_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Split text into chunks of at most `size` chars with `overlap` between them.

    For our corpus this returns a single chunk per document (each doc is shorter
    than `size`). The sliding-window logic is a guard for any longer source so a
    big document still gets stored rather than truncated.
    """
    text = text.strip()
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


def load_documents():
    """Read every source file in documents/ and yield chunk records."""
    records = []
    files = sorted(
        p for p in config.DOCUMENTS_DIR.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    if not files:
        raise SystemExit(f"No documents found in {config.DOCUMENTS_DIR}")

    for path in files:
        doc = parse_document(path)
        if not doc["body"]:
            print(f"  ! skipping {path.name}: no body text")
            continue

        chunks = chunk_text(doc["body"])
        for i, chunk in enumerate(chunks):
            # ChromaDB metadata values must be str/int/float/bool, so flatten
            # the parsed JSON metadata into stringified key/values.
            flat_meta = {"source": doc["source"], "file": doc["file"]}
            for k, v in doc["metadata"].items():
                flat_meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)

            records.append({
                "id": f"{path.stem}__{i}",
                "text": chunk,
                "metadata": flat_meta,
            })
    return records


def main():
    print(f"Loading documents from {config.DOCUMENTS_DIR} ...")
    records = load_documents()
    print(f"  -> {len(records)} chunk(s) from the corpus")

    print(f"Embedding with {config.EMBEDDING_MODEL} ...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = model.encode(
        [r["text"] for r in records],
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    # Rebuild from scratch so re-running is idempotent.
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[r["id"] for r in records],
        embeddings=embeddings,
        documents=[r["text"] for r in records],
        metadatas=[r["metadata"] for r in records],
    )

    print(f"Stored {collection.count()} chunk(s) in collection "
          f"'{config.COLLECTION_NAME}' at {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
