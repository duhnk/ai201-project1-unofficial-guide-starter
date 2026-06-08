"""Retrieval stage from planning.md.

Embeds the user's query with the same model used at ingest time, then asks
ChromaDB for the top-k nearest chunks (cosine similarity). Returns the chunk
text plus its source metadata so the generator can cite where each fact came
from.
"""

import chromadb
from sentence_transformers import SentenceTransformer

import config


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        try:
            self.collection = client.get_collection(config.COLLECTION_NAME)
        except Exception as exc:  # collection missing -> not ingested yet
            raise SystemExit(
                f"Collection '{config.COLLECTION_NAME}' not found. "
                f"Run `python ingest.py` first.\n  ({exc})"
            )

    def retrieve(self, query, k=config.TOP_K):
        """Return up to k chunks most relevant to `query`.

        Each result: {text, source, metadata, score} where score is cosine
        similarity in [0, 1] (1 = identical direction).
        """
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
        )

        hits = []
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        for text, meta, dist in zip(documents, metadatas, distances):
            hits.append({
                "text": text,
                "source": meta.get("source", "unknown"),
                "metadata": meta,
                "score": 1 - dist,   # cosine distance -> similarity
            })
        return hits


if __name__ == "__main__":
    # Quick manual smoke test.
    import sys

    query = " ".join(sys.argv[1:]) or "What is the population of Tokyo?"
    retriever = Retriever()
    print(f"Query: {query}\n")
    for i, hit in enumerate(retriever.retrieve(query), 1):
        print(f"[{i}] {hit['source']}  (similarity={hit['score']:.3f})")
        print(f"    {hit['text']}\n")
