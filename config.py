"""Central configuration for the Unofficial Guide RAG pipeline.

All tunable knobs live here so ingest.py, retriever.py, generator.py and
app.py share one source of truth. The values mirror the decisions documented
in planning.md (Chunking Strategy + Retrieval Approach).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"          # persisted vector store (gitignored)
COLLECTION_NAME = "unofficial_guide_cities"

# --- Chunking (see planning.md > Chunking Strategy) ----------------------
# Each source file is a short, self-contained description of one city, so we
# keep one document per chunk with no overlap. CHUNK_SIZE is only a guard for
# any unusually long source (e.g. the Wikipedia entry).
CHUNK_SIZE = 512      # characters
CHUNK_OVERLAP = 0     # characters

# --- Retrieval (see planning.md > Retrieval Approach) --------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, local, fast, good on short text
TOP_K = 3                              # chunks retrieved per query

# --- Generation ----------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.2      # low: factual lookups, not creative writing
