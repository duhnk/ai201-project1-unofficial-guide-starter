# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->
Cities and simple facts about them
---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | dr5hn Countries-States-Cities Database| Toronto |https://github.com/dr5hn/countries-states-cities-database |
| 2 | SimpleMaps Free World Cities Database| Tokyo| https://simplemaps.com/data/world-cities|
| 3 | Kaggle Countries States Cities Dataset| Paris| https://www.kaggle.com/datasets/max-mind/world-cities-database|
| 4 | GitHub Datasets World Cities|Sydney | https://github.com/datasets/world-cities|
| 5 | JSONLint World Countries Dataset| Brazil| https://jsonlint.com/datasets/countries|
| 6 | Google Canonical Countries CSV| The United Kingdom|https://developers.google.com/public-data/docs/canonical/countries_csv |
| 7 | GeoNames REST Web Services| Cairo| https://www.geonames.org/export/web-services.html|
| 8 | Back4app Database Hub|Nairobi | https://www.back4app.com/database/back4app/list-of-all-continents-countries-cities|
| 9 | GeoDB Cities API|Mumbai | https://github.com/topics/cities-database|
| 10 | Wikipedia & Wikidata Text Dumps| Berlin| https://en.wikipedia.org/wiki/Berlin|

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** One chunk = one whole document (~512 characters max). Each source file is a single, self-contained description of one city (header + metadata line + a 2–4 sentence paragraph, ~300–400 chars total), so I keep the entire document together as one chunk rather than splitting it.

**Overlap:** 0 characters. Because every document is short and each chunk already holds a complete, standalone fact about one city, there is no key fact that spans a chunk boundary — so overlap would only duplicate content and waste embedding space.

**Reasoning:** My documents are short factual blurbs, not long guides, so the chunking question is really "split or don't split." A fixed small character count (e.g. 200 chars) would cut a single city's description in half — separating "Tokyo is the capital of Japan" from "...population of approximately 37,732,000" — and neither half would answer a population query on its own. A whole-document chunk keeps the city name, its attributes, and its source/metadata together so every retrieved chunk is self-sufficient and carries attribution. If chunks were too small I'd see retrieval return fragments missing the city name or the answer value; too large isn't a risk here since no document exceeds a few sentences. The ~512-char ceiling is just a guard for any longer source (e.g. the Wikipedia/Berlin entry) so it still fits comfortably in one chunk.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers`. It produces 384-dimensional embeddings, is fast and lightweight enough to run locally with no API cost, and performs well on the short, single-sentence-style text that makes up my corpus.

**Top-k:** 3. Each document is a complete fact about one city, so the answer to a city-specific question (e.g. "What is Tokyo's population?") usually lives in a single chunk. Retrieving 3 gives the LLM the best match plus two backups in case the query is phrased ambiguously, without flooding the prompt with unrelated cities. Too few (k=1) risks missing the right document if the top match is wrong; too many (k=8+) pulls in irrelevant cities that can distract the model and dilute the context.

Semantic search works here even without shared words because the embedding model maps meaning, not exact tokens — so a query like "How many people live in Tokyo?" lands near a chunk that says "population of approximately 37,732,000," even though "how many people" never appears in the text.

**Production tradeoff reflection:** If cost weren't a constraint and this served real users, I'd weigh: (1) **multilingual support** — city/place data is global, so a multilingual model (e.g. `paraphrase-multilingual-MiniLM-L12-v2` or a hosted OpenAI/Cohere embedding) would let users query in their own language and match place names with non-Latin scripts; (2) **accuracy on domain-specific text** — a larger model like `all-mpnet-base-v2` (768-dim) gives better retrieval accuracy on subtle distinctions (capital vs. primary city, ISO codes); (3) **context length** — not a real concern for these short docs, but matters if I later ingest full Wikipedia articles; and (4) **latency** — bigger models and hosted APIs add per-query latency, so for an interactive app I'd balance accuracy against keeping responses snappy. For this project, MiniLM-L6-v2 is the right local default; mpnet or a hosted multilingual model would be the upgrade path.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1.

2.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
