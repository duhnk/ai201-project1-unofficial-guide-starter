"""Query interface for the Unofficial Guide (Milestone 5).

Ties the full pipeline together:
    query -> Retriever (top-k chunks) -> Generator (grounded answer)

Usage:
    python ingest.py     # build the vector store first (once)
    python app.py        # then start the interactive Q&A loop

Type a question and press Enter. Type 'quit', 'exit', or Ctrl-C to leave.
"""

from generator import Generator
from retriever import Retriever
import config


def answer_question(retriever, generator, query, show_sources=True):
    chunks = retriever.retrieve(query)
    answer = generator.generate(query, chunks)

    print(f"\n{answer}\n")
    if show_sources and chunks:
        print("Sources:")
        for c in chunks:
            print(f"  - {c['source']} (similarity {c['score']:.2f})")
    print()


def main():
    print("Loading models and vector store ...")
    retriever = Retriever()
    generator = Generator()

    print("\n" + "=" * 60)
    print("  The Unofficial Guide — ask me about cities")
    print(f"  (top-{config.TOP_K} retrieval, model: {config.GROQ_MODEL})")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        answer_question(retriever, generator, query)


if __name__ == "__main__":
    main()
