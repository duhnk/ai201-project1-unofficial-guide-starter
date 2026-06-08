"""Generation stage from planning.md.

Takes the user's question plus the chunks returned by the retriever, builds a
grounded prompt, and asks a Groq-hosted LLM to answer using only that context.
Keeping the model "in the box" (answer from context, say so if unknown) is what
makes this RAG rather than a plain chatbot.
"""

from groq import Groq

import config

SYSTEM_PROMPT = (
    "You are the Unofficial Guide, a factual assistant that answers questions "
    "about cities using ONLY the provided context. Follow these rules:\n"
    "- Answer strictly from the context below. Do not use outside knowledge.\n"
    "- If the context does not contain the answer, say you don't have that "
    "information rather than guessing.\n"
    "- Be concise and cite the source name(s) you used in parentheses."
)


def build_prompt(query, chunks):
    """Assemble the context block + question into a user message."""
    if not chunks:
        context = "(no relevant documents were retrieved)"
    else:
        context = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
        )
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above."
    )


class Generator:
    def __init__(self):
        if not config.GROQ_API_KEY:
            raise SystemExit(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add "
                "your key (get one free at https://console.groq.com)."
            )
        self.client = Groq(api_key=config.GROQ_API_KEY)

    def generate(self, query, chunks):
        """Return the LLM's grounded answer as a string."""
        response = self.client.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=config.TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(query, chunks)},
            ],
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Manual smoke test: retrieve + generate for one hardcoded question.
    from retriever import Retriever

    query = "What is the population of Tokyo?"
    chunks = Retriever().retrieve(query)
    print(Generator().generate(query, chunks))
