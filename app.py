"""Query interface for the Unofficial Guide (Milestone 5).

Ties the full pipeline together behind a Gradio web UI:
    query -> Retriever (top-k chunks) -> Generator (grounded answer)

Usage:
    python ingest.py     # build the vector store first (once)
    python app.py        # then open http://localhost:7860

Type a question and press Enter (or click "Ask"). The answer is generated only
from the retrieved chunks, and the sources panel shows where each fact came
from along with its similarity score.
"""

import gradio as gr

from generator import Generator
from retriever import Retriever
import config

# Load the embedding model, vector store, and LLM client once at startup.
# These are expensive to construct, so we reuse them across every request.
print("Loading models and vector store ...")
retriever = Retriever()
generator = Generator()


def handle_query(question):
    """Run the full RAG pipeline for one question.

    Returns a (answer, sources) tuple of strings for the two output boxes.
    """
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""

    chunks = retriever.retrieve(question)
    answer = generator.generate(question, chunks)

    if chunks:
        sources = "\n".join(
            f"• {c['source']} (similarity {c['score']:.2f})" for c in chunks
        )
    else:
        sources = "No relevant documents were retrieved."

    return answer, sources


with gr.Blocks(title="The Unofficial Guide") as demo:
    gr.Markdown(
        "# The Unofficial Guide\n"
        "Ask me about cities. Answers are grounded only in the ingested "
        f"documents (top-{config.TOP_K} retrieval, model: `{config.GROQ_MODEL}`)."
    )

    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. What is the population of Tokyo?",
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    gr.Examples(
        examples=[
            "What is the population of Tokyo?",
            "What is the best time of year to visit?",
        ],
        inputs=inp,
    )

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
