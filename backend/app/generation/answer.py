"""Generates the final cited answer from retrieved chunks.

Uses a larger model than the agent loop's internal sufficiency check
(gpt-oss-120b vs gpt-oss-20b) since this is the actual user-facing
output -- worth spending a bit more here for quality.
"""

from app.generation.llm import chat
from app.retrieval.hybrid import RetrievalResult

ANSWER_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You are a code assistant that answers questions about a codebase "
    "using ONLY the provided code chunks as context. Rules:\n"
    "1. Answer only from the given chunks -- never use outside knowledge "
    "of common frameworks/libraries to fill gaps.\n"
    "2. Every claim must cite its source as [file_path:start_line-end_line], "
    "using the exact citation shown above each chunk.\n"
    "3. If the chunks don't contain enough information to answer, say so "
    "plainly rather than guessing.\n"
    "4. Be concise and technical -- this is for a developer reading code, "
    "not a general audience."
)


def _format_context(chunks: list[RetrievalResult]) -> str:
    parts = []
    for c in chunks:
        m = c.metadata
        citation = f"{m.get('file_path')}:{m.get('start_line')}-{m.get('end_line')}"
        parts.append(f"[{citation}]\n{c.text}")
    return "\n\n".join(parts)


def generate_answer(query: str, chunks: list[RetrievalResult]) -> str:
    if not chunks:
        return "I couldn't find any relevant code for this question."

    context = _format_context(chunks)
    user_prompt = f"Question: {query}\n\nRelevant code:\n{context}"

    return chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=ANSWER_MODEL,
        temperature=0.0,
    )