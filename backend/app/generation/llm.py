"""Thin wrapper around the Groq API, used by both the agent loop
(sufficiency checks, query reformulation) and final answer generation
(Phase 5). Centralizing this in one file means swapping models or
providers later only touches this one place.
"""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client: Groq | None = None

DEFAULT_MODEL = "openai/gpt-oss-20b"


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key from console.groq.com "
                "and put it in a .env file in the backend/ folder."
            )
        _client = Groq(api_key=api_key)
    return _client


def chat(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> str:
    """Sends a chat completion request, returns the response text.
    temperature=0 by default -- this is retrieval-grounded Q&A and
    structured yes/no checks, where consistency matters more than
    creativity."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""