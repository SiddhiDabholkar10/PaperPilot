import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model assignments per node
# Change here to swap models globally
MODELS = {
    "decomposer":    "deepseek/deepseek-chat",   # V3 — reliable JSON
    "critic":        "deepseek/deepseek-chat",   # V3 — reliable JSON
    "reformulator":  "deepseek/deepseek-chat",   # V3 — was failing JSON
    "synthesizer":   "deepseek/deepseek-v4-flash", # keep — long text generation
    "verifier":      "deepseek/deepseek-chat",   # V3 — reliable JSON
}


def get_llm(
    node: str,
    temperature: float = 0.1,
    max_tokens: int = 1024
):
    """
    Get a configured LLM for a given agent node.
    Uses OpenRouter with DeepSeek V4 Flash.

    Args:
        node: one of decomposer, critic, reformulator, synthesizer, verifier
        temperature: sampling temperature
        max_tokens: max output tokens
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    model = MODELS.get(node, "deepseek/deepseek-v4-flash")

    return ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )