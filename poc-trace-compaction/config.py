"""Yapılandırma — ortam değişkenlerini ve eşikleri yükler."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv yoksa ortam değişkenleri doğrudan okunur

# --- LLM (OpenAI-uyumlu endpoint) ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://your-llm-endpoint.example/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemma-4-31B-it")

# --- Trace compaction ---
TRACE_TOKEN_BUDGET = int(os.getenv("TRACE_TOKEN_BUDGET", "4000"))
TRACE_PROTECT_WINDOW = int(os.getenv("TRACE_PROTECT_WINDOW", "3"))

# Ajanın üzerinde çalıştığı gerçek repo
SAMPLE_REPO = Path(__file__).parent / "sample_repo"


def estimate_tokens(text: str) -> int:
    """Kaba token tahmini: ~4 karakter = 1 token. Deterministik ve bağımlılıksız."""
    return max(1, len(text) // 4)
