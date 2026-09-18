"""Shared thin helpers for scripts: real service wiring from env/config."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def build_real_service(settings):
    """Wire production backends from env. Raises SystemExit when missing.

    Chat goes through the configured provider; embeddings are always
    local and free (SentenceTransformerEmbedder from settings).
    """
    from longmem.env import load_dotenv
    from longmem.llm import (
        GROQ_BASE_URL,
        LLMReranker,
        LlamaCppLLM,
        OPENROUTER_BASE_URL,
        OpenAICompatLLM,
        SentenceTransformerEmbedder,
    )
    from longmem.service import MemoryService
    from longmem.store import PostgresDurableStore, RedisWorkingStore

    load_dotenv(REPO_ROOT / ".env")

    dsn = os.environ.get(settings.postgres.dsn_env, "")
    if not dsn:
        raise SystemExit(f"missing env var {settings.postgres.dsn_env}")
    redis_url = os.environ.get(settings.redis.url_env, "")
    if not redis_url:
        raise SystemExit(f"missing env var {settings.redis.url_env}")

    models = settings.models
    if models.provider == "local":
        llm = LlamaCppLLM(model_path=models.local_model)
    else:
        if models.provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                raise SystemExit("missing env var OPENROUTER_API_KEY (paste it into .env)")
            base_url = os.environ.get("OPENAI_BASE_URL") or OPENROUTER_BASE_URL
            extra_body = {"reasoning": {"enabled": True}}
        elif models.provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                raise SystemExit("missing env var GROQ_API_KEY (paste it into .env)")
            base_url = os.environ.get("GROQ_BASE_URL") or GROQ_BASE_URL
            extra_body = None
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise SystemExit("missing env var OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL") or None
            extra_body = None
        llm = OpenAICompatLLM(
            api_key=api_key,
            base_url=base_url,
            judge_model=models.judge,
            classifier_model=models.classifier,
            understand_model=models.understand,
            extra_body=extra_body,
        )

    embedder = SentenceTransformerEmbedder(
        model=models.embed, dim=models.embedding_dim
    )
    # Chat-model rerank; the config rerank key stays reserved for a
    # future cross-encoder backend.
    reranker = LLMReranker(llm, model=models.understand)
    durable = PostgresDurableStore(
        dsn, settings.models.embedding_dim, embedder=embedder
    )
    working = RedisWorkingStore(redis_url)
    return MemoryService(durable, working, llm, reranker, settings, embedder)
