from typing import Optional

from langchain_openai import OpenAIEmbeddings

from portfolio_agent.config.settings import get_settings

_embeddings_client: Optional[OpenAIEmbeddings] = None


def get_embedding_client() -> OpenAIEmbeddings:
    global _embeddings_client
    if _embeddings_client is None:
        settings = get_settings()
        _embeddings_client = OpenAIEmbeddings(model=settings.embedding_model)
    return _embeddings_client
