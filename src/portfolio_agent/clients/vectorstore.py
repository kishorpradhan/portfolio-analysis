from typing import Optional

from portfolio_agent.config.settings import get_settings
from portfolio_agent.utils.vectorstore import PineconeVectorStore

_vector_store: Optional[PineconeVectorStore] = None


def get_vector_store(dimension: int) -> PineconeVectorStore:
    global _vector_store
    settings = get_settings()
    if _vector_store is None:
        if not settings.pinecone_api_key:
            raise EnvironmentError("PINECONE_API_KEY is not configured.")
        _vector_store = PineconeVectorStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            dimension=dimension,
        )
    return _vector_store
