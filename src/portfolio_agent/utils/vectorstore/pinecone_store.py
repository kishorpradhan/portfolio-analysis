import hashlib
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional

from pinecone import Pinecone, ServerlessSpec

from portfolio_agent.utils.logger import get_logger

logger = get_logger(__name__)


class PineconeVectorStore:
    """
    Production-grade wrapper around Pinecone serverless vector DB
    with metadata support for dataset-driven code generation caching.
    """

    def __init__(self, api_key: str, index_name: str, dimension: int = 1536):
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension

        logger.info("Initializing Pinecone client")
        self.pc = Pinecone(api_key=self.api_key)

        if index_name not in self.pc.list_indexes().names():
            logger.warning(f"Index {index_name} not found. Creating a new one…")
            self.create_index()

        self.index = self.pc.Index(index_name)
        logger.info(f"Pinecone index loaded: {index_name}")

    # ----------------------------------------------------------------------
    # 1) Create Index
    # ----------------------------------------------------------------------
    def create_index(self):
        logger.info(f"Creating Pinecone index: {self.index_name}")
        self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info("Index created successfully.")

    # ----------------------------------------------------------------------
    # 2) Create a deterministic hash for caching
    # ----------------------------------------------------------------------
    def generate_query_hash(self, query: str, datasets: List[str]) -> str:
        raw = json.dumps({"query": query, "datasets": datasets}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ----------------------------------------------------------------------
    # 3) Upsert (Insert) new vector + metadata
    # ----------------------------------------------------------------------
    def upsert(
        self,
        embedding: List[float],
        code: str,
        query: str,
        datasets: List[str],
        model_name: str = "text-embedding-3-large",
        language: str = "python",
    ):
        query_hash = self.generate_query_hash(query, datasets)

        metadata = {
            "query": query,
            "datasets_used": datasets,
            "language": language,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "query_hash": query_hash,
            "code_length": len(code),
        }

        logger.info(f"Upserting code snippet with hash={query_hash[:12]}…")

        self.index.upsert(
            vectors=[
                {
                    "id": query_hash,
                    "values": embedding,
                    "metadata": metadata | {"python_code": code},
                }
            ]
        )

        logger.info("Upsert completed successfully.")
        return query_hash

    # ----------------------------------------------------------------------
    # 4) Semantic Search (retrieve code)
    # ----------------------------------------------------------------------
    def search(
        self,
        embedding: List[float],
        top_k: int = 3,
        include_values: bool = False,
        include_metadata: bool = True,
    ):
        logger.info("Querying Pinecone vector store…")

        result = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_values=include_values,
            include_metadata=include_metadata,
        )

        logger.info(
            f"Search returned {len(result['matches'])} matches"
        )
        return result["matches"]

    # ----------------------------------------------------------------------
    # 5) Check cache hit (query_hash match)
    # ----------------------------------------------------------------------
    def search_by_query(
        self,
        query: str,
        embedding: List[float],
        datasets: List[str],
        threshold: float = 0.87,
    ) -> Optional[Dict]:
        """Returns the metadata/code if a cached version exists."""

        matches = self.search(embedding, top_k=10, include_metadata=True)
        for match in matches:
            score = match.get("score")
            if score is not None and score >= threshold:
                logger.info(
                    "Cache HIT 🎉 Returning stored python code with cosine score %.3f.",
                    score,
                )
                return match.get("metadata")

        logger.info("Cache MISS ❌ No sufficiently similar code found.")
        return None

    # ----------------------------------------------------------------------
    # 6) Delete vectors (optional)
    # ----------------------------------------------------------------------
    def delete(self, ids: List[str]):
        logger.warning("Deleting vectors from Pinecone index")
        self.index.delete(ids)

    def top_records(self, limit: int = 5) -> List[Dict]:
        """
        Return the oldest `limit` records from the index using an empty query vector.
        """
        logger.info("Fetching top %d records from Pinecone cache", limit)
        results = self.index.query(
            vector=[0.0] * self.dimension,
            top_k=limit,
            include_metadata=True,
            include_values=False,
            filter=None,
        )
        return results.get("matches", [])


# --------------------------------------------------------------------------
# USAGE EXAMPLE (your agent will call these)
# --------------------------------------------------------------------------

"""
vector_store = PineconeVectorStore(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name="portfolio-agent-cache",
    dimension=1536
)

# step 1: embed the user query
embed = embed_model.embed_query("calculate my tech exposure")

# step 2: check if cached
hit = vector_store.search_by_query(
    query="calculate my tech exposure",
    embedding=embed,
    datasets=["positions", "transactions"]
)

if hit:
    python_code = hit["python_code"]
else:
    python_code = llm.generate_python_code(...)
    vector_store.upsert(
        embedding=embed,
        code=python_code,
        query="calculate my tech exposure",
        datasets=["positions", "transactions"]
    )
"""
