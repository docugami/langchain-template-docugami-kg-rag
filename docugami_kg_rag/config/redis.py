# Reference: https://python.langchain.com/docs/integrations/vectorstores/redis
from typing import List, Optional

from docugami_langchain.retrievers.fused_summary import (
    FULL_DOC_SUMMARY_ID_KEY,
    PARENT_DOC_ID_KEY,
    SOURCE_KEY,
)
from langchain_community.vectorstores.redis.base import Redis, check_index_exists
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

REDIS_URL = "redis://localhost:6379"
REDIS_INDEX_SCHEMA = {
    "text": [
        {"name": "id"},
        {"name": PARENT_DOC_ID_KEY},
        {"name": FULL_DOC_SUMMARY_ID_KEY},
        {"name": SOURCE_KEY},
    ],
}


def vector_store_index_exists(docset_id: str, embeddings: Embeddings) -> bool:
    conn = Redis(redis_url=REDIS_URL, index_name=docset_id, embedding=embeddings)
    return check_index_exists(conn.client, docset_id)


def get_vector_store_index(docset_id: str, embeddings: Embeddings) -> Optional[VectorStore]:
    if vector_store_index_exists(docset_id, embeddings):
        return Redis.from_existing_index(
            embedding=embeddings, index_name=docset_id, schema=REDIS_INDEX_SCHEMA, redis_url=REDIS_URL  # type: ignore
        )
    else:
        return None


def init_vector_store_index(docset_id: str, docs: List[Document], embeddings: Embeddings, force: bool = True) -> VectorStore:
    if force and vector_store_index_exists(docset_id, embeddings):
        del_vector_store_index(docset_id)

    return Redis.from_documents(docs, index_name=docset_id, embedding=embeddings, redis_url=REDIS_URL)


def del_vector_store_index(docset_id: str) -> None:
    Redis.drop_index(docset_id, True, redis_url=REDIS_URL)
