import os
from pathlib import Path
from typing import Optional, List

from langchain_core.vectorstores import VectorStore

from langchain.cache import SQLiteCache
from langchain.schema import Document
from langchain.globals import set_llm_cache

DOCUGAMI_API_KEY = os.environ.get("DOCUGAMI_API_KEY")
if not DOCUGAMI_API_KEY:
    raise Exception("Please set the DOCUGAMI_API_KEY environment variable")


INDEXING_LOCAL_STATE_PATH = os.environ.get("INDEXING_LOCAL_STATE_PATH", "/tmp/docugami/indexing_local_state.pkl")
os.makedirs(Path(INDEXING_LOCAL_STATE_PATH).parent, exist_ok=True)

INDEXING_LOCAL_REPORT_DBS_ROOT = os.environ.get("INDEXING_LOCAL_REPORT_DBS_ROOT", "/tmp/docugami/report_dbs")
os.makedirs(Path(INDEXING_LOCAL_REPORT_DBS_ROOT).parent, exist_ok=True)

LOCAL_LLM_CACHE_DB_FILE = os.environ.get("LOCAL_LLM_CACHE", "/tmp/docugami/.langchain.db")
os.makedirs(Path(LOCAL_LLM_CACHE_DB_FILE).parent, exist_ok=True)
set_llm_cache(SQLiteCache(database_path=LOCAL_LLM_CACHE_DB_FILE))

DEFAULT_USE_REPORTS = False
AGENT_MAX_ITERATIONS = 5

##### <LLMs and Embeddings>
# OpenAI models and Embeddings
# Reference: https://platform.openai.com/docs/models
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

if "OPENAI_API_KEY" not in os.environ:
    raise Exception("OPENAI_API_KEY environment variable not set")

LARGE_CONTEXT_LLM = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview", cache=True)  # 128k tokens
SMALL_CONTEXT_LLM = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106", cache=True)  # 16k tokens
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-ada-002")

# Lengths for the Docugami loader are in terms of characters, 1 token ~= 4 chars in English
# Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
MIN_LENGTH_TO_SUMMARIZE = 2048  # chunks and docs below this length are embedded as-is
MAX_FULL_DOCUMENT_TEXT_LENGTH = 1024 * 56  # ~14k tokens
MAX_CHUNK_TEXT_LENGTH = 1024 * 26  # ~6.5k tokens
MIN_CHUNK_TEXT_LENGTH = 1024 * 6  # ~1.5k tokens
SUB_CHUNK_TABLES = False
INCLUDE_XML_TAGS = True
PARENT_HIERARCHY_LEVELS = 2
RETRIEVER_K = 8

BATCH_SIZE = 16

# FireworksAI models and local embeddings
# Reference: https://fireworks.ai/models
# Reference: https://huggingface.co/models
# import torch
# from langchain.chat_models.fireworks import ChatFireworks
# from langchain_community.embeddings import HuggingFaceEmbeddings

# if "FIREWORKS_API_KEY" not in os.environ:
#     raise Exception("FIREWORKS_API_KEY environment variable not set")
# LARGE_CONTEXT_LLM = ChatFireworks(
#     model="accounts/fireworks/models/mixtral-8x7b-instruct",
#     model_kwargs={"temperature": 0, "max_tokens": 1024},
#     cache=True,
# )  # 128k tokens
# SMALL_CONTEXT_LLM = LARGE_CONTEXT_LLM  # Use the same model for large and small context tasks
# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"

# EMBEDDINGS = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     model_kwargs={"device": device},
# )

# # Lengths for the Docugami loader are in terms of characters, 1 token ~= 4 chars in English
# # Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
# MIN_LENGTH_TO_SUMMARIZE = 2048  # chunks and docs below this length are embedded as-is
# MAX_FULL_DOCUMENT_TEXT_LENGTH = 1024 * 56  # ~14k tokens
# MAX_CHUNK_TEXT_LENGTH = 1024 * 18  # ~4.5k tokens
# MIN_CHUNK_TEXT_LENGTH = 1024 * 6  # ~1.5k tokens
# SUB_CHUNK_TABLES = False
# INCLUDE_XML_TAGS = True
# PARENT_HIERARCHY_LEVELS = 2
# RETRIEVER_K = 1

# BATCH_SIZE = 16
##### </LLMs and Embeddings>

##### <Vector Store>
# ChromaDB
# Reference: https://python.langchain.com/docs/integrations/vectorstores/chroma
from langchain_community.vectorstores.chroma import Chroma
import chromadb

CHROMA_DIRECTORY = Path("/tmp/docugami/chroma_db")
CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)


def vector_store_index_exists(docset_id: str) -> bool:
    persistent_client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY.absolute()))
    collections = persistent_client.list_collections()
    for c in collections:
        if c.name == docset_id:
            return True

    return False


def get_vector_store_index(docset_id: str) -> Optional[VectorStore]:
    if vector_store_index_exists(docset_id):
        return Chroma(
            collection_name=docset_id,
            persist_directory=str(CHROMA_DIRECTORY.absolute()),
            embedding_function=EMBEDDINGS,
        )

    return None


def init_vector_store_index(docset_id: str, docs: List[Document], force=True) -> VectorStore:
    if force and vector_store_index_exists(docset_id):
        del_vector_store_index(docset_id)

    return Chroma.from_documents(
        documents=docs,
        collection_name=docset_id,
        embedding=EMBEDDINGS,
        persist_directory=str(CHROMA_DIRECTORY.absolute()),
    )


def del_vector_store_index(docset_id: str):
    persistent_client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY.absolute()))
    persistent_client.delete_collection(docset_id)


# Redis
# Reference: https://python.langchain.com/docs/integrations/vectorstores/redis
# from langchain_community.vectorstores.redis.base import Redis, check_index_exists
# from docugami_kg_rag.helpers.fused_summary_retriever import PARENT_DOC_ID_KEY, FULL_DOC_SUMMARY_ID_KEY, SOURCE_KEY

# REDIS_URL = "redis://localhost:6379"
# REDIS_INDEX_SCHEMA = {
#     "text": [
#         {"name": "id"},
#         {"name": PARENT_DOC_ID_KEY},
#         {"name": FULL_DOC_SUMMARY_ID_KEY},
#         {"name": SOURCE_KEY},
#     ],
# }


# def vector_store_index_exists(docset_id: str) -> bool:
#     conn = Redis(redis_url=REDIS_URL, index_name=docset_id, embedding=EMBEDDINGS)
#     return check_index_exists(conn.client, docset_id)


# def get_vector_store_index(docset_id: str) -> Optional[VectorStore]:
#     if vector_store_index_exists(docset_id):
#         return Redis.from_existing_index(
#             embedding=EMBEDDINGS, index_name=docset_id, schema=REDIS_INDEX_SCHEMA, redis_url=REDIS_URL  # type: ignorecl
#         )
#     else:
#         return None


# def init_vector_store_index(docset_id: str, docs: List[Document], force=True) -> VectorStore:
#     if force and vector_store_index_exists(docset_id):
#         del_vector_store_index(docset_id)

#     return Redis.from_documents(docs, index_name=docset_id, embedding=EMBEDDINGS, redis_url=REDIS_URL)


# def del_vector_store_index(docset_id: str):
#     Redis.drop_index(docset_id, True, redis_url=REDIS_URL)

##### </Vector Store>
