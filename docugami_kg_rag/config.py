import os
from pathlib import Path

from langchain_core.vectorstores import VectorStore

from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

##### <LLMs and Embeddings>
# OpenAI models and Embeddings
# Reference: https://platform.openai.com/docs/models
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

if "OPENAI_API_KEY" not in os.environ:
    raise Exception("OPENAI_API_KEY environment variable not set")

LARGE_CONTEXT_LLM = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", cache=True)  # 128k tokens
SMALL_CONTEXT_LLM = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106", cache=True)  # 16k tokens
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-ada-002")

# Lengths for the Docugami loader are in terms of characters, 1 token ~= 4 chars in English
# Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
MIN_LENGTH_TO_SUMMARIZE = 2048  # chunks and docs below this length are embedded as-is
MAX_FULL_DOCUMENT_TEXT_LENGTH = 1024 * 56  # ~14k tokens
MAX_CHUNK_TEXT_LENGTH = 1024 * 26  # ~6.5k tokens
MIN_CHUNK_TEXT_LENGTH = 1024 * 6  # ~1.5k tokens
SUB_CHUNK_TABLES = False
INCLUDE_XML_TAGS = False
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
# MAX_CHUNK_TEXT_LENGTH = 1024 * 26  # ~6.5k tokens
# MIN_CHUNK_TEXT_LENGTH = 1024 * 6  # ~1.5k tokens
# SUB_CHUNK_TABLES = False
# INCLUDE_XML_TAGS = False
# PARENT_HIERARCHY_LEVELS = 2
# RETRIEVER_K = 8

# BATCH_SIZE = 16
##### </LLMs and Embeddings>

##### <Vector Store>
from langchain_community.vectorstores.chroma import Chroma
import chromadb

CHROMA_DIRECTORY = Path("/tmp/docugami/chroma_db")


def get_vector_store_index(docset_id: str) -> VectorStore:
    return Chroma(
        collection_name=docset_id,
        persist_directory=str(CHROMA_DIRECTORY.absolute()),
        embedding_function=EMBEDDINGS,
    )


def vector_store_index_exists(docset_id: str) -> bool:
    persistent_client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY.absolute()))
    collections = persistent_client.list_collections()
    for c in collections:
        if c.name == docset_id:
            return True

    return False


def del_vector_store_index(docset_id: str):
    persistent_client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY.absolute()))
    persistent_client.delete_collection(docset_id)


##### </Vector Store>

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

USE_REPORTS = True
