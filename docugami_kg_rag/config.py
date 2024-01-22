import os
from pathlib import Path

from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache


##### <LLMs and Embeddings>
# OpenAI models and Embeddings
# Reference: https://platform.openai.com/docs/models
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings

LARGE_CONTEXT_LLM = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")  # 128k tokens
SMALL_CONTEXT_LLM = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")  # 16k tokens
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-ada-002")
##### </LLMs and Embeddings>

##### <Vector Store>
CHROMA_DIRECTORY = "/tmp/docugami/chroma_db"
os.makedirs(Path(CHROMA_DIRECTORY).parent, exist_ok=True)
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

# Lengths for the Docugami loader are in terms of characters, 1 token ~= 4 chars in English
# Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
MAX_CHUNK_TEXT_LENGTH = 1024 * 26  # ~6.5k tokens
MIN_CHUNK_TEXT_LENGTH = 1024 * 6  # ~1.5k tokens
SUB_CHUNK_TABLES = False
INCLUDE_XML_TAGS = True
PARENT_HIERARCHY_LEVELS = 2
RETRIEVER_K = 8

BATCH_SIZE = 16
