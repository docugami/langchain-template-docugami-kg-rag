import os
from pathlib import Path

from langchain.cache import SQLiteCache

from langchain.globals import set_llm_cache

from docugami_kg_rag.config.openai import *
from docugami_kg_rag.config.chromadb import *

# from docugami_kg_rag.config.fireworksai import *
# from docugami_kg_rag.config.huggingface import *
# from docugami_kg_rag.config.redis import *

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

EXAMPLES_PATH = Path(__file__).parent.parent / "green_examples"

DEFAULT_USE_REPORTS = False
