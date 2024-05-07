import hashlib
import os
from pathlib import Path

from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain.cache import GPTCache
from langchain_core.globals import set_llm_cache

from docugami_kg_rag.config.chromadb import *
from docugami_kg_rag.config.fireworksai_llama3 import *

# from docugami_kg_rag.config.fireworksai_mixtral import *
# from docugami_kg_rag.config.openai import *
from docugami_kg_rag.config.huggingface import *

# from docugami_kg_rag.config.redis import *

DOCUGAMI_API_ENDPOINT = "https://api.docugami.com/v1preview1"
DOCUGAMI_API_KEY = os.environ.get("DOCUGAMI_API_KEY")
if not DOCUGAMI_API_KEY:
    raise Exception("Please set the DOCUGAMI_API_KEY environment variable")


INDEXING_LOCAL_STATE_PATH = os.environ.get(
    "INDEXING_LOCAL_STATE_PATH", "/tmp/docugami/indexing_local_state.pkl"
)
os.makedirs(Path(INDEXING_LOCAL_STATE_PATH).parent, exist_ok=True)

INDEXING_LOCAL_REPORT_DBS_ROOT = os.environ.get(
    "INDEXING_LOCAL_REPORT_DBS_ROOT", "/tmp/docugami/report_dbs"
)
os.makedirs(Path(INDEXING_LOCAL_REPORT_DBS_ROOT).parent, exist_ok=True)

LOCAL_LLM_CACHE_DIR = os.environ.get("LOCAL_LLM_CACHE", "/tmp/docugami/langchain_cache")
os.makedirs(Path(LOCAL_LLM_CACHE_DIR).parent, exist_ok=True)


def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    hashed_llm_dir = Path(LOCAL_LLM_CACHE_DIR) / hashed_llm
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(
            manager="map", data_dir=str(hashed_llm_dir.absolute())
        ),
    )


set_llm_cache(GPTCache(init_gptcache))

EXAMPLES_PATH = Path(__file__).parent.parent / "green_examples"

DEFAULT_USE_REPORTS = True
DEFAULT_USE_CONVERSATIONAL_TOOLS = True
