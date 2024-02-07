# OpenAI models and Embeddings
# Reference: https://platform.openai.com/docs/models
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

if "OPENAI_API_KEY" not in os.environ:
    raise Exception("OPENAI_API_KEY environment variable not set")

LARGE_CONTEXT_INSTRUCT_LLM = ChatOpenAI(temperature=0.5, model="gpt-4-turbo-preview", cache=True)  # 128k tokens
SMALL_CONTEXT_INSTRUCT_LLM = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-1106", cache=True)  # 16k tokens
SQL_GEN_LLM = SMALL_CONTEXT_INSTRUCT_LLM  # Use the same model for sql gen

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
