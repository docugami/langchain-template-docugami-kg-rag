# OpenAI models and Embeddings
# Reference: https://platform.openai.com/docs/models
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

if "OPENAI_API_KEY" not in os.environ:
    raise Exception("OPENAI_API_KEY environment variable not set")

LARGE_CONTEXT_INSTRUCT_LLM = ChatOpenAI(
    temperature=0,
    model="gpt-4-turbo-preview",  # input context limit is 128k tokens
    cache=True,
    max_tokens=2 * 1024,  # only output tokens
)
SMALL_CONTEXT_INSTRUCT_LLM = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-1106",  # input context limit is 16k tokens
    cache=True,
    max_tokens=2 * 1024,  # only output tokens
)
SQL_GEN_LLM = SMALL_CONTEXT_INSTRUCT_LLM  # Use the same model for sql gen
LLM_BATCH_SIZE = 256

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-ada-002")

# Lengths for the Docugami loader are in terms of characters, 1 token ~= 4 chars in English
# Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
MIN_LENGTH_TO_SUMMARIZE = 2048  # chunks and docs below this length are embedded as-is
MAX_FULL_DOCUMENT_TEXT_LENGTH = int(1024 * 4 * 14)  # ~14k tokens
MAX_CHUNK_TEXT_LENGTH = int(1024 * 4 * 6.5)  # ~6.5k tokens
MIN_CHUNK_TEXT_LENGTH = int(1024 * 4 * 1.5)  # ~1.5k tokens
SUB_CHUNK_TABLES = True
INCLUDE_XML_TAGS = True
PARENT_HIERARCHY_LEVELS = 2
RETRIEVER_K = 8

