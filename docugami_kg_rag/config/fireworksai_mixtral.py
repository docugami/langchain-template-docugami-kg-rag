# Reference: https://fireworks.ai/models
import os

from langchain_fireworks.chat_models import ChatFireworks

if "FIREWORKS_API_KEY" not in os.environ:
    raise Exception("FIREWORKS_API_KEY environment variable not set")
LARGE_CONTEXT_INSTRUCT_LLM = ChatFireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",  # input context limit is 32k tokens
    temperature=0,
    max_tokens=32 * 1024,  # this sets the total token max (input and output)
    model_kwargs={
        "context_length_exceeded_behavior": "truncate",
    },
    cache=True,
)
SMALL_CONTEXT_INSTRUCT_LLM = LARGE_CONTEXT_INSTRUCT_LLM  # Use the same model for large and small context tasks
SQL_GEN_LLM = LARGE_CONTEXT_INSTRUCT_LLM  # Use the same model for sql gen
LLM_BATCH_SIZE = 32

# Lengths for the Docugami loader are in terms of characters, 1 token ~= 4 chars in English
# Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
MIN_LENGTH_TO_SUMMARIZE = 2048  # chunks and docs below this length are embedded as-is
MAX_FULL_DOCUMENT_TEXT_LENGTH = int(1024 * 4 * 14)  # ~14k tokens
MAX_CHUNK_TEXT_LENGTH = int(1024 * 4 * 4.5)  # ~4.5k tokens
MIN_CHUNK_TEXT_LENGTH = int(1024 * 4 * 1.5)  # ~1.5k tokens
SUB_CHUNK_TABLES = False
INCLUDE_XML_TAGS = False
PARENT_HIERARCHY_LEVELS = 2
RETRIEVER_K = 8
