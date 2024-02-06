# Reference: https://fireworks.ai/models
import os

from langchain.chat_models.fireworks import ChatFireworks

if "FIREWORKS_API_KEY" not in os.environ:
    raise Exception("FIREWORKS_API_KEY environment variable not set")
LARGE_CONTEXT_LLM = ChatFireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    model_kwargs={"temperature": 0, "max_tokens": 1024},
    cache=True,
)  # 128k tokens
SMALL_CONTEXT_LLM = LARGE_CONTEXT_LLM  # Use the same model for large and small context tasks

# Lengths for the Docugami loader are in terms of characters, 1 token ~= 4 chars in English
# Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
MIN_LENGTH_TO_SUMMARIZE = 2048  # chunks and docs below this length are embedded as-is
MAX_FULL_DOCUMENT_TEXT_LENGTH = 1024 * 56  # ~14k tokens
MAX_CHUNK_TEXT_LENGTH = 1024 * 18  # ~4.5k tokens
MIN_CHUNK_TEXT_LENGTH = 1024 * 6  # ~1.5k tokens
SUB_CHUNK_TABLES = False
INCLUDE_XML_TAGS = True
PARENT_HIERARCHY_LEVELS = 2
RETRIEVER_K = 1

BATCH_SIZE = 16
