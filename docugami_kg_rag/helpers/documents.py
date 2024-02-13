import hashlib
from tqdm import tqdm
from typing import Dict

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chat_models.base import BaseChatModel
from langchain.schema.runnable import RunnableLambda, RunnableBranch
from langchain_core.documents import Document

from langchain_docugami.retrievers.fused_summary import PARENT_DOC_ID_KEY
from langchain_docugami.prompts import (
    CREATE_CHUNK_SUMMARY_SYSTEM_MESSAGE,
    CREATE_FULL_DOCUMENT_SUMMARY_PROMPT,
    CREATE_CHUNK_SUMMARY_PROMPT,
    CREATE_FULL_DOCUMENT_SUMMARY_SYSTEM_MESSAGE,
)

from docugami_kg_rag.config import (
    BATCH_SIZE,
    INCLUDE_XML_TAGS,
    MAX_CHUNK_TEXT_LENGTH,
    MAX_FULL_DOCUMENT_TEXT_LENGTH,
    MIN_LENGTH_TO_SUMMARIZE,
    SMALL_CONTEXT_INSTRUCT_LLM,
    LARGE_CONTEXT_INSTRUCT_LLM,
)


def _build_summary_mappings(
    docs_by_id: Dict[str, Document],
    system_message: str,
    prompt_template: str,
    llm: BaseChatModel = SMALL_CONTEXT_INSTRUCT_LLM,
    min_length_to_summarize=MIN_LENGTH_TO_SUMMARIZE,
    max_length_cutoff=MAX_CHUNK_TEXT_LENGTH,
    label="summaries",
) -> Dict[str, Document]:
    """
    Build summaries for all the given documents.
    """
    summaries: Dict[str, Document] = {}
    format = "text" if not INCLUDE_XML_TAGS else "semantic XML without any namespaces or attributes"

    # Splitting the documents into batches
    doc_items = list(docs_by_id.items())
    for i in tqdm(
        range(0, len(doc_items), BATCH_SIZE),
        f"Creating {label} in batches",
    ):
        batch = doc_items[i : i + BATCH_SIZE]

        # Preparing batch input
        batch_input = [
            {
                "format": format,
                "document": doc.page_content[:max_length_cutoff],
            }
            for _, doc in batch
        ]

        summarize_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", system_message),
                    ("human", prompt_template),
                ]
            )
            | llm
            | StrOutputParser()
        )
        noop_chain = RunnableLambda(lambda x: x["document"])

        # Build meta chain that only summarizes inputs larger than threshold
        chain = RunnableBranch(
            (lambda x: len(x["document"]) > min_length_to_summarize, summarize_chain),  # type: ignore
            noop_chain,
        )

        batch_summaries = chain.batch(batch_input)

        # Assigning summaries to the respective document IDs
        for (id, doc), summary in zip(batch, batch_summaries):
            summary_id = hashlib.md5(summary.encode()).hexdigest()
            meta = doc.metadata
            meta["id"] = summary_id
            meta[PARENT_DOC_ID_KEY] = id

            summaries[id] = Document(
                page_content=summary,
                metadata=meta,
            )

    return summaries


def build_full_doc_summary_mappings(docs_by_id: Dict[str, Document]) -> Dict[str, Document]:
    """
    Build summaries for all the given full documents.
    """

    return _build_summary_mappings(
        docs_by_id=docs_by_id,
        system_message=CREATE_FULL_DOCUMENT_SUMMARY_SYSTEM_MESSAGE,
        prompt_template=CREATE_FULL_DOCUMENT_SUMMARY_PROMPT,
        llm=LARGE_CONTEXT_INSTRUCT_LLM,
        max_length_cutoff=MAX_FULL_DOCUMENT_TEXT_LENGTH,
        label="full document summaries",
    )


def build_chunk_summary_mappings(docs_by_id: Dict[str, Document]) -> Dict[str, Document]:
    """
    Build summaries for all the given chunks.
    """

    return _build_summary_mappings(
        docs_by_id=docs_by_id,
        system_message=CREATE_CHUNK_SUMMARY_SYSTEM_MESSAGE,
        prompt_template=CREATE_CHUNK_SUMMARY_PROMPT,
        llm=SMALL_CONTEXT_INSTRUCT_LLM,
        max_length_cutoff=MAX_CHUNK_TEXT_LENGTH,
        label="chunk summaries",
    )
