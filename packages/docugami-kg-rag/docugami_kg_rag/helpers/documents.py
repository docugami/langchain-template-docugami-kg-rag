from tqdm import tqdm
from typing import Dict

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.documents import Document

from docugami_kg_rag.config import (
    BATCH_SIZE,
    INCLUDE_XML_TAGS,
    LARGE_FRAGMENT_MAX_TEXT_LENGTH,
    LLM,
)
from docugami_kg_rag.helpers.prompts import (
    ASSISTANT_SYSTEM_MESSAGE,
    CREATE_FULL_DOCUMENT_SUMMARY_PROMPT,
    CREATE_CHUNK_SUMMARY_PROMPT,
)


def _build_summary_mappings(docs_by_id: Dict[str, Document], prompt_template: str) -> Dict[str, str]:
    """
    Build summaries for all the given documents.
    """
    summaries: Dict[str, str] = {}
    format = "text" if not INCLUDE_XML_TAGS else "semantic XML without any namespaces or attributes"

    # Splitting the documents into batches
    doc_items = list(docs_by_id.items())
    for i in tqdm(
        range(0, len(doc_items), BATCH_SIZE),
        "Creating full document summaries in batches",
    ):
        batch = doc_items[i : i + BATCH_SIZE]

        # Preparing batch input
        batch_input = [
            {
                "format": format,
                "document": doc.page_content[:LARGE_FRAGMENT_MAX_TEXT_LENGTH],
            }
            for _, doc in batch
        ]

        # Processing the batch
        chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", ASSISTANT_SYSTEM_MESSAGE),
                    ("human", prompt_template),
                ]
            )
            | LLM
            | StrOutputParser()
        )
        batch_summaries = chain.batch(batch_input)

        # Assigning summaries to the respective document IDs
        for (id, _), summary in zip(batch, batch_summaries):
            summaries[id] = summary

    return summaries


def build_full_doc_summary_mappings(docs_by_id: Dict[str, Document]) -> Dict[str, str]:
    """
    Build summaries for all the given full documents.
    """

    return _build_summary_mappings(
        docs_by_id=docs_by_id,
        prompt_template=CREATE_FULL_DOCUMENT_SUMMARY_PROMPT,
    )


def build_chunk_summary_mappings(docs_by_id: Dict[str, Document]) -> Dict[str, str]:
    """
    Build summaries for all the given chunks.
    """

    return _build_summary_mappings(
        docs_by_id=docs_by_id,
        prompt_template=CREATE_CHUNK_SUMMARY_PROMPT,
    )
