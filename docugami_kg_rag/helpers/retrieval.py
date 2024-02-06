from dataclasses import dataclass, field
import re
from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain.storage.in_memory import InMemoryStore
from langchain.tools.base import Tool

from docugami_kg_rag.config import (
    LARGE_CONTEXT_LLM,
    RETRIEVER_K,
    MAX_CHUNK_TEXT_LENGTH,
    get_vector_store_index,
)
from docugami_kg_rag.helpers.fused_summary_retriever import (
    FusedSummaryRetriever,
    SearchType,
)
from docugami_kg_rag.helpers.prompts import (
    CREATE_DIRECT_RETRIEVAL_TOOL_SYSTEM_MESSAGE,
    CREATE_DIRECT_RETRIEVAL_TOOL_DESCRIPTION_PROMPT,
)
from docugami_kg_rag.helpers.reports import ReportDetails


@dataclass
class LocalIndexState:
    full_doc_summaries_by_id: InMemoryStore
    """Mapping of ID to full document summaries."""

    chunks_by_id: InMemoryStore
    """Mapping of ID to chunks."""

    retrieval_tool_function_name: str
    """Function name for retrieval tool e.g. "search_earnings_calls."""

    retrieval_tool_description: str
    """Description of retrieval tool e.g. Searches for and returns chunks from earnings call documents."""

    reports: List[ReportDetails] = field(default_factory=list)
    """Details about any reports for this docset."""


class RetrieverInput(BaseModel):
    """Input to the retriever."""

    query: str = Field(description="query to look up in retriever")


def docset_name_to_direct_retriever_tool_function_name(name: str) -> str:
    """
    Converts a docset name to a direct retriever tool function name.

    Direct retriever tool function names follow these conventions:
    1. Retrieval tool function names always start with "search_".
    2. The rest of the name should be a lowercased string, with underscores for whitespace.
    3. Exclude any characters other than a-z (lowercase) from the function name, replacing them with underscores.
    4. The final function name should not have more than one underscore together.

    >>> docset_name_to_direct_retriever_tool_function_name('Earnings Calls')
    'search_earnings_calls'
    >>> docset_name_to_direct_retriever_tool_function_name('COVID-19   Statistics')
    'search_covid_19_statistics'
    >>> docset_name_to_direct_retriever_tool_function_name('2023 Market Report!!!')
    'search_2023_market_report'
    """
    # Replace non-letter characters with underscores and remove extra whitespaces
    name = re.sub(r"[^a-z\d]", "_", name.lower())
    # Replace whitespace with underscores and remove consecutive underscores
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    name = name.strip("_")

    return f"search_{name}"


def chunks_to_direct_retriever_tool_description(name: str, chunks: List[Document]):
    """
    Converts a set of chunks to a direct retriever tool description.
    """
    texts = [c.page_content for c in chunks[:100]]
    document = "\n".join(texts)[:MAX_CHUNK_TEXT_LENGTH]

    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", CREATE_DIRECT_RETRIEVAL_TOOL_SYSTEM_MESSAGE),
                ("human", CREATE_DIRECT_RETRIEVAL_TOOL_DESCRIPTION_PROMPT),
            ]
        )
        | LARGE_CONTEXT_LLM
        | StrOutputParser()
    )
    summary = chain.invoke({"docset_name": name, "document": document})
    return f"Searches for and returns chunks from {name} documents. {summary}. Action input must be a single 'query' string."


def get_retrieval_tool_for_docset(docset_id: str, docset_state: LocalIndexState) -> Optional[Tool]:
    """
    Gets a retrieval tool for an agent.
    """

    chunk_vectorstore = get_vector_store_index(docset_id)

    if not chunk_vectorstore:
        return None

    retriever = FusedSummaryRetriever(
        vectorstore=chunk_vectorstore,
        parent_doc_store=docset_state.chunks_by_id,
        full_doc_summary_store=docset_state.full_doc_summaries_by_id,
        search_kwargs={"k": RETRIEVER_K},
        search_type=SearchType.mmr,
    )

    if not retriever:
        return None

    def wrapped_get_relevant_documents(query, callbacks=None, tags=None, metadata=None, run_name=None, **kwargs):
        docs: List[Document] = retriever.get_relevant_documents(
            query, callbacks=callbacks, tags=tags, metadata=metadata, run_name=run_name, **kwargs
        )
        return "\n\n".join([doc.page_content for doc in docs])

    async def awrapped_get_relevant_documents(query, callbacks=None, tags=None, metadata=None, run_name=None, **kwargs):
        docs: List[Document] = await retriever.aget_relevant_documents(
            query, callbacks=callbacks, tags=tags, metadata=metadata, run_name=run_name, **kwargs
        )
        return "\n\n".join([doc.page_content for doc in docs])

    return Tool(
        name=docset_state.retrieval_tool_function_name,
        description=docset_state.retrieval_tool_description,
        func=wrapped_get_relevant_documents,
        coroutine=awrapped_get_relevant_documents,
        args_schema=RetrieverInput,
    )
