from dataclasses import dataclass, field
from typing import List


from langchain.storage.in_memory import InMemoryStore
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
