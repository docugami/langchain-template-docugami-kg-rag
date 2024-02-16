import sys
from typing import List, Tuple

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.base import BaseTool
from langchain.tools.render import render_text_description

from langchain_docugami.output_parsers.soft_react_json_single_input import SoftReActJsonSingleInputOutputParser
from langchain_docugami.tools.reports import get_retrieval_tool_for_report
from langchain_docugami.tools.retrieval import get_retrieval_tool_for_docset
from langchain_docugami.agents.assistant import ASSISTANT_SYSTEM_MESSAGE

from docugami_kg_rag.config import (
    AGENT_MAX_ITERATIONS,
    EXAMPLES_PATH,
    LARGE_CONTEXT_INSTRUCT_LLM,
    SQL_GEN_LLM,
    DEFAULT_USE_REPORTS,
    RETRIEVER_K,
    EMBEDDINGS,
    get_vector_store_index,
)
from docugami_kg_rag.indexing import read_all_local_index_state


def _get_tools(use_reports=DEFAULT_USE_REPORTS) -> List[BaseTool]:
    """
    Build retrieval tools.
    """

    local_state = read_all_local_index_state()

    tools: List[BaseTool] = []
    for docset_id in local_state:
        docset_state = local_state[docset_id]
        chunk_vectorstore = get_vector_store_index(docset_id, EMBEDDINGS)

        if chunk_vectorstore is not None:

            direct_retrieval_tool = get_retrieval_tool_for_docset(
                chunk_vectorstore=chunk_vectorstore,
                retrieval_tool_function_name=docset_state.retrieval_tool_function_name,
                retrieval_tool_description=docset_state.retrieval_tool_description,
                full_doc_summary_store=docset_state.full_doc_summaries_by_id,
                parent_doc_store=docset_state.chunks_by_id,
                retrieval_k=RETRIEVER_K,
            )
            if direct_retrieval_tool:
                # Direct retrieval tool for each indexed docset (direct KG-RAG against semantic XML)
                tools.append(direct_retrieval_tool)

        if use_reports:
            for report in docset_state.reports:
                # Report retrieval tool for each published report (user-curated views on semantic XML)
                report_retrieval_tool = get_retrieval_tool_for_report(
                    local_xlsx_path=report.local_xlsx_path,
                    report_name=report.name,
                    retrieval_tool_function_name=report.retrieval_tool_function_name,
                    retrieval_tool_description=report.retrieval_tool_description,
                    sql_llm=SQL_GEN_LLM,
                    embeddings=EMBEDDINGS,
                    sql_fixup_examples_file=EXAMPLES_PATH / "sql_fixup_examples.yaml",
                    sql_examples_file=EXAMPLES_PATH / "sql_examples.yaml",
                )
                if report_retrieval_tool:
                    tools.append(report_retrieval_tool)

    return tools


# setup ReAct style prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ASSISTANT_SYSTEM_MESSAGE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}\n\n{agent_scratchpad}"),
    ]
)

model_with_stop = LARGE_CONTEXT_INSTRUCT_LLM.bind(stop=["\nObservation"])


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        "tools": lambda x: render_text_description(_get_tools(x["use_reports"])),
        "tool_names": lambda x: ", ".join([t.name for t in _get_tools(x["use_reports"])]),
    }
    | prompt
    | model_with_stop
    | SoftReActJsonSingleInputOutputParser()
)


class AgentInput(BaseModel):
    input: str = ""
    use_reports: bool = Field(
        default=DEFAULT_USE_REPORTS,
        extra={"widget": {"type": "bool", "input": "input", "output": "output"}},
    )
    chat_history: List[Tuple[str, str]] = Field(
        default=[],
        extra={
            "widget": {"type": "chat", "input": "input", "output": "output"},
        },
    )


agent = AgentExecutor(
    agent=agent,  # type: ignore
    tools=_get_tools(True),  # pass in ALL the tools here (the prompt will filter down to non-report ones if needed)
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=AGENT_MAX_ITERATIONS,
).with_types(
    input_type=AgentInput,  # type: ignore
)

if __name__ == "__main__":
    if sys.gettrace():
        # This code will only run if a debugger is attached

        output = agent.invoke(
            {
                "input": "What is the project number for the contract with snelson?",
                "chat_history": [],
                "use_reports": False,
            }
        )

        print(output)
