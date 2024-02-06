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
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.base import BaseTool
from langchain.tools.render import render_text_description


from docugami_kg_rag.config import AGENT_MAX_ITERATIONS, LARGE_CONTEXT_LLM, DEFAULT_USE_REPORTS
from docugami_kg_rag.helpers.indexing import read_all_local_index_state
from docugami_kg_rag.helpers.prompts import ASSISTANT_SYSTEM_MESSAGE
from docugami_kg_rag.helpers.reports import get_retrieval_tool_for_report
from docugami_kg_rag.helpers.retrieval import get_retrieval_tool_for_docset


def _get_tools(use_reports=DEFAULT_USE_REPORTS) -> List[BaseTool]:
    """
    Build retrieval tools.
    """

    local_state = read_all_local_index_state()

    tools: List[BaseTool] = []
    for docset_id in local_state:
        docset_state = local_state[docset_id]
        direct_retrieval_tool = get_retrieval_tool_for_docset(docset_id, docset_state)
        if direct_retrieval_tool:
            # Direct retrieval tool for each indexed docset (direct KG-RAG against semantic XML)
            tools.append(direct_retrieval_tool)

        if use_reports:
            for report in docset_state.reports:
                # Report retrieval tool for each published report (user-curated views on semantic XML)
                report_retrieval_tool = get_retrieval_tool_for_report(report)
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

model_with_stop = LARGE_CONTEXT_LLM.bind(stop=["\nObservation"])


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
    | ReActJsonSingleInputOutputParser()
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


chain = AgentExecutor(
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

        output = chain.invoke(
            {
                "input": "What is the project number for the contract with snelson?",
                "chat_history": [],
                "use_reports": False,
            }
        )

        print(output)
