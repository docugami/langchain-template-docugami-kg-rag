import sys
from typing import Dict, List, Optional, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.runnable import Runnable, RunnableLambda, RunnableParallel
from langchain.tools.base import BaseTool
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.pydantic_v1 import BaseModel, Field

from docugami_kg_rag.config import LARGE_CONTEXT_LLM, DEFAULT_USE_REPORTS
from docugami_kg_rag.helpers.indexing import read_all_local_index_state
from docugami_kg_rag.helpers.prompts import ASSISTANT_SYSTEM_MESSAGE
from docugami_kg_rag.helpers.reports import get_retrieval_tool_for_report
from docugami_kg_rag.helpers.retrieval import get_retrieval_tool_for_docset


def _get_tools(use_reports=DEFAULT_USE_REPORTS) -> List[BaseTool]:
    """
    Build retrieval tools.
    """

    local_state = read_all_local_index_state()

    docset_retrieval_tools: List[BaseTool] = []
    report_retrieval_tools: List[BaseTool] = []
    for docset_id in local_state:
        docset_state = local_state[docset_id]
        direct_retrieval_tool = get_retrieval_tool_for_docset(docset_id, docset_state)
        if direct_retrieval_tool:
            # Direct retrieval tool for each indexed docset (direct KG-RAG against semantic XML)
            docset_retrieval_tools.append(direct_retrieval_tool)

        for report in docset_state.reports:
            # Report retrieval tool for each published report (user-curated views on semantic XML)
            report_retrieval_tool = get_retrieval_tool_for_report(report)
            if report_retrieval_tool:
                report_retrieval_tools.append(report_retrieval_tool)

    tools = docset_retrieval_tools

    if use_reports:
        tools = tools + report_retrieval_tools

    return tools


def _llm_with_tools(input: Dict) -> Runnable:
    return RunnableLambda(lambda x: x["input"]) | LARGE_CONTEXT_LLM.bind(functions=input["functions"])  # type: ignore


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ASSISTANT_SYSTEM_MESSAGE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = (
    RunnableParallel(
        {
            "input": lambda x: x["input"],  # type: ignore
            "chat_history": lambda x: _format_chat_history(x["chat_history"]),  # type: ignore
            "agent_scratchpad": lambda x: format_to_openai_functions(x["intermediate_steps"]),  # type: ignore
            "functions": lambda x: [format_tool_to_openai_function(tool) for tool in _get_tools("use_reports" in x and x["use_reports"] is True)],  # type: ignore
        }
    )
    | {
        "input": prompt,
        "functions": lambda x: x["functions"],
    }
    | _llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)


class AgentInput(BaseModel):
    input: str = ""
    use_reports: bool = Field(
        default=DEFAULT_USE_REPORTS,
        extra={"widget": {"type": "bool", "input": "input", "output": "output"}},
    )
    chat_history: Optional[List[Tuple[str, str]]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


chain = AgentExecutor(
    agent=agent,  # type: ignore
    tools=_get_tools(),
).with_types(
    input_type=AgentInput,  # type: ignore
)

if __name__ == "__main__":
    if sys.gettrace():
        # This code will only run if a debugger is attached

        chain.invoke(
            {
                "input": "What happened in Yelm, Washington?",
                "chat_history": [],
            }
        )
