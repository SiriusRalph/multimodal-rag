"""Étape 6 : agent ReAct composé avec LangGraph.

MIND (le LLM) raisonne et décide ; BODY (l'outil) agit et renvoie une observation.
La boucle s'arrête lorsque MIND répond sans demander d'action.
"""
from collections.abc import Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from . import config
from .retrieval import Retriever
from .tools import build_retriever_tool

SYSTEM_PROMPT = """\
You answer questions about the paper "Attention Is All You Need" using retriever_tool.

Work in a loop: think about what is missing, call retriever_tool, read the observation,
then either search again or answer.
- Always search before answering a question about the paper.
- Write search queries in English, even if the user writes in another language.
- When the user provides an image path, pass it as image_path.
- Answer only from the observations. Every number you state must appear word for word
  in an observation. Do not quote text that is not in an observation.
- If the observations do not contain the answer, reply that the paper does not give
  this information, and stop.
- Cite the page of each fact, for example (page 5), and write figure names as "Figure 1".
- Answer in the language of the user's question.
"""


def build_agent(retriever: Retriever, model: str = config.OLLAMA_MODEL):
    tool = build_retriever_tool(retriever)
    llm = ChatOllama(model=model, temperature=0, num_ctx=config.OLLAMA_NUM_CTX).bind_tools([tool])

    def mind(state: MessagesState) -> dict:
        return {"messages": [llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("mind", mind)
    graph.add_node("body", ToolNode([tool]))
    graph.add_edge(START, "mind")
    graph.add_conditional_edges("mind", tools_condition, {"tools": "body", END: END})
    graph.add_edge("body", "mind")
    return graph.compile()


def user_message(question: str, image_path: str | None = None) -> HumanMessage:
    if image_path:
        question = f"{question}\n\nImage provided by the user: {image_path}"
    return HumanMessage(question)


def run_agent(agent, question: str, image_path: str | None = None) -> Iterator[dict]:
    """Exécute l'agent et publie chaque étape du cycle Reasoning -> Action -> Observation."""
    inputs = {"messages": [user_message(question, image_path)]}
    run_config = {"recursion_limit": config.MAX_AGENT_STEPS}
    for update in agent.stream(inputs, run_config, stream_mode="updates"):
        for node, payload in update.items():
            for message in payload["messages"]:
                if isinstance(message, AIMessage) and message.tool_calls:
                    if message.content:
                        yield {"step": "reasoning", "text": message.content}
                    for call in message.tool_calls:
                        yield {"step": "action", "tool": call["name"], "args": call["args"]}
                elif isinstance(message, ToolMessage):
                    yield {"step": "observation", "text": message.content, "results": message.artifact or []}
                elif isinstance(message, AIMessage):
                    yield {"step": "answer", "text": message.content}
