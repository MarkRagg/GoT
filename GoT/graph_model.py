from venv import logger

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from GoT.ollama_llm import OllamaLLM
from GoT.runtime_graph import RuntimeGraph, RuntimeNode

load_dotenv()

# Defining LLM
llm = OllamaLLM().ollamaLLM
# Defining runtime graph
runtime_graph = RuntimeGraph("")

def goal(prompt: MessagesState):  # "calculate 2+2
    return prompt

def expand(goal: MessagesState):
    msg = goal["messages"][-1].content
    messages = [
        HumanMessage(msg),
        AIMessage(
            "Which tools can I use to solve this problem? Please make a list using '-' to denote each tool."
        ),
    ]
    res = llm.invoke(messages)

    runtime_graph.add_node(RuntimeNode(MessagesState(res), type="tool"))
    # parse res into 3 parts
    # add 3 nodes to the runtime graph
    return res

# https://docs.langchain.com/oss/python/langgraph/overview

def invoke_graph():
    graph = StateGraph(MessagesState)
    graph.add_node(goal)
    graph.add_node(expand)
    graph.add_edge(START, "goal")
    graph.add_edge("goal", "expand")

    graph = graph.compile()

    res = graph.invoke({"messages": [{"role": "user", "content": "calculate 2+2"}]})

    logger.info(res)
    logger.info(graph.get_graph().draw_mermaid())

    return res
