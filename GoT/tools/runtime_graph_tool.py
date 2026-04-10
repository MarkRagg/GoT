from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool

from GoT.model.ollama_llm import LLM
from GoT.model.runtime_graph import ReasoningNode, RuntimeGraph
from GoT.model.utils.utils import parse_response

MAX_INTERACTIONS = 10

@tool
def divide_thought(first_part: str, second_part: str, first_context: str, second_context: str, reasoning_type: str = "pure_reasoning") -> str:
    """
    This is a tool to divide the thought process into smaller steps.
    HOW TO USE THIS TOOL:
    - Call it when you think the problem is complex.
    - The two parts must be as independent as possible.
     Arguments:
    - first_part: the first part of the thought process
    - second_part: the second part of the thought process
    - first_context: the context related to the first part
    - second_context: the context related to the second part
    - reasoning_type: the type of reasoning to use for each part, it can be "pure_reasoning" or "tool_use"
    """
    tool_agent = LLM().create_custom_agent(
        LLM().get_tools(),
        SystemMessage(
            "You are an assistant specialized in tools. Your goal is to resolve the problem with "
            " the tool that the user indicates to you. You should to use the tool that the assistant indicates to you."
            "Do not write natural language outside the function. "
            "If you fail to respect the format, the evaluation will fail."
        ),
    )
    reasoning_agent = LLM().create_custom_agent(
        [],
        SystemMessage(
            "You are an assistant specialized in reasoning. " \
            "Your goal is to resolve the problem with reasoning. You should to reason step by step and write all your reasoning. " \
            "If the problem is too complex, you can divide it into smaller parts."
        ),
    )
    runtime_graph = RuntimeGraph()
    n1 = ReasoningNode("")
    n2 = ReasoningNode("")
    runtime_graph.add_node(n1)
    runtime_graph.add_node(n2)
    runtime_graph.add_edge(runtime_graph.temp_node, n1)
    runtime_graph.add_edge(runtime_graph.temp_node, n2)
    msg1 = [HumanMessage("Reason aboout this problem: " + first_part + "\nContext: " + first_context)]
    msg2 = [HumanMessage("Reason aboout this problem: " + second_part + "\nContext: " + second_context)]
    if reasoning_type == "pure_reasoning":
        res1 = parse_response(reasoning_agent.invoke({"messages": msg1}, config={"recursion_limit": MAX_INTERACTIONS}))
        res2 = parse_response(reasoning_agent.invoke({"messages": msg2}, config={"recursion_limit": MAX_INTERACTIONS}))
    else:
        res1 = parse_response(tool_agent.invoke({"messages": msg1}, config={"recursion_limit": MAX_INTERACTIONS}))
        res2 = parse_response(tool_agent.invoke({"messages": msg2}, config={"recursion_limit": MAX_INTERACTIONS}))
    runtime_graph.resolve_node(n1, res1)
    runtime_graph.resolve_node(n2, res2)

    result = f"First part: {res1}\nSecond part: {res2}"
    return result