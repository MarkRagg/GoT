from langchain.messages import HumanMessage

from GoT.core.graph_model import call_graph
from GoT.core.llm import LLM


def custom_test(text: str, is_graph_mode: bool):
    if not is_graph_mode:
        call_graph(text)
    else:
        agent = LLM().create_custom_agent(LLM().get_tools())
        agent.invoke(
            {"messages": [HumanMessage(content=text)]},
            config={"recursion_limit": 20},
        )