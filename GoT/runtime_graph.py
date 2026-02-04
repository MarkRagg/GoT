from typing import List
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from GoT.utils import parse_response

class RuntimeNode:
    def __init__(self, prompt: SystemMessage, response: AIMessage, type: str, resolved: bool = False):
        self.prompt = prompt
        self.response = response
        self.type = type
        self.resolved = resolved

    def __hash__(self):
        return hash((
            getattr(self.prompt, "content", None),
            getattr(self.response, "content", None),
            self.type,
        ))

    def __eq__(self, other):
        if not isinstance(other, RuntimeNode):
            return False
        return (
            getattr(self.prompt, "content", None) == getattr(other.prompt, "content", None)
            and getattr(self.response, "content", None) == getattr(other.response, "content", None)
            and self.type == other.type
        )

class RuntimeGraph:
    def __init__(self, goal: MessagesState):
        self.goal = goal
        self.nodes = {}
        self.temp_node = None

    def add_node(self, node: RuntimeNode): 
        self.nodes.setdefault(node, [])

    def add_edge(self, n1: RuntimeNode, n2: RuntimeNode): 
        self.nodes.setdefault(n1, []).append(n2)
        self.nodes.setdefault(n2, [])

    def resolve_node(self, node: RuntimeNode, response: AIMessage) -> None:
        node.response = response
        node.resolved = True

    def call_tool_node(self) -> RuntimeNode:
        nodes = list(self.nodes.keys())
        call_nodes = [n for n in nodes if n.type.startswith("call_tool")]
        return call_nodes[0] if call_nodes else None
    
    def runtime_node_to_state(self, node: RuntimeNode) -> MessagesState:
        messages = []

        messages.append(HumanMessage(self.goal["messages"][-1].content))

        if node.prompt:
            messages.append(node.prompt)

        return MessagesState(messages=messages)