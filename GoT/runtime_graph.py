from typing import Dict, List
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, SystemMessage

from GoT.utils import parse_response

class RuntimeNode:
    _id_counter = 0  # Contatore globale per ID unici
    
    def __init__(self, prompt: SystemMessage, response: AIMessage, type: str, resolved: bool = False):
        self.id = RuntimeNode._id_counter  # ID unico per ogni nodo
        RuntimeNode._id_counter += 1    
        self.prompt = prompt
        self.response = response
        self.type = type
        self.resolved = resolved

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, RuntimeNode):
            return False
        return self.id == other.id
    
    def __repr__(self):
        return f"RuntimeNode(id={self.id}, type={self.type}, resolved={self.resolved})"

class TestNode(RuntimeNode):
    def __init__(self, prompt: SystemMessage, response: AIMessage, type: str, score: int, resolved: bool = False):
        super().__init__(prompt, response, type, resolved)
        self.score = score

class RuntimeGraph:
    def __init__(self, goal: MessagesState):
        self.goal = goal
        self.nodes: Dict[RuntimeNode, List[RuntimeNode]] = {}
        self.tools_available: Dict[RuntimeNode, str] = {}
        self.temp_node = None

    def add_node(self, node: RuntimeNode): 
        self.nodes.setdefault(node, [])

    def add_edge(self, n1: RuntimeNode, n2: RuntimeNode): 
        self.nodes.setdefault(n1, []).append(n2)
        self.nodes.setdefault(n2, [])

    def add_tool_link(self, call_node: RuntimeNode, tool_name: str): 
        self.tools_available.setdefault(call_node, tool_name)

    def resolve_node(self, node: RuntimeNode, response: AIMessage) -> None:
        node.response = response
        node.resolved = True

    def call_tool_node(self) -> RuntimeNode: # TODO change name
        nodes = list(self.nodes.keys())
        call_nodes = [n for n in nodes if (n.type.startswith("call_tool") and not n.resolved)]
        return call_nodes[0] if call_nodes else None
    
    def get_resolved_tools(self):
        resolved_nodes = [t for t in self.tools_available.keys() if t.resolved is True]
        return [self.tools_available[n] for n in resolved_nodes]

    def runtime_node_to_state(self, node: RuntimeNode) -> MessagesState: # TODO change name
        messages = []

        if node.prompt:
            messages.append(node.prompt)

        return MessagesState(messages=messages)