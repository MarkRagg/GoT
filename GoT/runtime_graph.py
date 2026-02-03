from typing import List
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage

class RuntimeNode:
    def __init__(self, msg: AIMessage, type: str):
        self.msg = msg
        self.type = type

class RuntimeGraph:
    def __init__(self, goal: MessagesState):
        self.goal = goal
        self.nodes = {}

    def add_node(self, node: RuntimeNode): 
        print(node.msg["content"])
        self.nodes.setdefault(node, [])

    def add_edge(self, n1, n2): 
        self.nodes.setdefault(n1, []).append(n2)
        self.nodes.setdefault(n2, [])