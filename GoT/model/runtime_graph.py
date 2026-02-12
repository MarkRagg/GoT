from typing import Dict, List
from langgraph.graph import MessagesState
from langchain_core.messages import AnyMessage
from pydantic import BaseModel


class RuntimeNode:
    _id_counter = 0  # Contatore globale per ID unici

    def __init__(
        self,
        resolved: bool = False,
    ):
        self.id = RuntimeNode._id_counter  # ID unico per ogni nodo
        RuntimeNode._id_counter += 1
        self.resolved = resolved

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, RuntimeNode):
            return False
        return self.id == other.id

    def __repr__(self):
        return f"RuntimeNode(id={self.id}, resolved={self.resolved})"


class TestNode(RuntimeNode):
    def __init__(
        self,
        prompt: str,
        response: str,
        score: int,
        tool_used: List[str] = [],
        resolved: bool = False,
    ):
        super().__init__(resolved)
        self.prompt = prompt
        self.response = response
        self.score = score
        self.tool_used = tool_used


class ToolNode(RuntimeNode):
    def __init__(
        self,
        prompt: str,
        response: str,
        tool_name: str,
        resolved: bool = False,
    ):
        super().__init__(resolved)
        self.prompt = prompt
        self.response = response
        self.tool_name = tool_name


class GoalNode(RuntimeNode):
    def __init__(
        self,
        prompt: str,
        resolved: bool = False,
    ):
        super().__init__(resolved)
        self.prompt = prompt


class BackTrackNode(RuntimeNode):
    def __init__(
        self,
        feedback: str,
        resolved: bool = True,
    ):
        super().__init__(resolved)
        self.feedback = feedback


class Score(BaseModel):
    """Rapresents a score for a test node.

    Attributes:
        score: int - The score assigned to the test node.
        description: str - A description or rationale for the assigned score.
    """

    score: int
    description: str


class RuntimeGraph:
    def __init__(self):
        self.goal: MessagesState = MessagesState(messages=[])
        self.nodes: Dict[RuntimeNode, List[RuntimeNode]] = {}
        self.tools_available: Dict[RuntimeNode, str] = {}
        self.temp_node: RuntimeNode = RuntimeNode()

    def add_node(self, node: RuntimeNode):
        self.nodes.setdefault(node, [])

    def add_edge(self, n1: RuntimeNode, n2: RuntimeNode):
        self.nodes.setdefault(n1, []).append(n2)
        self.nodes.setdefault(n2, [])

    def add_tool_link(self, call_node: RuntimeNode, tool_name: str):
        self.tools_available.setdefault(call_node, tool_name)

    def resolve_node(self, node: RuntimeNode, response: str) -> None:
        node.response = response
        node.resolved = True

    def call_tool_node(self) -> RuntimeNode:  # TODO change name
        nodes = list(self.nodes.keys())
        call_nodes = [n for n in nodes if (isinstance(n, ToolNode) and not n.resolved)]
        return call_nodes[0]

    def exist_tool_available(self) -> bool:
        nodes = list(self.nodes.keys())
        call_nodes = [n for n in nodes if (isinstance(n, ToolNode) and not n.resolved)]
        return True if call_nodes else False

    def get_resolved_tools(self):
        resolved_nodes = [t for t in self.tools_available.keys() if t.resolved is True]
        return [self.tools_available[n] for n in resolved_nodes]

    def runtime_node_to_state(
        self, node: RuntimeNode
    ) -> MessagesState:  # TODO change name
        messages: list[AnyMessage] = []

        if node.prompt:
            messages.append(node.prompt)

        return MessagesState(messages=messages)

    def print_mermaid(self) -> str:
        lines = []

        lines.append("---")
        lines.append("config:")
        lines.append("  flowchart:")
        lines.append("    curve: linear")
        lines.append("---")
        lines.append("graph TD;")

        # Nodi (rettangoli con angoli arrotondati)
        for node in self.nodes:
            node_type = type(node).__name__
            label = f"<b>{node.id}</b><br/>{node_type}<br/>"
            if isinstance(node, TestNode):
                label += f"Score: {node.score}<br/>"
            lines.append(f'    {node.id}("{label}");')

        # Edge
        for n1, children in self.nodes.items():
            for n2 in children:
                lines.append(f"    {n1.id} --> {n2.id};")

        # Stili
        lines.append("")
        lines.append(
            "    classDef resolved fill:#b7f7c0,stroke:#2ecc71,stroke-width:2px;"
        )
        lines.append(
            "    classDef rejected fill:#f7b7b7,stroke:#e74c3c,stroke-width:2px;"
        )
        lines.append(
            "    classDef unresolved fill:#e0e0e0,stroke:#9e9e9e,stroke-width:2px;"
        )

        for node in self.nodes:
            cls = self.__get_color(node)
            lines.append(f"    class {node.id} {cls};")

        return "\n".join(lines)

    def __get_color(self, node: RuntimeNode) -> str:
        if isinstance(node, TestNode) and node.score < 5:
            return "rejected"
        elif node.resolved:
            return "resolved"
        else:
            return "unresolved"
