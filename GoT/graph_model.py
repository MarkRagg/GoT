from venv import logger

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from GoT.ollama_llm import OllamaLLM
from GoT.runtime_graph import RuntimeGraph, RuntimeNode
from GoT.utils import parse_response

load_dotenv()

# Defining LLM
agent = OllamaLLM().agent
# Defining runtime graph
runtime_graph = RuntimeGraph("")

def goal(prompt: MessagesState): 
    runtime_graph.goal = prompt
    return prompt

def tool_expand(goal: MessagesState):
    msg = parse_response(goal)
    sys_msg = "Which tools that I have can I use to solve this problem? Please make a list using '-' to denote each tool, don't use this character for other reasons."
    messages = [
        HumanMessage(msg),
        SystemMessage(sys_msg),
    ]
    res = agent.invoke({"messages": messages})
    str_res = parse_response(res)
    # print("[INFO]: " + str_res)
    tool_list = [p.strip() for p in str_res.split("-", 3) if p.strip()] # Toglie elementi inutili

    # add tool nodes in the runtime graph
    for tool in tool_list:
        tool_node = RuntimeNode(SystemMessage(sys_msg), AIMessage(tool), type="tool", resolved = True)
        call_node = RuntimeNode(SystemMessage("Please, resolve the problem with the tool: " + tool + ". You can't use other tools!"), AIMessage(""), type="call_tool")
        runtime_graph.add_node(tool_node)
        runtime_graph.add_node(call_node)
        runtime_graph.add_edge(tool_node, call_node)

    # extract a tool to call
    runtime_graph.temp_node = runtime_graph.call_tool_node()
    return runtime_graph.runtime_node_to_state(runtime_graph.temp_node)

def tool_call(messages: MessagesState):  
    # It calls the llm and it resolves the call node 
    call_node = runtime_graph.temp_node
    res = parse_response(agent.invoke(messages))
    runtime_graph.resolve_node(call_node, AIMessage(res))

    # Add test node
    test_node = RuntimeNode(SystemMessage("Is this solution correct? \n" + res), AIMessage(""), type="test")
    runtime_graph.add_node(test_node)
    runtime_graph.add_edge(call_node, test_node)
    runtime_graph.temp_node = test_node
    return runtime_graph.runtime_node_to_state(test_node)
    
def test_result(result_msg: MessagesState): # TODO: Pensare ad un modo per testare
    res = parse_response(result_msg)
    if len(res) > 0:
        return "backtrack"
    else:
        return END

def backtrack(messages: MessagesState):
    runtime_graph.temp_node = runtime_graph.call_tool_node()
    return runtime_graph.runtime_node_to_state(runtime_graph.temp_node)

# https://docs.langchain.com/oss/python/langgraph/overview

def invoke_graph():
    graph = StateGraph(MessagesState)
    graph.add_node(goal)
    graph.add_node(tool_expand)
    graph.add_node(tool_call)
    graph.add_node(backtrack)
    graph.add_edge(START, "goal")
    graph.add_edge("goal", "tool_expand")
    graph.add_edge("tool_expand", "tool_call")
    graph.add_conditional_edges("tool_call", test_result)
    graph.add_edge("backtrack", "tool_call")

    graph = graph.compile()

    res = graph.invoke({"messages": [{"role": "user", "content": "Solve 100 + 100 + 200 + 400"}]})

    # logger.info(res)
    # logger.info(graph.get_graph().draw_mermaid())

    return res
