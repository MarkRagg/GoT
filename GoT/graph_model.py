from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from GoT.ollama_llm import OllamaLLM
from GoT.runtime_graph import RuntimeGraph, RuntimeNode, TestNode
from GoT.utils import parse_response, parse_score, parse_tool_list, remove_tools_from_list

load_dotenv()

# Defining agents
starting_agent = OllamaLLM().create_custom_agent(OllamaLLM().get_tools(), 
                "You are an assistant specialized in tools. Your goal is not to resolve the problem," \
                " only to make list with the best tool to use. " \
                "The list MUST be in this format and it is not possible to format the tool_name in any way: " \
                "- tool_name " \
                "- tool_name " \
                "- tool_name ")

chat_completition_agent = OllamaLLM().create_custom_agent([])

judge_agent = OllamaLLM().create_custom_agent([], 
                "You are an assistant specialized in validation of response, like an LLM-as-a-judge. " \
                "Your duty is to score, from 0 to 6, the response that user gives to you and assign to it a score. " \
                "Your response MUST be a description of the score and then the" \
                "output format should be: " \
                "Score: <number> \n" \
                "0: The response is impossible to understand and completely wrong. " \
                "1: The response is near to be completely wrong. " \
                "2: The response is in the correct language but it doesn't follow the instruction. " \
                "3: The response try to resolve the problem but doesn't follow the instruction or the response is wrong. " \
                "4: The response follow the instruction but the result is wrong or the result is correct but doesn't follow the instruction. " \
                "5: The response follow the instruction and the result is near to the solution (If the task is hard, the solution should be near to the corrected one). " \
                "6: The response follow the instruction and the result is perfectly correct.")

# Defining runtime graph
runtime_graph = RuntimeGraph("")

def goal(prompt: MessagesState): 
    runtime_graph.goal = prompt
    return prompt

def tool_expand(goal: MessagesState):
    msg = parse_response(goal)
    sys_msg = "Which tools that I have can I use to solve this problem? Please make a list using '-' to denote each tool in a probabilistic order, don't use this character for other reasons."
    messages = [
        HumanMessage(msg),
        SystemMessage(sys_msg),
    ]
    res = starting_agent.invoke({"messages": messages})
    str_res = parse_response(res)
    tool_list = parse_tool_list(str_res) # Toglie elementi inutili
    # add tool nodes in the runtime graph
    for tool in tool_list:
        tool_node = RuntimeNode(SystemMessage(sys_msg), AIMessage(tool), type="tool", resolved = True)
        call_node = RuntimeNode(SystemMessage("Please, resolve the problem with the tool: " + tool + ". You can't use other tools!"), AIMessage(""), type="call_tool")
        runtime_graph.add_node(tool_node)
        runtime_graph.add_node(call_node)
        runtime_graph.add_edge(tool_node, call_node)
        runtime_graph.add_tool_link(call_node, tool)
    # extract a tool to call
    runtime_graph.temp_node = runtime_graph.call_tool_node()
    return runtime_graph.runtime_node_to_state(runtime_graph.temp_node)

def tool_call(messages: MessagesState):  
    # It calls the llm and it resolves the call node 
    call_node = runtime_graph.temp_node
    tool_agent = OllamaLLM().create_custom_agent(remove_tools_from_list(OllamaLLM().get_tools(), runtime_graph.get_resolved_tools()), 
                            "You are an assistant specialized in tools. Your goal is to resolve the problem with " \
                            " the tool that the user indicates to you. You MUST use the tool that user indicates to you. ")
    res = parse_response(tool_agent.invoke(messages))
    runtime_graph.resolve_node(call_node, AIMessage(res))

    # Add test node
    test_node = TestNode(SystemMessage("Score this solution: \n" + res), AIMessage(""), type="test", score=0)
    runtime_graph.add_node(test_node)
    runtime_graph.add_edge(call_node, test_node)
    runtime_graph.temp_node = test_node
    return runtime_graph.runtime_node_to_state(runtime_graph.temp_node)
    
def test_result(result_msg: MessagesState): # TODO: Pensare ad un modo per testare
    n = runtime_graph.call_tool_node()
    
    # Get the actual tool execution result from the resolved call_node
    test_node = runtime_graph.temp_node
    call_node_response = test_node.prompt.content  # The actual solution to judge
    
    # Create a proper message for the judge with the solution
    judge_messages = [
        HumanMessage(content=call_node_response),
        SystemMessage(content="Score this solution based on correctness and following instructions. Remember that you can't verify the usage of the tools.")
    ]
    
    score_res = parse_response(judge_agent.invoke({"messages": judge_messages}))
    test_node.score = parse_score(score_res)
    runtime_graph.resolve_node(test_node, AIMessage(score_res))

    if test_node.score < 5 and n is not None:
        return "backtrack"
    elif n is None:
        chat_completition_node = RuntimeNode(SystemMessage("Please, solve this problem"), AIMessage(""), type="chat_completition")
        runtime_graph.add_node(chat_completition_node)
        runtime_graph.temp_node = chat_completition_node
        return "chat_completition"
    else:
        return END

def backtrack(messages: MessagesState):
    runtime_graph.temp_node = runtime_graph.call_tool_node()
    return runtime_graph.runtime_node_to_state(runtime_graph.temp_node)

def chat_completition(messages: MessagesState):
    new_messages_history = runtime_graph.goal
    new_messages_history["messages"].append(runtime_graph.temp_node.prompt)
    chat_completition_agent.invoke(new_messages_history)
    return new_messages_history

# https://docs.langchain.com/oss/python/langgraph/overview

def invoke_graph():
    graph = StateGraph(MessagesState)
    graph.add_node(goal)
    graph.add_node(tool_expand)
    graph.add_node(tool_call)
    graph.add_node(backtrack)
    graph.add_node(chat_completition)
    graph.add_edge(START, "goal")
    graph.add_edge("goal", "tool_expand")
    graph.add_edge("tool_expand", "tool_call")
    graph.add_conditional_edges("tool_call", test_result)
    graph.add_edge("backtrack", "tool_call")
    graph.add_edge("chat_completition", END)

    graph = graph.compile()

    res = graph.invoke({"messages": [{"role": "user", "content": "Solve 100 + 100 + 200 + 400"}]})

    # logger.info(res)
    # logger.info(graph.get_graph().draw_mermaid())

    return res
