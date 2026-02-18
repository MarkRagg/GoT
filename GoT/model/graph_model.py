from venv import logger
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from GoT.model.ollama_llm import OllamaLLM
from GoT.model.runtime_graph import (
    BacktrackNode,
    CompletitionNode,
    GoalNode,
    Response,
    RuntimeGraph,
    RuntimeNode,
    Score,
    TestNode,
    ToolNode,
)
from GoT.model.utils.utils import (
    extract_tool_used,
    parse_response,
    parse_response_for_tool_node,
    parse_score,
    parse_tool_list,
    remove_tools_from_list,
)


# Defining agents
starting_agent = OllamaLLM().create_custom_agent(
    OllamaLLM().get_tools(),
    SystemMessage(
        "You are an assistant specialized in tools. Your goal is not to resolve the problem,"
        " only to make list with the best tool to use. "
        "The list MUST be in this format and it is not possible to format the tool_name in any way: "
        "- tool_name "
        "- tool_name "
        "- tool_name "
    ),
)

chat_completition_agent = OllamaLLM().create_custom_agent([])

judge_agent = OllamaLLM().create_custom_agent(
    [],
    SystemMessage(
        "You are an assistant specialized in validation of response, like an LLM-as-a-judge. "
        "Your duty is to score, from 0 to 6, the response that user gives to you and assign to it a score. "
        "You MUST respond ONLY using the Score function. "
        "Do not write natural language outside the function. "
        "If you fail to respect the format, the evaluation will fail."
        "\n0: The response is impossible to understand and completely wrong. "
        "\n1: The response is near to be completely wrong. "
        "\n2: The response is in the correct language but it doesn't follow the instruction. "
        "\n3: The response try to resolve the problem but doesn't follow the instruction or the response is wrong. "
        "\n4: The response follow the instruction but the result is wrong or the result is correct but doesn't follow the instruction. "
        "\n5: The response follow the instruction and the result is near to the solution (If the task is hard, the solution should be near to the corrected one). "
        "\n6: The response follow the instruction and the result is perfectly correct."
    ),
    response_format=Score,
)

# Defining runtime graph
runtime_graph = RuntimeGraph()


def goal(prompt: MessagesState):
    runtime_graph.goal = prompt
    goal_node = GoalNode(parse_response(prompt), resolved=True)
    runtime_graph.add_node(goal_node)
    runtime_graph.temp_node = goal_node
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
    tool_list = parse_tool_list(str_res)  # Toglie elementi inutili
    # add tool nodes in the runtime graph
    for tool in tool_list:
        tool_node = RuntimeNode(resolved=True)
        call_node = ToolNode(
            f"Please, resolve the problem with the tool: {tool}",
            "",
            tool_name=tool,
        )
        runtime_graph.add_node(tool_node)
        runtime_graph.add_edge(
            runtime_graph.temp_node, tool_node
        )  # edge from goal to tool node
        runtime_graph.add_node(call_node)
        runtime_graph.add_edge(tool_node, call_node)
        runtime_graph.add_tool_link(call_node, tool)
    # extract a tool to call
    runtime_graph.temp_node = runtime_graph.call_tool_node()
    return runtime_graph.append_prompt_to_messages_state(runtime_graph.temp_node)


def tool_call(messages: MessagesState):
    # It calls the llm and it resolves the call node
    call_node = runtime_graph.temp_node
    tool_agent = OllamaLLM().create_custom_agent(
        remove_tools_from_list(
            OllamaLLM().get_tools(), runtime_graph.get_resolved_tools()
        ),
        SystemMessage(
            "You are an assistant specialized in tools. Your goal is to resolve the problem with "
            " the tool that the user indicates to you. You MUST use the tool that user indicates to you."
            "You MUST respond ONLY using the Response function. "
            "Do not write natural language outside the function. "
            "If you fail to respect the format, the evaluation will fail."
        ),
        response_format=Response,
    )

    res = tool_agent.invoke(messages)
    tool_used = extract_tool_used(res)
    runtime_graph.temp_response.response = parse_response_for_tool_node(res).response
    parsed_res = f"Response: {parse_response_for_tool_node(res).response}\nExplanation: {parse_response_for_tool_node(res).explanation}"
    runtime_graph.resolve_node(call_node, parsed_res)

    test_node = TestNode(
        f"{parsed_res}",
        "",
        score=0,
        tool_used=tool_used,
    )
    runtime_graph.add_node(test_node)
    runtime_graph.add_edge(call_node, test_node)
    runtime_graph.temp_node = test_node
    messages["messages"].append(AIMessage(parsed_res))
    return messages


def response_evaluation(messages: MessagesState):
    # Get the actual tool execution result from the resolved call_node
    test_node = runtime_graph.temp_node
    if not isinstance(test_node, TestNode):
        raise TypeError("Expected TestNode for scoring")
    call_node_response = test_node.prompt  # The actual solution to judge

    # Create a proper message for the judge with the solution
    judge_messages = [
        HumanMessage(content="Original task:\n" + parse_response(runtime_graph.goal)),
        HumanMessage(content="Solution:\n" + call_node_response),
        SystemMessage(
            content="Score this solution based on correctness and following instructions."
        ),
        AIMessage(
            content="Tool used in the response: " + ", ".join(test_node.tool_used)
        ),
    ]

    score_res = parse_score(judge_agent.invoke({"messages": judge_messages}))
    test_node.score = score_res.score
    runtime_graph.resolve_node(test_node, score_res.description)

    return messages


def test_result(messages: MessagesState):
    n = runtime_graph.exist_tool_available()
    test_node = runtime_graph.temp_node
    if not isinstance(test_node, TestNode):
        raise TypeError("Expected TestNode for scoring")

    if test_node.score >= 5:
        runtime_graph.add_edge(test_node, runtime_graph.temp_response)
        runtime_graph.temp_response.resolved = True
        return END
    elif test_node.score < 5 and n is True:
        return "backtrack"
    else:
        chat_completition_node = CompletitionNode(
            "Please, solve this problem",
            "",
        )
        runtime_graph.add_node(chat_completition_node)
        runtime_graph.add_edge(test_node, chat_completition_node)
        runtime_graph.temp_node = chat_completition_node
        return "chat_completition"


def backtrack(messages: MessagesState):
    test_node = runtime_graph.temp_node
    if not isinstance(test_node, TestNode):
        raise TypeError("Expected TestNode for backtracking")
    backtrack_node = BacktrackNode(feedback=test_node.response, resolved=True)
    runtime_graph.add_node(backtrack_node)
    runtime_graph.add_edge(test_node, backtrack_node)
    runtime_graph.temp_node = runtime_graph.call_tool_node()
    runtime_graph.add_edge(
        backtrack_node, runtime_graph.temp_node
    )  # tool call node that we want to resolve
    messages = runtime_graph.append_prompt_to_messages_state(runtime_graph.temp_node)
    messages.get("messages", []).append(AIMessage(backtrack_node.feedback))
    return messages


def chat_completition(messages: MessagesState):
    new_messages_history = runtime_graph.goal
    if isinstance(runtime_graph.temp_node, CompletitionNode):
        new_messages_history["messages"].append(
            AIMessage(content=runtime_graph.temp_node.prompt)
        )
    result = parse_response(chat_completition_agent.invoke(new_messages_history))
    runtime_graph.resolve_node(runtime_graph.temp_node, result)
    runtime_graph.temp_response.response = result
    runtime_graph.temp_response.resolved = True
    new_messages_history["messages"].append(AIMessage(content=result))
    return new_messages_history


# https://docs.langchain.com/oss/python/langgraph/overview

content = ""


def call_graph(prompt: str):
    global content
    content = prompt
    return invoke_graph()


def invoke_graph():
    graph = StateGraph(MessagesState)
    graph.add_node(goal)
    graph.add_node(tool_expand)
    graph.add_node(tool_call)
    graph.add_node(backtrack)
    graph.add_node(chat_completition)
    graph.add_node(response_evaluation)
    graph.add_edge(START, "goal")
    graph.add_edge("goal", "tool_expand")
    graph.add_edge("tool_expand", "tool_call")
    graph.add_edge("tool_call", "response_evaluation")
    graph.add_edge("backtrack", "tool_call")
    graph.add_edge("chat_completition", END)
    graph.add_conditional_edges(
        "response_evaluation",
        test_result,
        {"backtrack": "backtrack", "chat_completition": "chat_completition", END: END},
    )

    graph = graph.compile()

    # logger.info(graph.get_graph().draw_mermaid())

    res = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        }
    )

    res["output"] = runtime_graph.temp_response.response

    # logger.info(res)
    print(runtime_graph.print_mermaid())
    runtime_graph.clear()
    return res
