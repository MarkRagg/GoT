from venv import logger
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from GoT.model.ollama_llm import LLM
from GoT.model.runtime_graph import (
    BacktrackNode,
    CompletitionNode,
    CraftingNode,
    GoalNode,
    ReasoningNode,
    Response,
    RuntimeGraph,
    Score,
    TestNode,
    ToolNode,
)
from GoT.model.utils.utils import (
    extract_tool_used,
    parse_response,
    parse_response_for_tool_node,
    parse_score,
)
from GoT.tools.runtime_graph_tool import divide_thought

SCORE_THRESHOLD = 5
COMPLEXITY_COEFFICIENT = 0.5
COMPLEXITY_THRESHOLD = 5.5
MAX_INTERACTIONS = 10

# Defining agents
starting_agent = LLM().create_custom_agent(
    LLM().get_tools(),
    SystemMessage(
        "You are an assistant specialized in tools. "
        " Your goal is not to resolve the problem, but only to make list with the best tool to use. "
        " If you are not sure of the tool you have, think also a generic tool to craft that could be useful to solve the problem (Specify the craft is needed). "
        " If the problem require too much steps with the current tools, consider crafting a new tool."
        "The list MUST be in this format and it is not possible to format the tool_name in any way: "
        "- tool_name: explanation of why this tool is useful in this problem "
        "- tool_name: explanation of why this tool is useful in this problem "
        "- tool_name: explanation of why this tool is useful in this problem "
    ),
    type="remote_standard",
)

chat_completition_agent = LLM().create_custom_agent([])

judge_agent = LLM().create_custom_agent(
    [],
    SystemMessage(
        """
        You are an assistant specialized in validating responses, like an LLM-as-a-judge.
        Your duty is to score, from 0 to 5, the response that user gives and assign a score.

        Rules:
        - You MUST respond ONLY using the Score function.
        - You must consider if the format of the answer follow the instruction
        - You cannot give the full solution, only hints.
        - If a response suggest the need of crafting a tool, score it with 1 or less and specify clearly the need of a new tool to solve the problem.
        - Do not write natural language outside the function.
        - Always consider creating a tool if it makes the response correct or reusable.

        Score meanings:
        0: Impossible to understand / completely wrong
        1: Nearly completely wrong
        2: Correct language but does not follow instruction
        3: Tries to solve but fails instruction / wrong
        4: Follows instruction but result wrong or incomplete
        5: Follows instruction, result correct

        Example:
        - User response: cannot compute because missing helper
        - Hint: implement a tool that computes the missing function
        """
    ),
    response_format=Score,
    type="remote_score_format",
)

crafter_agent = LLM().create_custom_agent(
    LLM().get_craft_tool(),
    SystemMessage(
        """
        You are a master coder specialized in crafting tools for other agents.
        You create reusable Python tools for other agents.

        The tool must be GENERAL and parameterized.
        Never hardcode values from the current problem.

        Bad example (too specific):
        def multiply_3_and_7():
            return 3 * 7

        Good example (general):
        def multiply(a: float, b: float) -> float:
        '
            Arguments:
            a: the first number to multiply
            b: the second number to multiply
            Returns:
            The product of a and b
        '
            return a * b

        Rules:
        - Prefer generic names and parameters, never craft specific functions.
        - If the function contains specific numbers or values, it is wrong.
        - Craft only one function, it must contains always the docs.
        - Never craft tool that raise exceptions.
        - Respond ONLY using the tool available.
        - No natural language.
        - No comments in the python interpreter.
        """
    ),
    response_format=Response,
    type="remote_response_format",
)

reasoning_agent = LLM().create_custom_agent(
    [divide_thought],
    SystemMessage(
        "You are an assistant specialized in divide the complexity. "
        "Your goal is to resolve the problem with reasoning. You should to reason step by step and write all your reasoning. "
        "You are forced to divide it into smaller parts."
        "The parts must be indipendent from each other and must be solvable at the same time."
        "Respond with the indicated format."
    ),
    response_format=Response,
    type="remote_response_format",
)

# Defining runtime graph
runtime_graph = RuntimeGraph()


def goal(prompt: MessagesState):
    runtime_graph.goal = prompt
    goal_node = GoalNode(parse_response(prompt), resolved=True)
    runtime_graph.add_node(goal_node)
    runtime_graph.temp_node = goal_node
    for i in range(0, 3):
        call_node = ToolNode(
            "Please, resolve the problem with the tools given, you MUST follow the previous reasoning.",
            "",
            tool_name="",
        )
        reasoning_node = ReasoningNode("")
        runtime_graph.add_node(reasoning_node)
        runtime_graph.add_node(call_node)
        runtime_graph.add_edge(goal_node, reasoning_node)
        runtime_graph.add_edge(reasoning_node, call_node)
    reasoning_node = ReasoningNode("")
    runtime_graph.add_node(reasoning_node)
    runtime_graph.add_edge(goal_node, reasoning_node)
    # extract a reasoning node to resolve
    runtime_graph.temp_node = runtime_graph.call_tool_node()
    return prompt


# def tool_expand(goal: MessagesState):
#     msg = parse_response(goal)
#     sys_msg = "Please make a list using '-' to denote each tool in a probabilistic order, don't use this character for other reasons. Select only the tool(s) you want to use to solve this problem."
#     messages = [
#         HumanMessage(msg),
#         SystemMessage(sys_msg),
#     ]
#     res = starting_agent.invoke({"messages": messages}, config={"recursion_limit": MAX_INTERACTIONS})
#     str_res = parse_response(res)
#     goal["messages"].append(AIMessage(content=str_res))
#     # tool_list = parse_tool_list(str_res)  # Toglie elementi inutili
#     # add tool nodes in the runtime graph

#     return goal


def tool_reasoning(messages: MessagesState):
    messages["messages"].append(
        HumanMessage(
            "Please, reason step by step about how to use these tools to solve the problem, without solving it. If the main solution require to craft a tool, only explain how to craft the tool"
        )
    )
    result = parse_response(
        starting_agent.invoke(messages, config={"recursion_limit": MAX_INTERACTIONS})
    )
    messages["messages"].append(AIMessage(result))
    runtime_graph.resolve_node(runtime_graph.temp_node, result)
    runtime_graph.temp_node = runtime_graph.nodes.get(runtime_graph.temp_node, [])[0]
    if not isinstance(runtime_graph.temp_node, ToolNode):
        raise TypeError("Expected ToolNode after reasoning")
    messages["messages"].append(SystemMessage(runtime_graph.temp_node.prompt))
    return messages


def tool_call(messages: MessagesState):
    # It calls the llm and it resolves the call node
    call_node = runtime_graph.temp_node
    tool_agent = LLM().create_custom_agent(
        LLM().get_tools() + [divide_thought],
        SystemMessage(
            "You are an assistant specialized in tools. Your goal is to resolve the problem with "
            " the tool that the user indicates to you. You HAVE to use or craft the tool that the assistant indicates to you."
            "Do not write natural language outside the function. "
            "If you fail to respect the format, the evaluation will fail."
        ),
        response_format=Response,
        type="remote_response_format",
    )
    try:
        res = tool_agent.invoke(
            {"messages": messages["messages"], "tool_choice": Response},
            config={"recursion_limit": MAX_INTERACTIONS},
        )
    except Exception:
        message = [
            HumanMessage(
                content="Original task:\n" + parse_response(runtime_graph.goal)
            ),
            HumanMessage(
                content="Divide the problem into smaller parts, you can't call the same tool twice."
            ),
        ]
        tool_agent = LLM().create_custom_agent(
            [divide_thought],
            SystemMessage(
                "You are an assistant specialized in tools. Your goal is to resolve the problem with "
                " the tool that the user indicates to you. You HAVE to use the tool that the assistant indicates to you."
                "Do not write natural language outside the function. "
                "If you fail to respect the format, the evaluation will fail."
            ),
            response_format=Response,
            type="remote_response_format",
        )
        res = tool_agent.invoke(
            {"messages": message, "tool_choice": Response},
            config={"recursion_limit": MAX_INTERACTIONS},
        )
    tool_used = extract_tool_used(res)
    runtime_graph.temp_response.response = parse_response_for_tool_node(res).response
    runtime_graph.temp_response.explanation = parse_response_for_tool_node(res).explanation
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
        HumanMessage(content=call_node_response),
        SystemMessage(
            content="Score this solution based on correctness and following instructions."
        ),
        AIMessage(
            content="Tool used in the response: " + ", ".join(test_node.tool_used)
        ),
    ]

    score_res = parse_score(
        judge_agent.invoke(
            {"messages": judge_messages, "tool_choice": Score},
            config={"recursion_limit": MAX_INTERACTIONS},
        )
    )
    test_node.score = score_res.score
    test_node.need_tool_crafting = score_res.need_tool_crafting
    test_node.problem_complexity = score_res.problem_complexity
    runtime_graph.resolve_node(test_node, score_res.description)
    return messages


def crafting(messages: MessagesState):
    crafting_node = CraftingNode(response="", tool_crafted="", resolved=False)
    runtime_graph.add_node(crafting_node)
    runtime_graph.add_edge(runtime_graph.temp_node, crafting_node)
    runtime_graph.temp_node = crafting_node
    ai_feedback = runtime_graph.temp_response.explanation
    crafting_messages = [
        HumanMessage(content="Original task:\n" + parse_response(runtime_graph.goal)),
        AIMessage(content=ai_feedback),
        SystemMessage(
            content="Craft a tool to solve this problem using craft_tool. It must be a function"
        ),
    ]
    craft_res = crafter_agent.invoke(
        {"messages": crafting_messages}, config={"recursion_limit": MAX_INTERACTIONS}
    )
    runtime_graph.temp_response.response = parse_response_for_tool_node(
        craft_res
    ).response
    parsed_res = f"Response: {parse_response_for_tool_node(craft_res).response}\nExplanation: {parse_response_for_tool_node(craft_res).explanation}"
    runtime_graph.resolve_node(crafting_node, parsed_res)
    runtime_graph.temp_node = runtime_graph.call_tool_node()
    runtime_graph.add_edge(crafting_node, runtime_graph.temp_node)
    global starting_agent
    starting_agent = LLM().create_custom_agent(
        LLM().get_tools(),
        SystemMessage(
            "You are an assistant specialized in tools. Your goal is not to resolve the problem,"
            " only to make list with the best tool to use. "
            "The list MUST be in this format and it is not possible to format the tool_name in any way: "
            "- tool_name "
            "- tool_name "
            "- tool_name "
        ),
        type="remote_standard",
    )
    messages["messages"].append(AIMessage(parsed_res))
    return messages


def test_result(messages: MessagesState):
    n = runtime_graph.exist_tool_available()
    test_node = runtime_graph.temp_node
    if not isinstance(test_node, TestNode):
        raise TypeError("Expected TestNode for scoring")

    if test_node.score >= (
        COMPLEXITY_THRESHOLD - COMPLEXITY_COEFFICIENT * test_node.problem_complexity
    ):
        runtime_graph.add_edge(test_node, runtime_graph.temp_response)
        runtime_graph.temp_response.resolved = True
        return END
    elif (
        test_node.score
        < (COMPLEXITY_THRESHOLD - COMPLEXITY_COEFFICIENT * test_node.problem_complexity)
        and n is True
        and test_node.need_tool_crafting is True
    ):
        return "crafting"
    elif (
        test_node.score
        < (COMPLEXITY_THRESHOLD - COMPLEXITY_COEFFICIENT * test_node.problem_complexity)
        and n is True
    ):
        if test_node.need_tool_crafting is True:
            test_node.response = "The problem is too complex to craft a new tool, try reason step by step or divide complexity."
        return "backtrack"
    elif (
        test_node.score
        < (COMPLEXITY_THRESHOLD - COMPLEXITY_COEFFICIENT * test_node.problem_complexity)
        and n is False
        and runtime_graph.exist_reasoning_node_available()
    ):
        return "reasoning_mode"
    else:
        chat_completition_node = CompletitionNode(
            "Please, solve this problem with the corrected format.",
            "",
        )
        runtime_graph.add_node(chat_completition_node)
        runtime_graph.add_edge(test_node, chat_completition_node)
        runtime_graph.temp_node = chat_completition_node
        return "chat_completition"


def reasoning_mode(messages: MessagesState):
    reasoning_node = runtime_graph.call_tool_node()
    if not isinstance(reasoning_node, ReasoningNode):
        raise TypeError("Expected ReasoningNode for reasoning mode")
    msg = [HumanMessage(content=parse_response(runtime_graph.goal))]
    result = parse_response(
        reasoning_agent.invoke(
            {"messages": msg}, config={"recursion_limit": MAX_INTERACTIONS}
        )
    )
    runtime_graph.resolve_node(reasoning_node, result)
    runtime_graph.temp_response.response = result
    test_node = TestNode(
        f"{result}",
        "",
        score=0,
        tool_used=[],
    )
    runtime_graph.add_node(test_node)
    runtime_graph.add_edge(reasoning_node, test_node)
    runtime_graph.temp_node = test_node
    messages["messages"].append(AIMessage(result))
    return messages


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
    # messages = runtime_graph.append_prompt_to_messages_state(runtime_graph.temp_node)
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


content = ""


def call_graph(prompt: str):
    global content
    content = prompt
    return invoke_graph()


def invoke_graph():
    graph = StateGraph(MessagesState)
    graph.add_node(goal)
    # graph.add_node(tool_expand)
    graph.add_node(tool_reasoning)
    graph.add_node(tool_call)
    graph.add_node(backtrack)
    graph.add_node(crafting)
    graph.add_node(chat_completition)
    graph.add_node(response_evaluation)
    graph.add_node(reasoning_mode)
    graph.add_edge(START, "goal")
    # graph.add_edge("goal", "tool_expand")
    graph.add_edge("goal", "tool_reasoning")
    graph.add_edge("tool_reasoning", "tool_call")
    graph.add_edge("tool_call", "response_evaluation")
    graph.add_edge("reasoning_mode", "response_evaluation")
    graph.add_edge("backtrack", "tool_reasoning")
    graph.add_edge("crafting", "tool_reasoning")
    graph.add_edge("chat_completition", END)
    graph.add_conditional_edges(
        "response_evaluation",
        test_result,
        {
            "crafting": "crafting",
            "backtrack": "backtrack",
            "reasoning_mode": "reasoning_mode",
            "chat_completition": "chat_completition",
            END: END,
        },
    )

    graph = graph.compile()

    logger.info(graph.get_graph().draw_mermaid())
    try:
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
    except Exception as e:
        runtime_graph.clear()
        raise e

    res["output"] = runtime_graph.temp_response.response

    logger.info(res)
    print(runtime_graph.print_mermaid())
    runtime_graph.clear()
    return res
