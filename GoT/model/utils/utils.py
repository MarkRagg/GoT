import re

from GoT.model.runtime_graph import Score
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage


def parse_response(res) -> str:
    """
    Parse the LLm response using agent.invoke

    :param res: the MessagesState
    :return: The response in string
    """
    return res["messages"][-1].content


def parse_tool_list(response: str) -> list[str]:
    """
    Parse tool names from LLM response containing bullet-pointed list.
    Handles both * and - formats and extracts only the tool name,
    ignoring any extra descriptive text.

    Args:
        response: Raw text response from LLM
    Returns:
        List of tool names extracted from the response
    """
    tool_list = []
    counter = 0
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("*") or line.startswith("-"):
            # Remove the bullet point and extract the tool name
            content = re.sub(r"^[\*\-]\s+", "", line)
            # Extract just the first word (the tool name)
            tool_name = content.split()[0]
            if tool_name and counter <= 2:  # Limit to 3 tools
                counter += 1
                tool_list.append(tool_name)

    return tool_list


def parse_score(response: MessagesState) -> Score:
    """
    Parse LLM response to get the score in a format like Score: number

    :param response: The LLM response
    :type response: str
    :return: The score
    :rtype: int
    """
    score = response.get("structured_response")
    if isinstance(score, Score):
        return score
    else:
        return Score(score=0, description="Failed to parse score")


def extract_tool_used(response: MessagesState) -> list[str]:
    """
    Extract the tool used from the response of the LLM, if present.

    :param response: The LLM response
    :type response: MessagesState
    :return: The list of tools used
    :rtype: list[str]
    """
    tools_used = []
    for msg in response.get("messages", []):
        if isinstance(msg, AIMessage):
            for tool_call in msg.tool_calls:
                tools_used.append(tool_call["name"])
    return tools_used


def remove_tools_from_list(tool_list, tools_to_remove):
    """
    Remove a list of tools and return the updated list

    :param tool_list: is a list of StructuredTool
    :param tools_to_remove: list of string representing the names of the tools
    """
    result = [tool for tool in tool_list if tool.func.__name__ not in tools_to_remove]
    return result


def extract_output(result) -> str:
    """
    Extracts the output from the result of invoke_graph.
    Supports various formats.
    """
    if isinstance(result, dict):
        if "output" in result and result["output"]:
            return str(result["output"])

        if "messages" in result and result["messages"]:
            messages = result["messages"]
            last_msg = messages[-1]

            if hasattr(last_msg, "content"):
                return str(last_msg.content)
            elif isinstance(last_msg, dict) and "content" in last_msg:
                return str(last_msg["content"])

    return str(result) if result else ""
