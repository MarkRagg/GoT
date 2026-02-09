import re


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
            if tool_name and counter <= 3:
                counter += 1
                tool_list.append(tool_name)

    return tool_list


def parse_score(response: str) -> int:
    """
    Parse LLM response to get the score in a format like Score: number

    :param response: The LLM response
    :type response: str
    :return: The score
    :rtype: int
    """
    # Look for pattern like "Score: <number>"
    match = re.search(r"Score:\s*(\d+)", response, re.IGNORECASE)

    if match:
        return int(match.group(1))

    raise ValueError(f"Could not find score in response: {response}")


def remove_tools_from_list(tool_list, tools_to_remove):
    """
    Remove a list of tools and return the updated list

    :param tool_list: is a list of StructuredTool
    :param tools_to_remove: list of string representing the names of the tools
    """
    result = [tool for tool in tool_list if tool.func.__name__ not in tools_to_remove]
    print(result)
    return result
