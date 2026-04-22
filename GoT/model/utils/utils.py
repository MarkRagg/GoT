import json
import re

import mlflow
import numpy as np
from sympy import simplify, sympify

from GoT.model.runtime_graph import Response, Score
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
    score_res = extract_output(response)
    if isinstance(score, Score):
        return score
    elif score_res is not None:
        data = json.loads(score_res)
        return Score.model_validate(data)
    else:
        return Score(
            score=0, description="Failed to parse score", need_tool_crafting=False
        )


def parse_response_for_tool_node(response: MessagesState) -> Response:
    """
    Parse LLM response to get the response for a tool node.

    :param response: The LLM response
    :type response: MessagesState
    :return: The response for the tool node
    :rtype: Response
    """
    structured_response = response.get("structured_response")
    score_res = extract_output(response)
    if isinstance(structured_response, Response):
        return structured_response
    elif score_res is not None:
        data = json.loads(score_res)
        return Response.model_validate(data)
    else:
        return Response(
            response="Failed to parse response",
            explanation="The structured response is not in the expected format.",
        )


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
                content = last_msg.content
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    return "".join(text_parts).strip()
                return str(last_msg.content)
            elif isinstance(last_msg, dict) and "content" in last_msg:
                return str(last_msg["content"])

    return str(result) if result else ""


def normalize_number(num_str: str) -> str:
    """Remove commas and dollar sign."""
    num_str = num_str.replace("$", "")
    num_str = num_str.replace(",", "")
    num_str = num_str.replace("*", "")
    num_str = num_str.strip()

    # Remove trailing .0 (and only .0)
    if num_str.endswith(".0"):
        # Ensure it's actually a valid number before trimming
        try:
            if float(num_str).is_integer():
                num_str = str(int(float(num_str)))
        except ValueError:
            pass

    return num_str


def normalize_list(s: str):
    "Extract numbers from a string and return them as a sorted list. This is useful for comparing answers that are lists of numbers regardless of order."
    nums = re.findall(r"-?\d+", s)
    return sorted(nums)


def symbolic_equal(a, b):
    """Check if two mathematical expressions are symbolically equal."""
    try:
        return simplify(sympify(a) - sympify(b)) == 0
    except Exception:
        return False


def print_benchmark_result(results: dict, task_name: str, filter: str) -> None:
    samples = results["samples"][task_name]

    flex_samples = [s for s in samples if filter in s.get("filter", "")]

    n_total = len(flex_samples)
    n_correct = sum(1 for s in flex_samples if s.get("exact_match", 0) == 1.0)
    n_wrong = n_total - n_correct

    print(f"Total: {n_total}")
    print(f"Correct answers (filter={filter}): {n_correct}")
    print(f"Wrong answers (filter={filter}): {n_wrong}")


def print_benchmark_result_loglikehood(
    results: dict, task_name: str, filter_val: str
) -> None:
    # 1. Recupero i samples
    samples = results.get("samples", {}).get(task_name, [])

    # 2. Filtro (solitamente filter="none" nel tuo caso)
    flex_samples = [s for s in samples if filter_val in s.get("filter", "")]

    n_total = len(flex_samples)
    n_correct = 0

    for s in flex_samples:
        try:
            scores = [r[0][0] for r in s.get("resps", [])]

            if not scores:
                continue

            predicted_idx = int(np.argmax(scores))

            if predicted_idx >= 5.0:
                n_correct += 1

        except (ValueError, IndexError):
            continue

    n_wrong = n_total - n_correct

    print(f"--- Risultati basati su RESPS (Task: {task_name}) ---")
    print(f"Total: {n_total}")
    print(f"Correct: {n_correct}")
    print(f"Wrong: {n_wrong}")

def download_mlflow_traces(n_max: int):
    traces = mlflow.search_traces(max_results=n_max, order_by=["timestamp DESC"])
    traces.to_csv("traces.csv", index=False)
