import ast
import os
from pathlib import Path
import re

from langchain.tools import tool


@tool
def python_tool(code: str) -> str:
    """Execute Python code and return the output. The code should assign the final result to a variable named 'result'."""

    def sanitize_input(query: str) -> str:
        """Sanitize input to the python REPL.
        Remove whitespace, backtick & python
        (if llm mistakes python console as terminal)
        """
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        query = re.sub(r"(\s|`)*$", "", query)
        return query

    try:
        # WARNING: Using eval/exec can be dangerous. This is just for demonstration purposes.
        namespace: dict[str, object] = {}
        exec(sanitize_input(code), namespace, namespace)
        return str(namespace.get("result", "No result variable defined."))
    except Exception as e:
        return str(e)


@tool
def install_dependency(package_name: str) -> str:
    """Install a Python package using poetry."""
    try:
        os.system(f"poetry add {package_name}")
        return f"Package {package_name} installed successfully."
    except Exception as e:
        return str(e)


def is_valid_annotation(annotation):
    if isinstance(annotation, ast.Name):
        if annotation.id in {"list", "dict", "set"}:
            return False
        return True

    # Caso tipo generico: List[int], dict[str, int]
    if isinstance(annotation, ast.Subscript):
        return True

    return False

@tool
def craft_tool(tool_function: str) -> str:
    """Save the function definition provided by the LLM as a tool that can be used by other agents.
    The function should be defined as a python function.
    The function should be general and reusable, and should not be specific to the current problem.
    The function should be defined in a way that it can be imported and used by other agents."""

    def sanitize_input(query: str) -> str:
        """Sanitize input to the python REPL.
        Remove whitespace, backtick & python
        (if llm mistakes python console as terminal)
        """
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        query = re.sub(r"(\s|`)*$", "", query)
        return query
    
    code = f"""\n\n{sanitize_input(tool_function)}"""

    # Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error, tool not saved: {e}"

    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1:
        return "Error: exactly one function must be defined."
    
    func = functions[0]

    for arg in func.args.args:
        if arg.annotation is None:
            return f"Error: missing type annotation for '{arg.arg}'"

        if not is_valid_annotation(arg.annotation):
            return f"Error: invalid type for '{arg.arg}' (must be typed, e.g. List[int], set[str], dict[str, int], etc.)"
    
    if func.returns is None:
        return "Error: missing return type"

    try:
        base_dir = Path(__file__).parent
        file_path = base_dir / "ai_tool.py"
        with open(file_path, "a") as f:
            f.write(code)
        return "Tool crafted successfully."
    except Exception as e:
        print(e)
        return str(e)
