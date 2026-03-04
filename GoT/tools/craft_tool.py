import os
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
        namespace = {}
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


@tool
def craft_tool(tool_function: str) -> str:
    """Craft a new tool by appending the function definition to tools.py."""
    try:
        os.system(f"echo '{tool_function}' > artifacts.py")
        return "Tool crafted successfully."
    except Exception as e:
        return str(e)