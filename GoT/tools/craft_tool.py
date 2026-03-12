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

    try:
        base_dir = Path(__file__).parent
        file_path = base_dir / "ai_tool.py"
        code = f"""\n\n@tool\n{sanitize_input(tool_function)}"""
        with open(file_path, "a") as f:
            f.write(code)
        return "Tool crafted successfully."
    except Exception as e:
        print(e)
        return str(e)