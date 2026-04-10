import inspect
import os
import importlib
import sys
from langchain.tools import BaseTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent
import mlflow

from GoT.tools.math_tool import (
    multiply,
    summing,
    minus,
    square_root,
    divide,
)

from GoT.tools.craft_tool import craft_tool, install_dependency

load_dotenv()

mlflow.set_experiment("marcoraggini-experiment")
mlflow.openai.autolog()
mlflow.gemini.autolog()
mlflow.langchain.autolog()

SYSTEM_PROMPT_GENERAL = """
                You are a helpful assistant able to address general problems. 
                ...
            """


class LLM:
    def __init__(self):
        with mlflow.set_active_model(name="ollama-agent-ministral-3"):
            self.ollamaLLM = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="dummy",
                model="ministral-3:8b",
                temperature=0.5,
                reasoning_effort="none",
            )

            # GEMINI LLMs
            self.remoteLLMStandard = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=os.environ.get("GEMINI_API_KEY"),
                temperature=1.0,  # Gemini 3.0+ defaults to 1.0
            )
            self.remoteLLMResponseFormat = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=os.environ.get("GEMINI_API_KEY"),
                temperature=1.0,  # Gemini 3.0+ defaults to 1.0
            )
            self.remoteLLMScoreFormat = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=os.environ.get("GEMINI_API_KEY"),
                temperature=0.7,  # Gemini 3.0+ defaults to 1.0
            )

            self.remoteLLMs = {
                "remote_standard": self.remoteLLMStandard,
                "remote_response_format": self.remoteLLMResponseFormat,
                "remote_score_format": self.remoteLLMScoreFormat,
            }

            self.system_prompt = SystemMessage(SYSTEM_PROMPT_GENERAL)

    def get_tools(self):
        initial_tools = [summing, minus, square_root, multiply, divide]
        crafted_tools = self.get_crafted_tools()
        return initial_tools + crafted_tools

    def get_craft_tool(self):
        return [craft_tool, install_dependency]

    def get_crafted_tools(self) -> list[BaseTool]:
        module_name = "GoT.tools.ai_tool"
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
        tools = []
        for name, obj in module.__dict__.items():
            if inspect.isfunction(obj) and obj.__module__ == module.__name__:
                tools.append(tool(obj))  # wrap runtime

        return tools

    def create_custom_agent(
        self,
        tools,
        system_prompt: SystemMessage = SystemMessage(SYSTEM_PROMPT_GENERAL),
        response_format=None,
        type: str = "remote_standard",
    ):
        return create_agent(
            model=self.remoteLLMs[type],
            tools=tools,
            system_prompt=system_prompt,
            response_format=response_format,
        )
