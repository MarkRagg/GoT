import os
import importlib
from langchain.tools import BaseTool
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
mlflow.langchain.autolog()

SYSTEM_PROMPT_GENERAL = """
                You are a helpful assistant able to address general problems. 
                ...
            """


class OllamaLLM:
    def __init__(self):
        with mlflow.set_active_model(name="ollama-agent-ministral-3"):
            self.ollamaLLM = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="dummy",
                model="ministral-3:8b",
                temperature=0.5,
                reasoning_effort="none",
            )

            # self.remoteLLM = ChatOpenAI(
            #     base_url="https://openrouter.ai/api/v1",
            #     api_key=os.environ.get("OPEN_ROUTER_KEY"),
            #     model="openai/gpt-oss-120b:free",
            #     temperature=0.5,
            # )
            self.remoteLLM = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                api_key=os.environ.get("GEMINI_API_KEY"),
                temperature=1.0,  # Gemini 3.0+ defaults to 1.0
            )

            self.system_prompt = SystemMessage(SYSTEM_PROMPT_GENERAL)

    def get_tools(self):
        initial_tools = [summing, minus, square_root, multiply, divide]
        crafted_tools = self.get_crafted_tools()
        return initial_tools + crafted_tools
    
    def get_craft_tool(self):
        return [craft_tool, install_dependency]
    
    def get_crafted_tools(self) -> list[BaseTool]:
        module = importlib.import_module("GoT.tools.ai_tool")
        tools = []
        for obj in module.__dict__.values():
            if isinstance(obj, BaseTool):
                tools.append(obj)
                
        return tools

    def create_custom_agent(
        self,
        tools,
        system_prompt: SystemMessage = SystemMessage(SYSTEM_PROMPT_GENERAL),
        response_format=None,
    ):
        return create_agent(
            model=self.ollamaLLM,
            tools=tools,
            system_prompt=system_prompt,
            response_format=response_format,
        )
