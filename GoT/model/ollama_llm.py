import os

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

from GoT.tools.craft_tool import python_tool, install_dependency

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
                model="qwen3:8b",
                temperature=0.5,
                reasoning_effort="none",
            )

            self.remoteLLM = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPEN_ROUTER_KEY"),
                model="openai/gpt-oss-120b:free",
                temperature=0.5,
            )

            self.system_prompt = SystemMessage(SYSTEM_PROMPT_GENERAL)

            self.agent = create_agent(
                model=self.ollamaLLM,
                tools=[summing, minus, square_root, multiply, divide],
                system_prompt=self.system_prompt,
            )

    def get_tools(self):
        return [summing, minus, square_root, multiply, divide]
    
    def get_craft_tool(self):
        return [python_tool, install_dependency]

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
