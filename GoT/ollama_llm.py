from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent
import mlflow

from GoT.tools.math_tool import sum_four, sum_three, summing, minus, square_root

mlflow.set_experiment("marcoraggini-experiment")
mlflow.openai.autolog()
mlflow.langchain.autolog()

class OllamaLLM:
    def __init__(self):
        with mlflow.set_active_model(name="ollama-agent-ministral-3"):
            self.ollamaLLM = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="dummy",
                model="ministral-3:8b",
            )

            self.system_prompt = SystemMessage("""
                You are a helpful assistant able to address general problems. 
                ...
            """)

            self.agent = create_agent(
                model=self.ollamaLLM,
                tools=[sum_four, summing, minus, sum_three, square_root],
                system_prompt=self.system_prompt,
            )

    def create_custom_agent(self, tools):
       return create_agent(
                model=self.ollamaLLM,
                tools=tools,
                system_prompt=self.system_prompt,
            )
