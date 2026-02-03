from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_agent

class OllamaLLM:
    def __init__(self):
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
            tools=[],
            system_prompt=self.system_prompt,
        )

        self.messages = [
            HumanMessage("Solve the equation x^3 - 2x - 5 = 0 numerically."),
        ]

    def invoke(self): 
        return self.agent.invoke({"messages": self.messages})
    