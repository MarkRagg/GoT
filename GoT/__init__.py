import logging

from GoT.ollama_llm import OllamaLLM
from GoT.graph_model import invoke_graph

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("GoT")


# this is the initial module of your app
# this is executed whenever some client-code is calling `import GoT` or `from GoT import ...`
# put your main classes here, eg:
class MyClass:
    def my_method(self):
        return "Hello World"

def main():
    # this is the main module of your app
    # it is only required if your project must be runnable
    # this is the script to be executed whenever some users writes `python -m GoT` on the command line, eg.
    # x = MyClass().my_method()
    # print(x)
    res = invoke_graph()


# let this be the last line of this file
logger.info("GoT loaded")
