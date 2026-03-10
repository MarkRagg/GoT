import json
import logging
import os
from dotenv import load_dotenv

from lm_eval import evaluator, tasks
from GoT.model.graph_model import call_graph
from GoT.model.lm_wrapper import (
    LangGraphLMWrapper,
    OllamaTestLMWrapper,
    LangGraphBigBenchWrapper
)
from GoT.model.utils.utils import print_benchmark_result

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("GoT")

load_dotenv()

# Possible filter = "flexible", "none", "strict"

def lm_eval_test_benchmark():
    task_name = "gsm8k"
    task_list = [task_name]
    test_lm = OllamaTestLMWrapper()
    task_dict = tasks.get_task_dict(task_list)

    results = evaluator.evaluate(
        lm=test_lm,
        task_dict=task_dict,
        limit=20,  # Limit the number of samples
        log_samples=True,
    )

    # Save results to a JSON file
    with open("ollama_test_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print_benchmark_result(results, task_name, filter="none")


def lm_eval_graph_benchmark():
    task_name = "bigbench_logical_sequence_generate_until"
    task_list = [task_name]
    lm = LangGraphBigBenchWrapper()
    task_dict = tasks.get_task_dict(task_list)

    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=5,  # Limit the number of samples
        log_samples=True,
    )

    # Save results to a JSON file
    with open("ollama_graph_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print_benchmark_result(results, task_name, filter="none")


def custom_test():
    call_graph(
        "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"
    )


def main():
    # It could be changed with custom_test() to test a custom problem instead of the benchmark
    custom_test()

# let this be the last line of this file
logger.info("GoT loaded")
