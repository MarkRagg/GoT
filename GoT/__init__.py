import json
import logging

from lm_eval import evaluator, tasks
from GoT.model.graph_model import call_graph
from GoT.model.lm_wrapper import LangGraphLMWrapper

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger("GoT")


def lm_eval_benchmark():
    task_list = ["gsm8k"]
    lm = LangGraphLMWrapper()
    task_dict = tasks.get_task_dict(task_list)

    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=10,  # Limit the number of samples
        log_samples=True,
    )

    # Save results to a JSON file
    with open("graph_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)


def custom_test():
    call_graph("What is 4726621 + 2 * 392 - 3432?")


def main():
    # It could be changed with custom_test() to test a custom problem instead of the benchmark
    lm_eval_benchmark()


# let this be the last line of this file
# logger.info("GoT loaded")
