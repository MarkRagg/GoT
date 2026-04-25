import json
import logging
from dotenv import load_dotenv

from lm_eval import evaluator, tasks
from GoT.core.graph_model import call_graph
from GoT.experiments.lm_wrapper import LangGraphBigBenchWrapper, TestBigBenchWrapper
from GoT.cli.parse_args import call_benchmark, defining_and_parse_args
from GoT.utils.utils import (
    print_benchmark_result,
    print_benchmark_result_loglikehood,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("GoT")

load_dotenv()

# Possible filter = "flexible", "none", "strict"


def lm_eval_test_benchmark():
    task_name = "gpqa_diamond_zeroshot"
    task_list = [task_name]
    test_lm = TestBigBenchWrapper()
    task_dict = tasks.get_task_dict(task_list)

    results = evaluator.evaluate(
        lm=test_lm,
        task_dict=task_dict,
        limit=2,  # Limit the number of samples
        log_samples=True,
        # samples={task_name: [20, 25, 100]},
    )

    # Save results to a JSON file
    with open("test_benchmark_results.json", "w") as f:
        json.dump(results["samples"], f, indent=2)

    print_benchmark_result(results, task_name, filter="strict-match")


def lm_eval_graph_benchmark():
    # hendrycks_math_geometry
    task_name = "gpqa_diamond_zeroshot"
    task_list = [task_name]
    lm = LangGraphBigBenchWrapper()
    task_dict = tasks.get_task_dict(task_list)

    results = evaluator.evaluate(
        lm=lm,
        # limit=1,
        task_dict=task_dict,
        samples={task_name: [20, 25]},
        log_samples=True,
    )

    # Save results to a JSON file
    with open("graph_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print_benchmark_result_loglikehood(results, task_name, filter_val="none")


def custom_test():
    call_graph("Solve this integral ∫x2⋅ex2dx")


def main():
    # It could be changed with custom_test() to test a custom problem instead of the benchmark
    args = defining_and_parse_args()
    call_benchmark(args)


# let this be the last line of this file
logger.info("GoT loaded")
