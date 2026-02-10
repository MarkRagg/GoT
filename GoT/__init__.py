import json
import logging

from lm_eval import evaluator, tasks
from GoT.model.lm_wrapper import LangGraphLMWrapper, LangGraphBigBenchWrapper

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("GoT")

def main():
    # tm = tasks.TaskManager()
    # all_tasks = tm.all_tasks

    # # Filtra solo i task di bigbench
    # bigbench_tasks = [t for t in all_tasks if 'bigbench' in t.lower()]

    # print("Task di BigBench disponibili:")
    # for task in sorted(bigbench_tasks):
    #     print(f"  - {task}")
    task_list = ["bigbench_arithmetic_generate_until"]
    lm = LangGraphBigBenchWrapper()
    task_dict = tasks.get_task_dict(task_list)

    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=1, # Limit to 3 samples for quick testing
        log_samples=True, 
        verbosity="INFO" 
    )

    # --- Salva in JSON ---
    with open("graph_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

# let this be the last line of this file
logger.info("GoT loaded")