import argparse
import sys

from GoT.model.utils.hf_formatter import use_gaia, use_gpqa, use_gsm8k, use_hendrycks_math


def defining_and_parse_args():
    parser = argparse.ArgumentParser(
        description="Run the GoT model on a benchmark or a custom problem."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        type=str,
        choices=["gsm8k", "gpqa", "hendrycks_math", "gaia"],
        help="The benchmark to run the model on.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=["graph", "standard"],
        help="Whether to run the standard model or the graph model.",
    )
    parser.add_argument(
        "--max_run",
        type=int,
        default=1,
        help="The maximum number of runs for the benchmark.",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="algebra",
        choices=[
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "precalculus",
            "prealgebra"
        ],
        help="The type of math problems to run, only for hendrycks_math.",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    return args


def call_benchmark(args):
    mode = args.mode if args.mode else "standard"
    test = mode == "standard"
    max_run = args.max_run
    if args.benchmark == "gsm8k":
        use_gsm8k(max_run=max_run, test=test, model_name=mode)
    elif args.benchmark == "gpqa":
        use_gpqa(max_run=max_run, test=test, model_name=mode)
    elif args.benchmark == "hendrycks_math":
        use_hendrycks_math(max_run=max_run, test=test, model_name=mode, type=args.type)
    elif args.benchmark == "gaia":
        use_gaia(max_run=max_run, test=test, model_name=mode)
