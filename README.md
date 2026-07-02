# Agentic GoT

Agentic GoT (**Graph of Thought**) is a LangChain / LangGraph based reasoning agent that solves problems by building and traversing a graph of intermediate reasoning, tool-call, scoring, and backtracking nodes, instead of a single linear chain-of-thought. It ships with:

- A **runtime reasoning graph** (`GoT/core/runtime_graph.py`, `GoT/core/graph_model.py`) with typed nodes (`GoalNode`, `ReasoningNode`, `ToolNode`, `TestNode`, `CraftingNode`, `BacktrackNode`, `CompletitionNode`, `ResponseNode`) and Mermaid export for visualizing a run.
- A pluggable **tool belt**: arithmetic (`agent_tools/math_tool.py`), web/knowledge lookup via Wikipedia and arXiv (`agent_tools/web_tool.py`), a sandboxed Python executor, and a **tool-crafting tool** that lets the agent write and persist brand-new tools for itself at runtime (`agent_tools/craft_tool.py`).
- **Benchmark harnesses** for GSM8K, GPQA (diamond), Hendrycks MATH, and GAIA, wired into [`lm-eval-harness`](https://github.com/EleutherAI/lm-evaluation-harness) (`GoT/experiments/`), so the graph agent (and a plain baseline agent) can be scored automatically.
- **MLflow** autologging for OpenAI/Gemini/LangChain calls, so every run is traced and inspectable.

## Requirements

| Tool | Version | Notes |
|---|---|---|
| Python | `>=3.10, <3.14` | CI tests on 3.10–3.13, on Ubuntu/Windows/macOS |
| [Poetry](https://python-poetry.org/) | `^2.2` | dependency & venv management |
| [Ollama](https://ollama.com/) | any recent | optional, only needed for running local Ollama models|

## Quick start

```bash
# 1. Clone
git clone https://github.com/MarkRagg/GoT.git
cd GoT

# 2. Install Poetry (pinned version, isolated from your system Python)
pip install -r requirements.txt

# 3. Install project + dev dependencies (creates an in-project .venv, see poetry.toml)
poetry install

# 4. Configure environment variables (see below)
cp .env.example .env   # if present — otherwise just create .env, see next section
$EDITOR .env

# 5. Run the test suite to confirm everything is wired correctly
poetry run poe test

# 6. Run the agent on a custom prompt in graph mode
poetry run python -m GoT --benchmark custom --mode graph --prompt "What is the square root of 144, then look up who proved it?"
```

> Tip: run `poetry shell` once to activate the virtualenv, so you can drop the `poetry run` prefix for the rest of the session.

## Environment variables

GoT loads environment variables from a `.env` file at import time via `python-dotenv` (see `GoT/__init__.py` and `GoT/core/llm.py`). Create a `.env` file in the repository root:

```dotenv
# Required — Gemini is the default remote LLM backend for every agent role
# (standard reasoning, structured/graph reasoning, tool crafting, and scoring).
# Get a key at https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your-gemini-api-key

# Required only if you run benchmarks that pull gated Hugging Face datasets
# (currently GPQA and GAIA). Get a token at https://huggingface.co/settings/tokens
# and make sure your HF account has accepted the dataset's access terms.
HF_TOKEN=your-huggingface-token
```

| Variable | Required | Used by | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes (for any Gemini-backed run — the default) | `GoT/core/llm.py` | Authenticates the four `ChatGoogleGenerativeAI` roles (`remote_standard`, `remote_response_format`, `remote_score_format`, `remote_crafter`) that power reasoning, response formatting, scoring, and tool crafting. |
| `HF_TOKEN` | Only for `--benchmark gpqa` / `--benchmark gaia` | `GoT/experiments/hf_formatter.py` | Downloads gated benchmark datasets from the Hugging Face Hub. `gsm8k` and `hendrycks_math` do not require it. |

### Optional / no setup needed

- **Local Ollama model** — `GoT/core/llm.py` also instantiates an `ollamaLLM` pointed at `http://localhost:11434/v1` with model `ministral-3:8b`, using the dummy API key `"dummy"` (Ollama's OpenAI-compatible endpoint doesn't check it). This path is only exercised if your own code selects it; it's not required for the default Gemini-backed CLI flows. If you want to use it: install [Ollama](https://ollama.com/download), then run `ollama pull ministral-3:8b` and make sure `ollama serve` is running before invoking GoT.
- **MLflow** — tracing is enabled automatically (`mlflow.set_experiment("marcoraggini-experiment")` plus autolog for OpenAI/Gemini/LangChain) and writes to a local `./mlruns` directory by default. Point it at a remote tracking server instead by exporting `MLFLOW_TRACKING_URI` before running GoT — no code changes needed.
- `.env` is already covered by `.gitignore` — never commit real API keys.

## Usage

The package entry point (`GoT/__main__.py` → `GoT.main()`) parses CLI args via `GoT/cli/parse_args.py`:

```bash
poetry run python -m GoT --benchmark <gsm8k|gpqa|hendrycks_math|gaia|custom> --mode <graph|standard> [options]
```

| Flag | Required | Values | Description |
|---|---|---|---|
| `--benchmark` | Yes | `gsm8k`, `gpqa`, `hendrycks_math`, `gaia`, `custom` | Which benchmark (or ad-hoc prompt) to run. |
| `--mode` | Yes | `graph`, `standard` | `graph` runs the full Graph-of-Thought reasoning pipeline; `standard` runs a single-pass baseline agent. |
| `--prompt` | Only for `custom` | free text | The prompt to run when `--benchmark custom` is selected. |
| `--max_run` | No (default `1`) | int | Number of benchmark samples/iterations to run. |
| `--category` | No (default `algebra`) | `algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`, `number_theory`, `precalculus`, `prealgebra` | Math subject filter, only used with `--benchmark hendrycks_math`. |

Examples:

```bash
# Ad-hoc question, full graph reasoning
poetry run python -m GoT --benchmark custom --mode graph --prompt "Explain and solve: integral of x^2 dx from 0 to 3"

# Baseline (non-graph) agent on 10 GSM8K problems
poetry run python -m GoT --benchmark gsm8k --mode standard --max_run 10

# Graph agent on Hendrycks MATH, geometry category
poetry run python -m GoT --benchmark hendrycks_math --mode graph --category geometry --max_run 5
```

Results are written as JSON in the working directory (e.g. `graph_benchmark_results.json`, `test_benchmark_results.json`, `<model_name>_eval_results.json`), and every run is traced in MLflow.

## Development

```bash
poetry install                    # install runtime + dev dependencies

poetry run poe test               # run the pytest suite
poetry run poe coverage           # run tests with coverage
poetry run poe coverage-report    # print coverage summary
poetry run poe coverage-html      # generate an HTML coverage report (htmlcov/)

poetry run poe static-checks      # ruff check + mypy
poetry run poe format             # auto-format with ruff
poetry run poe format-check       # check formatting without modifying files
poetry run poe compile            # byte-compile the package and tests (syntax check)
```

CI (`.github/workflows/check.yml`) runs the same static checks, formatting check, and coverage on every push/PR, then runs the test suite across Python 3.10–3.13 on Ubuntu, Windows, and macOS.

## Project structure

```
GoT/
├── GoT/
│   ├── __main__.py            # `python -m GoT` entry point
│   ├── cli/parse_args.py      # argparse CLI definition
│   ├── core/
│   │   ├── llm.py             # LLM roles (Gemini remote + local Ollama), tool wiring
│   │   ├── graph_model.py     # LangGraph graph definition / orchestration
│   │   └── runtime_graph.py   # Reasoning-graph node types + Mermaid export
│   ├── agent_tools/           # math_tool, web_tool (Wikipedia/arXiv), craft_tool, runtime_graph_tool, ai_tool (crafted tools land here)
│   ├── experiments/           # lm-eval-harness wrappers + per-benchmark dataset formatters
│   └── utils/utils.py         # answer parsing/normalization helpers
├── tests/                     # unit tests (pytest)
├── pyproject.toml             # Poetry config, dependencies, poe tasks
└── .github/workflows/         # CI (check.yml) and release (deploy.yml)
```

## Experiments

The experiments reported in the paper were run with the following command, varying only the `--category` flag across the Hendrycks MATH subject areas:

```bash
poetry run python -m GoT --benchmark hendrycks_math --mode graph --max_run 50 --category algebra
```

- `--mode graph` was used to produce the Agentic GoT results.
- `--mode standard` was used to produce the Zero-Shot CoT baseline results, keeping all other flags identical.
- `--category` was swapped in turn for each of the supported values (`algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`, `number_theory`, `precalculus`) depending on which subject was being evaluated.

By default, all four LLM roles use `gemini-2.5-flash`. To use a different model (in the experiments we use `gemini-2.5-flash`, `gemini-2.5-flash-lite` and `gemini-3.1-flash-lite`), edit `GoT/core/llm.py` and change the model name in the following four variables:

- `remoteLLMStandard`
- `remoteLLMReasoning`
- `remoteLLMCrafter`
- `remoteLLMEvaluator`

## License

See [`LICENSE`](./LICENSE).
