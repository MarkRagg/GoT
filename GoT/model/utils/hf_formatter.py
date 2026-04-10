import json
from random import shuffle
import re
from datasets import Dataset, load_dataset

from langchain.messages import HumanMessage

from GoT.model.graph_model import call_graph
from GoT.model.ollama_llm import LLM
from GoT.model.utils.utils import (
    extract_output,
    normalize_list,
    normalize_number,
    symbolic_equal,
)


class ResultEval:
    def __init__(
        self,
        question: str,
        response: str,
        filtered_answer: str,
        correct_answer: str,
        answer_success: float,
    ):
        self.question = question
        self.response = response
        self.filtered_answer = filtered_answer
        self.correct_answer = correct_answer
        self.answer_success = answer_success

    @staticmethod
    def create_empty_result(question: str, correct_answer: str):
        return ResultEval(
            question=question,
            response="Error",
            filtered_answer="",
            correct_answer=correct_answer,
            answer_success=0.0,
        )


def gpqa_format(dataset: Dataset) -> list[ResultEval]:
    questions = []
    # Mapping per trasformare l'indice della lista nelle lettere A, B, C, D
    index_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}

    for data in dataset:  # Vediamo i primi 2 esempi
        sample = data

        question = sample["Question"]
        correct_answer = sample["Correct Answer"]

        # Creiamo la lista delle opzioni partendo dai dati del sample
        choices = [
            correct_answer,
            sample["Incorrect Answer 1"],
            sample["Incorrect Answer 2"],
            sample["Incorrect Answer 3"],
        ]

        shuffle(choices)

        correct_idx = choices.index(correct_answer)
        correct_letter = index_to_letter[correct_idx]

        prompt = (
            "Answer the following multiple choice question. "
            "The last line of your response should be of the following format: "
            "‘ANSWER: LETTER’ (without quotes) where LETTER is one of ABCD. "
            "Think step by step before answering.\n\n"
            f"{question}\n"
            f"A) {choices[0]}\n"
            f"B) {choices[1]}\n"
            f"C) {choices[2]}\n"
            f"D) {choices[3]}\n"
            "Answer:"
        )
        questions.append(
            ResultEval.create_empty_result(
                question=prompt, correct_answer=correct_letter
            )
        )

    return questions


def gpqa_run(questions: list[ResultEval], max_run: int, test: bool) -> list[ResultEval]:
    responses = []
    run_counter = 0
    agent = LLM().create_custom_agent(LLM().get_tools() + LLM().get_craft_tool())
    for q in questions[25:]:
        if run_counter >= max_run:
            break
        prompt = q.question
        correct_letter = q.correct_answer
        try:
            if test:
                response = extract_output(
                    agent.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config={"recursion_limit": 10},
                    )
                )
            else:
                response = extract_output(call_graph(prompt))
            norm_res = normalize_number(response)
            responses.append(
                ResultEval(
                    question=prompt,
                    response=norm_res,
                    filtered_answer="",
                    correct_answer=correct_letter,
                    answer_success=0.0,
                )
            )
        except Exception as e:
            print(f"Error processing question: {e}")
            responses.append(
                ResultEval(
                    question=prompt,
                    response="Error",
                    filtered_answer="",
                    correct_answer=correct_letter,
                    answer_success=0.0,
                )
            )
        run_counter += 1
    return responses


def gpqa_eval(responses: list[ResultEval]):
    correct = 0

    for res in responses:
        match = re.search(r"ANSWER:\s*([A-D])", res.response, re.IGNORECASE)
        res.filtered_answer = match.group(1).upper() if match else "N/A"

        if f"ANSWER: {res.correct_answer}" in res.response:
            correct += 1
            res.answer_success = 1.0

    accuracy = correct / len(responses) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total: {len(responses)}")
    print(f"Correct: {correct}")


def save_eval_results(responses: list[ResultEval], model_name: str):
    with open(f"{model_name}_eval_results.json", "w") as f:
        json.dump(responses, f, indent=2)


def gsm8k_format(dataset: Dataset) -> list[ResultEval]:
    questions = []
    for data in dataset:
        sample = data
        question = sample["question"]
        correct_answer = sample["answer"]
        prompt = (
            "Answer the following math problem. Respond in the following format: #### NUMBER "
            "Think step by step before answering.\n\n"
            f"{question}\n"
            "Answer:"
        )

        questions.append(
            ResultEval.create_empty_result(
                question=prompt, correct_answer=correct_answer
            )
        )

    return questions


def gsm8k_run(
    questions: list[ResultEval], max_run: int, test: bool
) -> list[ResultEval]:
    responses = []
    run_counter = 0
    agent = LLM().create_custom_agent(LLM().get_tools() + LLM().get_craft_tool())
    for q in questions:
        if run_counter >= max_run:
            break
        prompt = q.question
        correct_answer = q.correct_answer
        try:
            if test:
                response = extract_output(
                    agent.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config={"recursion_limit": 20},
                    )
                )
            else:
                response = extract_output(call_graph(prompt))
            norm_res = normalize_number(response)
            responses.append(
                ResultEval(
                    question=prompt,
                    response=norm_res,
                    filtered_answer="",
                    correct_answer=correct_answer,
                    answer_success=0.0,
                )
            )
        except Exception as e:
            print(f"Error processing question: {e}")
            responses.append(
                ResultEval(
                    question=prompt,
                    response="Error",
                    filtered_answer="",
                    correct_answer=correct_answer,
                    answer_success=0.0,
                )
            )
        run_counter += 1
    return responses


def gsm8k_eval(responses: list[ResultEval]):
    correct = 0

    for res in responses:
        opt_res = re.search(r"####\s*(-?[\d,.]+)", res.response)
        norm_res = opt_res.group(1) if opt_res else "N/A"
        norm_correct = normalize_number(res.correct_answer)
        res.filtered_answer = norm_res

        if norm_res in norm_correct:
            correct += 1
            res.answer_success = 1.0

    accuracy = correct / len(responses) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total: {len(responses)}")
    print(f"Correct: {correct}")


def hendrycks_math_format(dataset: Dataset) -> list[ResultEval]:
    questions = []
    for data in dataset:
        sample = data
        question = sample["problem"]
        reg_exp = re.search(r"\\boxed\{(.*)\}", sample["solution"])
        correct_answer = reg_exp.group(1) if reg_exp else "N/A"

        prompt = (
            "Answer the following math problem. Respond in the following format: \\boxed{answer}. "
            "Think step by step before answering.\n\n"
            f"{question}\n"
            "Answer:"
        )

        questions.append(
            ResultEval.create_empty_result(
                question=prompt, correct_answer=correct_answer
            )
        )

    return questions


def hendrycks_math_run(
    questions: list[ResultEval], max_run: int, test: bool
) -> list[ResultEval]:
    responses = []
    run_counter = 0
    agent = LLM().create_custom_agent(LLM().get_tools() + LLM().get_craft_tool())
    for q in questions:
        if run_counter >= max_run:
            break
        prompt = q.question
        correct_answer = q.correct_answer
        try:
            if test:
                response = extract_output(
                    agent.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config={"recursion_limit": 20},
                    )
                )
            else:
                response = extract_output(call_graph(prompt))
            norm_res = normalize_number(response)
            responses.append(
                ResultEval(
                    question=prompt,
                    response=norm_res,
                    filtered_answer="",
                    correct_answer=correct_answer,
                    answer_success=0.0,
                )
            )
        except Exception as e:
            print(f"Error processing question: {e}")
            responses.append(
                ResultEval(
                    question=prompt,
                    response="Error",
                    filtered_answer="",
                    correct_answer=correct_answer,
                    answer_success=0.0,
                )
            )
        run_counter += 1
    return responses


def hendrycks_math_eval(responses: list[ResultEval]):
    correct = 0

    for res in responses:
        opt_res = re.search(r"\\boxed\{(.*)\}", res.response)
        norm_res = opt_res.group(1) if opt_res else "N/A"
        norm_correct = normalize_number(res.correct_answer)
        res.filtered_answer = norm_res

        if (
            (norm_res in norm_correct)
            or (normalize_list(norm_res) == normalize_list(norm_correct))
            or (symbolic_equal(norm_res, norm_correct))
        ):
            correct += 1
            res.answer_success = 1.0

    accuracy = correct / len(responses) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total: {len(responses)}")
    print(f"Correct: {correct}")


def use_gpqa(max_run: int, test: bool, model_name: str):
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    data = ds["train"]
    questions = gpqa_format(data)
    responses = gpqa_run(questions, max_run=max_run, test=test)
    gpqa_eval(responses)
    save_eval_results(responses, model_name=model_name)


def use_gsm8k(max_run: int, test: bool, model_name: str):
    ds = load_dataset("gsm8k", "main")
    data = ds["test"]
    questions = gsm8k_format(data)
    responses = gsm8k_run(questions, max_run=max_run, test=test)
    gsm8k_eval(responses)
    save_eval_results(responses, model_name=model_name)


def use_hendrycks_math(max_run: int, test: bool, model_name: str, type: str):
    ds = load_dataset("EleutherAI/hendrycks_math", type)
    data = ds["test"]
    questions = hendrycks_math_format(data)
    responses = hendrycks_math_run(questions, max_run=max_run, test=test)
    hendrycks_math_eval(responses)
    save_eval_results(responses, model_name=model_name)
