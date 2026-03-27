
import json
from random import shuffle
import re

from langchain.messages import HumanMessage

from GoT.model.graph_model import call_graph
from GoT.model.ollama_llm import LLM
from GoT.model.utils.utils import extract_output, normalize_number

def gpqa_format(dataset) -> list[dict[str, str]]:
    questions = []
    # Mapping per trasformare l'indice della lista nelle lettere A, B, C, D
    index_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}

    for data in dataset:  # Vediamo i primi 2 esempi
        sample = data
        
        question = sample['Question']
        correct_answer = sample['Correct Answer']
        
        # Creiamo la lista delle opzioni partendo dai dati del sample
        choices = [
            correct_answer,
            sample['Incorrect Answer 1'],
            sample['Incorrect Answer 2'],
            sample['Incorrect Answer 3']
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
        questions.append({
            "prompt": prompt,
            "correct_letter": correct_letter
        })
    
    return questions

def gpqa_run(questions: list[dict[str, str]], max_run: int, test: bool) -> list[dict[str, str]]:
    responses = []
    run_counter = 0
    agent = LLM().create_custom_agent(LLM().get_tools())
    for q in questions:
        if run_counter >= max_run:
            break
        prompt = q["prompt"]
        correct_letter = q["correct_letter"]

        if test:
            response = extract_output(agent.invoke({"messages": [HumanMessage(content=prompt)]}))
        else:
            response = extract_output(call_graph(prompt))
        norm_res = normalize_number(response)
        responses.append({"question": prompt, "response": norm_res, "filtered_answer": "", "correct_letter": correct_letter, "answer_success": 0.0})
        run_counter += 1

    return responses

def gpqa_eval(responses: list[dict[str, str]]):
    correct = 0
    
    for res in responses:
        match = re.search(r"ANSWER:\s*([A-D])", res["response"], re.IGNORECASE)    
        res["filtered_answer"] = match.group(1).upper() if match else "N/A"
        
        if f"ANSWER: {res["correct_letter"]}" in res["response"]:
            correct += 1
            res["answer_success"] = 1.0

    accuracy = correct / len(responses)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Total: {len(responses)}")
    print(f"Correct: {correct}")

def save_eval_results(responses: list[dict[str, str]], model_name: str):
    with open(f"{model_name}_eval_results.json", "w") as f:
        json.dump(responses, f, indent=2)
