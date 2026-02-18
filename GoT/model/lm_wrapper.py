import math
from GoT.model.graph_model import call_graph, invoke_graph
from lm_eval.api.registry import register_model
from lm_eval.api.model import LM

from GoT.model.utils.utils import extract_output, normalize_number


class LangGraphLM:
    def __init__(self):
        pass

    def generate(self, requests, max_new_tokens=None):
        """
        requests: list di dict {"prompt": "<prompt>"}
        Restituisce list di stringhe
        """
        outputs = []
        for r in requests:
            prompt = r["prompt"]
            result = call_graph(prompt)
            outputs.append(result["output"])
        return outputs


# Register your custom model with lm_eval
@register_model("langgraph_lm")
class LangGraphLMWrapper(LM):
    def __init__(self, model_args=""):
        super().__init__()
        self.lm = LangGraphLM()

    def generate_until(self, requests, until=None, max_new_tokens=None, **kwargs):
        """
        Generate text until a stopping condition is met.
        """
        outputs = []
        for i, request in enumerate(requests):
            try:
                # Extract the question directly from request.doc
                # This is much simpler than parsing the full prompt
                if hasattr(request, "doc") and isinstance(request.doc, dict):
                    question = request.doc.get("question", "")
                else:
                    # Fallback: try to extract from arguments
                    if hasattr(request, "arguments") and request.arguments:
                        full_prompt = request.arguments[0][0]
                        # Just take everything after the last "Answer:" as the question
                        last_answer_idx = full_prompt.rfind("Answer:")
                        if last_answer_idx != -1:
                            question = full_prompt[
                                last_answer_idx + 8 :
                            ].strip()  # 8 = len("Answer:")
                        else:
                            question = full_prompt
                    else:
                        question = str(request)

                # Call invoke_graph with just the question
                result = call_graph(question)

                # Extract the final answer from the result
                output = (
                    result["output"]
                    if isinstance(result, dict) and "output" in result
                    else ""
                )
                normalize_output = normalize_number(output)

                outputs.append(normalize_output if normalize_output else "")

            except Exception as e:
                print(f"Error in request {i}: {e}")
                import traceback

                traceback.print_exc()
                outputs.append("")

        return outputs

    def loglikelihood(self, requests):
        return super().loglikelihood(requests)

    def loglikelihood_rolling(self, requests):
        return super().loglikelihood_rolling(requests)


@register_model("langgraph_bigbench")
class LangGraphBigBenchWrapper(LM):
    """
    Wrapper specifico per BigBench tasks.
    Supporta principalmente generate_until e loglikelihood per multiple choice.
    """

    def __init__(self, model_args=""):
        super().__init__()

    def _extract_text_from_request(self, request):
        """
        Estrae il testo dalla request BigBench.
        BigBench passa il prompt completo in doc['inputs'].
        """
        # Prova con request.doc - BigBench usa il campo 'inputs'
        if hasattr(request, "doc") and isinstance(request.doc, dict):
            doc = request.doc

            # BigBench mette il prompt completo in 'inputs'
            if "inputs" in doc and doc["inputs"]:
                prompt = str(doc["inputs"]).strip()
                return prompt

            # Fallback per altri campi
            for field in ["input", "question", "text", "prompt", "instruction"]:
                if field in doc and doc[field]:
                    return str(doc[field])

        # Fallback: prova con request.arguments
        if hasattr(request, "arguments") and request.arguments:
            try:
                full_prompt = request.arguments[0][0]
                if full_prompt and len(str(full_prompt).strip()) > 1:
                    return full_prompt
            except (IndexError, TypeError):
                pass

        return str(request)

    def generate_until(self, requests, until=None, max_new_tokens=None, **kwargs):
        """
        Genera testo fino a una condizione di stop.
        Usato dalla maggior parte dei task di BigBench.
        """
        outputs = []

        # Se until non è specificato, usa default sensati
        if not until:
            until = ["\n\n", "\n"]
        elif isinstance(until, str):
            until = [until]

        print(f"DEBUG - Using stopping tokens: {until}")

        for i, request in enumerate(requests):
            try:
                question = self._extract_text_from_request(request)

                result = call_graph(question)

                # Estrai l'output
                output = extract_output(result)

                # Taglia l'output al primo stopping token
                if output and isinstance(output, str):
                    for stop_seq in until:
                        if stop_seq in output:
                            output = output[: output.index(stop_seq)]

                outputs.append(output.strip() if output else "")

            except Exception as e:
                print(f"Error in BigBench generate_until request {i}: {e}")
                import traceback

                traceback.print_exc()
                outputs.append("")

        return outputs

    def loglikelihood(self, requests):
        """
        Calcola la log-likelihood per BigBench multiple choice tasks.

        Riceve una lista di tuple: (context, continuation)
        Restituisce lista di tuple: (log_likelihood_score, is_greedy)

        NOTA: Poiché invoke_graph non fornisce logits, usiamo un'euristica.
        Se hai bisogno di precision massima, modifica invoke_graph per restituire logits.
        """
        outputs = []

        for i, request in enumerate(requests):
            try:
                # Estrai context e continuation
                if isinstance(request, tuple) and len(request) == 2:
                    context, continuation = request
                else:
                    # Fallback
                    context = ""
                    continuation = str(request)

                # Prepara il prompt per il modello
                full_text = context + continuation

                # Chiama il grafo
                result = call_graph(full_text)
                generated_output = extract_output(result)

                # Calcola il score di likelihood
                score = self._calculate_likelihood_score(
                    generated_output, continuation, context
                )

                outputs.append((score, False))

            except Exception as e:
                print(f"Error in BigBench loglikelihood request {i}: {e}")
                import traceback

                traceback.print_exc()
                outputs.append((float("-inf"), False))

        return outputs

    def loglikelihood_rolling(self, requests):
        """
        Per task che richiedono rolling loglikelihood.
        Di solito per task di perplexity.
        """
        return self.loglikelihood(requests)

    def _calculate_likelihood_score(
        self, generated_text, target_continuation, context=""
    ):
        """
        Calcola un score di likelihood per BigBench.

        Strategie:
        1. Se la generazione contiene esattamente la continuation target -> score alto (0.0)
        2. Se contiene la continuation parzialmente -> score medio
        3. Altrimenti -> score basso (log della similarità)

        Questo è un workaround poiché non abbiamo i veri logits.
        """
        gen_text = str(generated_text).strip().lower()
        target = str(target_continuation).strip().lower()

        # Caso 1: Match esatto
        if gen_text == target:
            return 0.0  # log(1) = 0, massima likelihood

        # Caso 2: Target è contenuto nel testo generato
        if target in gen_text:
            # Parziale match, score moderato
            return math.log(0.8)

        # Caso 3: Calcola sovrapposizione di parole
        gen_words = set(gen_text.split())
        target_words = set(target.split())

        if not target_words:
            return 0.0

        overlap = len(gen_words & target_words) / len(target_words)

        if overlap > 0:
            # Maggiore è l'overlap, più alto lo score
            return math.log(overlap)
        else:
            # Nessuna sovrapposizione
            return float("-inf")
