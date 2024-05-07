from typing import List, Optional
from base import ChatEngine
from llama_index.legacy.llms.llama_cpp import LlamaCPP


DEFAULT_QA_PROMPT = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: {context_str}

Question: {question_str}

Answer: """


class LlamaChatEngine(ChatEngine):
    def __init__(
        self,
        model_url: Optional[str] = None,
        model_path: Optional[str] = None,
        prompt_template: str = DEFAULT_QA_PROMPT,
    ) -> None:
        self.llm = LlamaCPP(
            model_url=model_url,
            model_path=model_path,
        )
        self.prompt_template = prompt_template

    def ask(
        self,
        context: str,
        question: str,
    ) -> str:
        prompt = self.prompt_template.format(context_str=context, question_str=question)
        response = self.llm.complete(prompt)
        return response["choices"][0]["text"]
