from typing import List
from abc import ABC, abstractmethod


class ChatEngine(ABC):
	@abstractmethod
    def ask(
        self,
        context: str,
        question: str,
    ) -> str:
        pass
