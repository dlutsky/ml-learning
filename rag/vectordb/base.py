from abc import ABC, abstractmethod
from typing import List, Sequence
from llama_index.core.schema import BaseNode, NodeWithScore, Document


class VectorDB(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
    ) -> List[NodeWithScore]:
        pass

    @abstractmethod
    def upsert_documents(
        self,
        documents: Sequence[Document],
    ) -> None:
        pass
