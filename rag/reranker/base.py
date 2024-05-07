from typing import List
from abc import ABC, abstractmethod
from llama_index.core.schema import NodeWithScore


class Reranker(ABC):
	@abstractmethod
    def rerank(
    	query: str,
    	nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
    	pass

