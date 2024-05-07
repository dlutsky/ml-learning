from typing import List
from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder


class CrossEncoderReranker(Reranker):
	def __init__(
        self,
    ) -> None:
    	self.cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    def rerank(
    	query: str,
        nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
    	node_pairs = [[query, node.get_text()] for node in nodes]
    	scores = self.cross_encoder_model.predict(node_pairs)
    	for idx in range(len(nodes)):
    		nodes[idx].score = scores[idx]
    	nodes = sorted(hits, key=lambda x: x.score, reverse=True)
    	return nodes
