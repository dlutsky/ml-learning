from typing import List, Sequence
from sqlalchemy import make_url
from llama_index.core.schema import NodeWithScore, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.legacy.vector_stores.postgres import PGVectorStore
from llama_index.legacy.embeddings import OpenAIEmbedding
from base import VectorDB


class PGVectorDB(VectorDB):
    def __init__(
        self,
        connection_string: str,
        db_name: str,
        table_name: str,
    ) -> None:
        url = make_url(connection_string)
        self.vector_store = PGVectorStore.from_params(
            database=db_name,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name=table_name,
            embed_dim=1536,
        )
        self.embed_model = OpenAIEmbedding()

    def search(
        self,
        query: str,
    ) -> List[NodeWithScore]:
        retriever = self.vector_store.as_retriever(embed_model=self.embed_model)
        return retriever.retrieve(query)

    def upsert_documents(
        self,
        documents: Sequence[Document],
    ) -> None:
        splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embed_model
        )
        nodes = splitter.get_nodes_from_documents(documents)
        self.vector_store.insert_nodes(nodes)
