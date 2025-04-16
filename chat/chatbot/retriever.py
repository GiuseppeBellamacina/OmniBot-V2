from typing import Any, List

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever, RetrieverLike

from utilities.colorize import color


def remove_duplicates(docs: list[Document]) -> list[Document]:
    seen = set()
    return [
        d
        for d in docs
        if not (d.metadata.get("id") in seen or seen.add(d.metadata.get("id")))
    ]


class Retriever(BaseRetriever):
    compressor: BaseDocumentCompressor
    retriever: RetrieverLike
    embedder: CohereEmbeddings
    vectorstore: FAISS
    retrieval_threshold: float
    distance_threshold: float
    simplifier: float
    config: dict

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        followup_docs: List[Document] = None,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            Sequence of relevant documents
        """
        verbose = False
        if "verbose" in kwargs:
            verbose = kwargs["verbose"]
        callbacks = run_manager.get_child()
        docs = self.retriever.invoke(query, config={"callbacks": callbacks}, **kwargs)
        if verbose:
            print(
                color("[Retriever]", True, "blue"),
                ": Retrieved documents with standard method:",
                len(docs),
                sep="",
            )
        if not docs:
            return []

        compressed_docs = self.compressor.compress_documents(
            docs, query, callbacks=callbacks
        )
        if verbose:
            print(
                color("[Retriever]", True, "blue"),
                ": Compressed documents after first compression:",
                len(compressed_docs),
                sep="",
            )
        if not compressed_docs:
            return []

        filtered_docs = self.filter_by_similarity(
            compressed_docs, self.retrieval_threshold * self.simplifier
        )
        if verbose:
            print(
                color("[Retriever]", True, "blue"),
                ": Filtered documents after first filter:",
                len(filtered_docs),
                sep="",
            )
        if not filtered_docs:
            return []

        similar_docs = self.search_by_vector(filtered_docs)
        if verbose:
            print(
                color("[Retriever]", True, "blue"),
                ": Searched documents in vector store:",
                len(similar_docs),
                sep="",
            )
        if not similar_docs:
            return []

        if followup_docs:
            similar_docs.extend(followup_docs)
            if verbose:
                print(
                    color("[Retriever]", True, "blue"),
                    ": Added followup documents to similar documents:",
                    len(similar_docs),
                    sep="",
                )

        similar_docs = remove_duplicates(similar_docs)
        if verbose:
            print(
                color("[Retriever]", True, "blue"),
                ": Removed duplicates from similar documents:",
                len(similar_docs),
                sep="",
            )

        reranked_docs = self.compressor.compress_documents(
            similar_docs, query, callbacks=callbacks
        )
        if verbose:
            print(
                color("[Retriever]", True, "blue"),
                ": Reranked documents after second compression:",
                len(reranked_docs),
                sep="",
            )
        if not reranked_docs:
            return []

        refiltered_docs = self.filter_by_similarity(
            reranked_docs, self.retrieval_threshold
        )
        if verbose:
            print(
                color("[Retriever]", True, "blue"),
                ": Filtered documents after second filter:",
                len(refiltered_docs),
                sep="",
            )
        if not refiltered_docs:
            return []

        return sorted(refiltered_docs, key=lambda x: x.metadata.get("id"))

    def filter_by_similarity(self, docs: list[Document], threshold=0) -> list[Document]:
        if threshold == 0:
            return docs
        return [d for d in docs if d.metadata.get("relevance_score") > threshold]

    def filter_by_distance(self, docs: list[Document], threshold=0) -> list[Document]:
        if threshold == 0:
            return [d for (d, _) in docs]
        return [d for (d, score) in docs if score < threshold]

    def search_by_vector(self, docs: List[Document]) -> list[Document]:
        embedded_docs = self.embedder.embed_documents([d.page_content for d in docs])
        similar_docs = []
        for doc in embedded_docs:
            sim = self.vectorstore.similarity_search_with_score_by_vector(doc)
            if sim:
                similar_docs.extend(sim)
        if similar_docs:
            return self.filter_by_distance(similar_docs, self.distance_threshold)
        return []


class RetrieverBuilder:
    @classmethod
    def build(self, config) -> Retriever:
        retrieval_threshold = config["retrieval_threshold"]
        distance_threshold = config["distance_threshold"]
        simplifier = config["simplifier"]
        embedder = CohereEmbeddings(model=config["embedder"])
        vectorstore = FAISS.load_local(
            config["db"], embeddings=embedder, allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": config["k"]}
        )
        compressor = CohereRerank(model=config["reranker"], top_n=config["top_n"])
        print(color("[Retriever]", True, "blue"), ": Retriever initialized", sep="")

        return Retriever(
            compressor=compressor,
            retriever=retriever,
            embedder=embedder,
            vectorstore=vectorstore,
            retrieval_threshold=retrieval_threshold,
            distance_threshold=distance_threshold,
            simplifier=simplifier,
            config=config,
        )
