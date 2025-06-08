from abc import ABC, abstractmethod 

"""
This is an abstract class for the model. It is used to define the methods that the model should have.
"""

class AbsApiRAG(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def load_knowledge_base(self, persist_dir, data_dir, num_retrievals):
        # Load the knowledge base from local storage or from scratch
        pass

    @abstractmethod
    def retrieve_relevant_nodes(self, query, use_window_retrieval=True):
        # Retrieve relevant nodes from the knowledge base based on the semantic similarity of the query
        # with optional window retrieval for adjacent nodes
        pass

    @abstractmethod
    def add_to_index(self, text_or_texts, chunk_size=50, chunk_overlap=5) -> None:
        pass