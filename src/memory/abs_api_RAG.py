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
    def retrieve_relevant_nodes(self, query):
        # Retrieve relevant nodes from the knowledge base
        pass

    @abstractmethod
    def add_document(self, text) -> None:
        pass