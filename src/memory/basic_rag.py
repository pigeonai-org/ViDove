import os
from logging import Logger

from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from ..memory.abs_api_RAG import AbsApiRAG
from ..translators.prompts import system_prompt

# SYSTEM_PROMPT = system_prompt

# Default persist directory and data directory
PERSIST_DIR = "./storage/basic_rag"
DATA_DIR = "domain_dict/SC2"

# text_qa_template


class BasicRAG(AbsApiRAG):
    def __init__(
        self,
        logger: Logger,
        domain="starcraft2",
        embedding_name: str = "text-embedding-3-small",
        is_azure: bool = False,
    ) -> None:
        super().__init__()
        if is_azure:
            self.embeddings = AzureOpenAIEmbedding(model=embedding_name)
        else:
            self.embeddings = OpenAIEmbedding(model=embedding_name)
        self.domain = domain
        self.index = None
        self.retriever = None
        self.memory = None
        self.logger = logger
        self.loaded = False

    def load_knowledge_base(self, persist_dir=PERSIST_DIR, data_dir=DATA_DIR, num_retrievals=5):
        Settings.embed_model = self.embeddings
        self.logger.info(
            f"Loading the model, set {Settings.embed_model} as embedding model"
        )
        if persist_dir is None and data_dir is None:
            self.logger.info("Creating one empty index without any data")
            self.index = VectorStoreIndex()
        else:    
            if not os.path.exists(persist_dir):
                self.logger.info("Loading the RAG from the data directory")
                documents = SimpleDirectoryReader(data_dir).load_data()
                index = VectorStoreIndex.from_documents(
                    documents,
                    transformations=[SentenceSplitter(chunk_size=40, chunk_overlap=5)],
                )
                index.storage_context.persist(persist_dir)
            else:
                self.logger.info("Loading the RAG from the storage directory")
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                index = load_index_from_storage(storage_context)        
        self.index = index
        self.retriever = index.as_retriever(similarity_top_k=num_retrievals)
        self.loaded = True
        self.logger.info("Model loaded")

    def retrieve_relevant_nodes(self, query):
        if self.retriever is None:
            self.load_knowledge_base()
        return self.retriever.retrieve(query)

    def add_to_index(self, text_or_texts, chunk_size=50, chunk_overlap=5):
        """
        Add one or more text documents to the index.
        
        Args:
            texts: A single string or a list of strings to add to the index.
            chunk_size: Size of each text chunk (default: 50).
            chunk_overlap: Overlap between chunks (default: 5).
        """
        if self.index is None:
            self.load_knowledge_base()

        # Normalize input to a list of Documents
        if isinstance(text_or_texts, str):
            documents = [Document(text=text_or_texts)]
        elif isinstance(text_or_texts, list):
            documents = [Document(text=text) for text in text_or_texts]
        else:
            raise ValueError("Input 'texts' must be a string or a list of strings.")

        # Split documents into nodes and insert
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents)
        self.index.insert_nodes(nodes)

        # Update the retriever
        self.retriever = self.index.as_retriever(similarity_top_k=5)
