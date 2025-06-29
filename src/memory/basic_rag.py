from logging import Logger
from typing import List
import csv
import os
from pathlib import Path

from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.cohere import CohereEmbedding

CO_API_KEY = os.getenv("CO_API_KEY", None)

from ..memory.abs_api_RAG import AbsApiRAG


class CSVParser:
    """Custom CSV parser that treats each row as a separate document node."""
    
    @staticmethod
    def parse_csv_files(data_dir: str, encoding: str = 'utf-8') -> List[Document]:
        """
        Parse CSV files in the given directory and return a list of Documents.
        
        Args:
            data_dir: Directory containing CSV files
            encoding: File encoding (default: utf-8)
            
        Returns:
            List of Document objects, one per CSV row
        """
        documents = []
        csv_files = []
        
        # Find all CSV files in the directory
        for file_path in Path(data_dir).rglob("*.csv"):
            csv_files.append(file_path)
        
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', encoding=encoding, newline='') as f:
                    # Try to detect delimiter
                    sample = f.read(1024)
                    f.seek(0)
                    
                    # Use csv.Sniffer to detect delimiter
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
                        delimiter = dialect.delimiter
                    except Exception:
                        delimiter = ','  # Default to comma
                    
                    reader = csv.reader(f, delimiter=delimiter)
                    
                    # Read header row and skip it
                    headers = next(reader, None)
                    if headers is None:
                        continue
                    
                    # Process each data row
                    for row_idx, row in enumerate(reader, start=1):
                        if not any(cell.strip() for cell in row):  # Skip empty rows
                            continue
                            
                        # Create structured text from row data
                        row_text_parts = []
                        for i, cell in enumerate(row):
                            if i < len(headers) and cell.strip():
                                header = headers[i].strip()
                                cell_value = cell.strip()
                                if header and cell_value:
                                    row_text_parts.append(f"{header}: {cell_value}")
                        
                        if row_text_parts:
                            # Join all column data for this row
                            row_text = " | ".join(row_text_parts)
                            
                            # Create document with metadata
                            metadata = {
                                'file_path': str(csv_file),
                                'file_name': csv_file.name,
                                'row_number': row_idx,
                                'source_type': 'csv',
                                'headers': headers
                            }
                            
                            # Add individual column data to metadata for searchability
                            for i, cell in enumerate(row):
                                if i < len(headers) and cell.strip():
                                    header_key = f"col_{headers[i].strip().lower().replace(' ', '_')}"
                                    metadata[header_key] = cell.strip()
                            
                            document = Document(
                                text=row_text,
                                metadata=metadata
                            )
                            documents.append(document)
                            
            except Exception as e:
                print(f"Error parsing CSV file {csv_file}: {e}")
                continue
        
        return documents
    
    @staticmethod
    def parse_single_csv(file_path: str, encoding: str = 'utf-8') -> List[Document]:
        """
        Parse a single CSV file and return a list of Documents.
        
        Args:
            file_path: Path to the CSV file
            encoding: File encoding (default: utf-8)
            
        Returns:
            List of Document objects, one per CSV row
        """
        return CSVParser.parse_csv_files(os.path.dirname(file_path), encoding)

class BasicRAG(AbsApiRAG):
    def __init__(
        self,
        logger: Logger,
        domain="starcraft2",
        embedding_name: str = "embed-v4.0",
        # is_azure: bool = False,
    ) -> None:
        super().__init__()
        # if is_azure:
        #     self.embeddings = AzureOpenAIEmbedding(model=embedding_name)
        # else:
        self.embeddings = CohereEmbedding(model_name=embedding_name, api_key=CO_API_KEY)
        self.domain = domain
        self.index = None
        self.retriever = None
        self.memory = None
        self.logger = logger
        self.loaded = False

    def load_knowledge_base(self, data_dir, num_retrievals=5, window_size=1, chunk_size=50, chunk_overlap=10, parse_csv=True):
        Settings.embed_model = self.embeddings
        self.logger.info(
            f"Loading the model, set {Settings.embed_model} as embedding model"
        )
        self.window_size = window_size
        
        if data_dir is None:
            self.logger.info("Creating a new VectorStoreIndex with no initial data")
            index = VectorStoreIndex.from_documents([])
        else:    
            # remove the persist_dir check, always load from data_dir
            # 1. window retrieval, 2. CSV format, 3. Split ways
            self.logger.info("Loading the RAG from the data directory")
            
            all_documents = []
            
            if parse_csv:
                # Parse CSV files first
                self.logger.info("Parsing CSV files...")
                csv_documents = CSVParser.parse_csv_files(data_dir)
                if csv_documents:
                    self.logger.info(f"Found {len(csv_documents)} CSV rows to index")
                    all_documents.extend(csv_documents)
            
            # Load other document types using SimpleDirectoryReader
            try:
                other_documents = SimpleDirectoryReader(
                    data_dir,
                    exclude=["*.csv"] if parse_csv else None  # Exclude CSV if we already parsed them
                ).load_data()
                if other_documents:
                    self.logger.info(f"Found {len(other_documents)} other documents to index")
                    all_documents.extend(other_documents)
            except Exception as e:
                self.logger.warning(f"Error loading other documents: {e}")
            
            if not all_documents:
                self.logger.warning("No documents found to index")
                index = VectorStoreIndex.from_documents([])
            else:
                self.logger.info(f"Total documents to index: {len(all_documents)}")
                
                # For CSV documents, we don't need chunking since each row is already a chunk
                csv_docs = [doc for doc in all_documents if doc.metadata.get('source_type') == 'csv']
                other_docs = [doc for doc in all_documents if doc.metadata.get('source_type') != 'csv']
                
                final_documents = []
                
                # Add CSV documents as-is (no chunking)
                final_documents.extend(csv_docs)
                
                # Chunk other documents
                if other_docs:
                    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    chunked_docs = splitter.get_nodes_from_documents(other_docs)
                    # Convert nodes back to documents for consistency
                    for node in chunked_docs:
                        doc = Document(text=node.text, metadata=node.metadata)
                        final_documents.append(doc)
                
                index = VectorStoreIndex.from_documents(final_documents)
                #index.storage_context.persist(persist_dir)
            # else:
            #     self.logger.info("Loading the RAG from the storage directory")
            #     storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            #     index = load_index_from_storage(storage_context)        
        self.index = index
        self.retriever = index.as_retriever(similarity_top_k=num_retrievals)
        
        self.loaded = True
        self.logger.info(f"Model loaded with window retrieval (window_size={window_size})")

    def retrieve_relevant_nodes(self, query, use_window_retrieval=False):
        """
        Retrieve relevant nodes with optional window retrieval.
        
        Args:
            query: The search query
            use_window_retrieval: Whether to use window retrieval for adjacent nodes
        """
        if self.retriever is None:
            self.load_knowledge_base()

        if query is None or not isinstance(query, str) or not query.strip():
            self.logger.error("Empty or invalid query provided to retrieve_relevant_nodes.")
            return []
        
        ret = self.retriever.retrieve(query)
        return ret

    def add_csv_to_index(self, csv_file_path: str, encoding: str = 'utf-8'):
        """
        Add CSV data to the index. Each row becomes a separate document.
        
        Args:
            csv_file_path: Path to the CSV file
            encoding: File encoding (default: utf-8)
        """
        if self.index is None:
            self.load_knowledge_base()

        # Parse the CSV file
        csv_documents = CSVParser.parse_single_csv(csv_file_path, encoding)
        
        if not csv_documents:
            self.logger.warning(f"No data found in CSV file: {csv_file_path}")
            return
        
        self.logger.info(f"Adding {len(csv_documents)} CSV rows to index")
        
        # Convert Documents to nodes and insert
        nodes = []
        for doc in csv_documents:
            # For CSV, we don't chunk since each row is already a logical unit
            from llama_index.core.schema import TextNode
            node = TextNode(
                text=doc.text,
                metadata=doc.metadata
            )
            nodes.append(node)
        
        self.index.insert_nodes(nodes)

        # Update the retrievers
        self.retriever = self.index.as_retriever(similarity_top_k=5)
        self.logger.info(f"Successfully added {len(nodes)} CSV rows to index")

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
        for i in range(5):
            try:
                self.index.insert_nodes(nodes)
                break  # Exit loop if successful
            except Exception as e:
                self.logger.error(f"{i}: Error adding to index: {e}. Retrying...")
                continue

        # Update the retrievers
        self.retriever = self.index.as_retriever(similarity_top_k=5)
