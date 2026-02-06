# This rag need to improved as there is no converation history management

import os
from logging import Logger

from llama_index.core.schema import Document
from tavily import TavilyClient

from ..memory.basic_rag import BasicRAG

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Load to local store, 
# 1. refactor the retrieve_relevant_nodes function, if one of the similarity is less than 0.2 threhold, then web search
# 2. after the websearch, store all the results to local store


class TavilySearchRAG(BasicRAG):
    def __init__(
        self,
        logger: Logger,
        domain="starcraft2",
        tavily_api_key=TAVILY_API_KEY,
        max_results=5,
    ) -> None:
        super().__init__(logger=logger, domain=domain)
        self.domain = domain
        self.tavily_client = TavilyClient(tavily_api_key)
        self.logger = logger
        self.max_results = max_results

    def retrieve_relevant_nodes(self, query: str, use_window_retrieval=True) -> list[Document]:
        #added this function for testing
        # Note: use_window_retrieval is ignored for web search as it doesn't apply
        if not isinstance(query, str) or not query.strip():
            self.logger.error("Empty or invalid query provided to Tavily search.")
            return []

        query = query.strip()

        try:
            responses = self.tavily_client.search(
                query=query,
                max_results=self.max_results,
                # search_depth="advanced",
                # include_domains=[self.domain],
            )
            return [Document(text=result["content"]) for result in responses.get("results", [])]
        except Exception as e:
            self.logger.exception(f"Error during Tavily search: {e}")
            return []
