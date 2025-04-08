# This rag need to improved as there is no converation history management

import os
from logging import Logger

from llama_index.core.schema import Document
from tavily import TavilyClient

from ..memory.basic_rag import BasicRAG

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


class TavilySearchRAG(BasicRAG):
    def __init__(
        self,
        logger: Logger,
        domain="starcraft2",
        tavily_api_key=TAVILY_API_KEY,
    ) -> None:
        super().__init__(logger=logger, domain=domain)
        self.domain = domain
        self.tavily_client = TavilyClient(tavily_api_key)
        self.logger = logger

    def seach_tavily(self, query: str, max_results: int = 5) -> list[Document]:
        #added this function for testing
        if not isinstance(query, str) or not query.strip():
            self.logger.error("Empty or invalid query provided to Tavily search.")
            return []

        query = query.strip()

        try:
            responses = self.tavily_client.search(
                query=query,
                max_results=max_results,
                # search_depth="advanced",
                # include_domains=[self.domain],
            )
            return [Document(text=result["content"]) for result in responses.get("results", [])]
        except Exception as e:
            self.logger.exception(f"Error during Tavily search: {e}")
            return []    

    def _seach_tavily(self, query: str, max_results: int = 5) -> list[Document]:
        responses = self.tavily_client.search(
            query,  # TODO: might needed to refine the query a bit to get better results
            max_results=max_results,
            # search_depth="advanced",
            # include_domains=[self.domain],
        )

        return [result["content"] for result in responses["results"]]
