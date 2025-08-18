from .abs_api_model import AbsApiModel
from llama_index.core import PromptTemplate
from openai import AzureOpenAI, OpenAI



# add RAG support in here


class LLM(AbsApiModel):
    def __init__(self, client:AzureOpenAI|OpenAI, model_name, system_prompt:PromptTemplate, temp=0.15, enable_rag = False) -> None:
        super().__init__()
        self.client = client
        if model_name in ["gpt-4o-mini", "gpt-4o", "gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            self.model_name = model_name
        else:
            raise NotImplementedError
        self.system_prompt = system_prompt
        self.history = []
        self.temp = temp

    def send_request(self, input, ):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input},
            ],
            temperature= 1 if self.model_name in [ "gpt-5", "gpt-5-mini", "gpt-5-nano"] else self.temp,
        )
        return response.choices[0].message.content.strip()
