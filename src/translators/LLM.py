from openai import OpenAI
from .abs_model import AbsModel

class LLM(AbsModel):
    def __init__(self, model_name, system_prompt, temp = 0.15) -> None:
        self.client = OpenAI()
        if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]:
            self.model_name = model_name
        else:
            raise NotImplementedError
        self.system_prompt = system_prompt
        self.temp = temp
    
    def send_request(self, input):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system","content": self.system_prompt},
                {"role": "user", "content": input}
            ],
            temperature=self.temp
        )
        return response.choices[0].message.content.strip()
