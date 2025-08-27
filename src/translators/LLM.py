from .abs_api_model import AbsApiModel
from llama_index.core import PromptTemplate
from openai import AzureOpenAI, OpenAI
import json
import threading
from datetime import datetime
from uuid import uuid4

class LLM(AbsApiModel):
    def __init__(self, client:AzureOpenAI|OpenAI, model_name, system_prompt:PromptTemplate, temp=0.15, enable_rag = False, task_id: str | None = None, usage_log_path: str | None = None) -> None:
        super().__init__()
        self.client = client
        if model_name in ["gpt-4o-mini", "gpt-4o", "gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            self.model_name = model_name
        else:
            raise NotImplementedError
        self.system_prompt = system_prompt
        self.history = []
        self.temp = temp
        # usage logging
        self.task_id = task_id
        self.usage_log_path = usage_log_path
        self._call_index = 0
        self._usage_lock = threading.Lock()

    def send_request(self, input, ):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input},
            ],
            temperature= 1 if self.model_name in [ "gpt-5", "gpt-5-mini", "gpt-5-nano"] else self.temp,
        )
        text = response.choices[0].message.content.strip()
        # Best-effort usage logging
        try:
            usage = getattr(response, "usage", None)
            pt = getattr(usage, "prompt_tokens", None) if usage else None
            ct = getattr(usage, "completion_tokens", None) if usage else None
            tt = getattr(usage, "total_tokens", None) if usage else None
            provider = "azure-openai" if isinstance(self.client, AzureOpenAI) else "openai"
            if self.usage_log_path:
                with self._usage_lock:
                    call_idx = self._call_index
                    self._call_index += 1
                    rec = {
                        "request_id": str(uuid4()),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "task_id": self.task_id,
                        "provider": provider,
                        "model": self.model_name,
                        "category": "text",
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "total_tokens": tt,
                        "phrase_index": call_idx,
                    }
                    # Tag agent for aggregation
                    rec["extra"] = {"agent": "translator"}
                    with open(self.usage_log_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            # do not break translation on logging issues
            pass
        return text
