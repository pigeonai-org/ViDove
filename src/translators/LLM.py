from .abs_api_model import AbsApiModel
from llama_index.core import PromptTemplate
from openai import AzureOpenAI, OpenAI
import json
import threading
from datetime import datetime
from uuid import uuid4

from src.openai_responses import (
    SUPPORTED_OPENAI_TEXT_MODELS,
    create_response_text,
    extract_usage_tokens,
    normalize_text_model,
    provider_for_client,
)

class LLM(AbsApiModel):
    def __init__(self, client:AzureOpenAI|OpenAI, model_name, system_prompt:PromptTemplate, temp=0.15, enable_rag = False, task_id: str | None = None, usage_log_path: str | None = None) -> None:
        super().__init__()
        self.client = client
        self.model_name = normalize_text_model(model_name)
        self.system_prompt = system_prompt
        self.history = []
        self.temp = temp
        # usage logging
        self.task_id = task_id
        self.usage_log_path = usage_log_path
        self._call_index = 0
        self._usage_lock = threading.Lock()

    def send_request(self, input, ):
        text, response = create_response_text(
            self.client,
            model=self.model_name,
            instructions=str(self.system_prompt),
            input_value=input,
            temperature=1 if self.model_name in SUPPORTED_OPENAI_TEXT_MODELS else self.temp,
        )
        # Best-effort usage logging
        try:
            pt, ct, tt = extract_usage_tokens(response)
            provider = provider_for_client(self.client)
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
