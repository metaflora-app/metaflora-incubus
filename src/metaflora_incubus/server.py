"""Small local OpenAI-compatible server for an exported Incubus model."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str = Field(max_length=32_000)


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(max_length=64)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0, le=2)


@dataclass
class LocalRuntime:
    model_path: str
    model: Any
    tokenizer: Any
    generation_lock: Lock = field(default_factory=Lock)

    @classmethod
    def load(cls, model_path: str) -> LocalRuntime:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=False,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            weights_only=True,
        )
        if torch.cuda.is_available():
            model.to("cuda")
        model.eval()
        return cls(model_path=model_path, model=model, tokenizer=tokenizer)

    def complete(self, messages: list[ChatMessage], max_tokens: int, temperature: float) -> str:
        import torch

        rendered = self.tokenizer.apply_chat_template(
            [message.model_dump() for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(rendered, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with self.generation_lock, torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        prompt_tokens = encoded["input_ids"].shape[1]
        return self.tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)


def create_app(runtime: LocalRuntime) -> FastAPI:
    app = FastAPI(title="Metaflora Incubus v1", docs_url=None, redoc_url=None)

    @app.get("/v1/models")
    def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": "metaflora-incubus-v1", "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest) -> dict[str, object]:
        if request.model != "metaflora-incubus-v1":
            raise HTTPException(status_code=404, detail="local model is metaflora-incubus-v1")
        content = runtime.complete(request.messages, request.max_tokens, request.temperature)
        created = int(time())
        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": "metaflora-incubus-v1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    return app
