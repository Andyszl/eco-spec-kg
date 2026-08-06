from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any
from typing import Protocol


class CompletionProvider(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


@dataclass(slots=True)
class MockProvider:
    response: str = '{"relations":[]}'

    def complete(self, system: str, prompt: str) -> str:
        del system, prompt
        return self.response


@dataclass(slots=True)
class OpenAICompatibleProvider:
    base_url: str
    api_key: str
    model: str = "Qwen/Qwen3-0.6B"
    timeout: int = 120
    max_tokens: int = 4096
    append_no_think: bool = False
    enable_thinking: bool | None = None
    last_raw_response: dict[str, Any] | None = None

    @classmethod
    def from_env(cls) -> "OpenAICompatibleProvider":
        thinking_value = os.environ.get("ECOSPEC_LLM_ENABLE_THINKING", "auto")
        if thinking_value.lower() in {"1", "true", "yes"}:
            enable_thinking: bool | None = True
        elif thinking_value.lower() in {"0", "false", "no"}:
            enable_thinking = False
        elif thinking_value.lower() == "auto":
            enable_thinking = None
        else:
            raise ValueError(
                "ECOSPEC_LLM_ENABLE_THINKING must be auto, true, or false"
            )
        return cls(
            base_url=os.environ.get("ECOSPEC_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.environ.get("ECOSPEC_LLM_API_KEY", "local-token"),
            model=os.environ.get("ECOSPEC_COMPLETION_MODEL", "Qwen/Qwen3-0.6B"),
            max_tokens=int(os.environ.get("ECOSPEC_LLM_MAX_TOKENS", "4096")),
            append_no_think=os.environ.get("ECOSPEC_LLM_APPEND_NO_THINK", "0")
            in {"1", "true", "True", "yes"},
            enable_thinking=enable_thinking,
        )

    def complete(self, system: str, prompt: str) -> str:
        user_content = prompt + "\n/no_think" if self.append_no_think else prompt
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        self.last_raw_response = result
        message = result["choices"][0]["message"]
        content = message.get("content")
        if content:
            return content
        return message.get("reasoning_content") or ""


class TransformersProvider:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the 'ml' optional dependencies") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def complete(self, system: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        output = self._model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated = output[0][inputs.input_ids.shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)
