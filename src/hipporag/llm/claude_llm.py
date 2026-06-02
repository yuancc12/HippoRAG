import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import anthropic
from filelock import FileLock
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)


def cache_response_claude(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        key_data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        lock_file = self.cache_file_name + ".lock"

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                return message, metadata, True

        result = func(self, *args, **kwargs)
        message, metadata = result

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper


def dynamic_retry_decorator_claude(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper


class ClaudeLLM(BaseLLM):
    """Anthropic Claude LLM implementation."""

    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "ClaudeLLM":
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir: str, global_config: BaseConfig,
                 cache_filename: str = None,
                 **kwargs) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config
        self.llm_name = global_config.llm_name

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self.max_retries = kwargs.get("max_retries", global_config.max_retry_attempts)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)

        self._init_llm_config()

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__
        config_dict["llm_name"] = self.global_config.llm_name
        config_dict["generate_params"] = {
            "model": self.global_config.llm_name,
            "max_tokens": config_dict.get("max_new_tokens", 2048) or 2048,
            "temperature": config_dict.get("temperature", 0.0),
        }
        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response_claude
    @dynamic_retry_decorator_claude
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[str, dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)

        # Separate system prompt from conversation messages
        system_prompt = None
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                conversation.append({"role": msg["role"], "content": msg["content"]})

        if not conversation:
            conversation = [{"role": "user", "content": system_prompt or ""}]
            system_prompt = None

        api_params = {
            "model": params["model"],
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "messages": conversation,
        }
        if system_prompt:
            api_params["system"] = system_prompt

        logger.debug(f"Calling Anthropic Claude API with model: {params['model']}")
        response = self.client.messages.create(**api_params)

        response_message = response.content[0].text
        assert isinstance(response_message, str), "response_message should be a string"

        metadata = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "finish_reason": response.stop_reason,
        }

        return response_message, metadata
