from abc import ABC, abstractmethod
import os
import time
import openai
import json

from dotenv import load_dotenv
from tenacity import wait_random_exponential, retry, stop_after_attempt


class LLM(ABC):
    def __init__(self):
        self.total_counter = 0
        self.total_time = 0
        self.total_tokens = 0
        self.cache = dict()
        self.cache_path = None

    def get_completion_from_messages(self, messages):
        res = self.get_completion_with_cache(messages)
        if res is None:
            res = self.get_completion_with_llm(messages)
            text = str(messages)
            self.cache[text] = res
            if self.total_counter % 10 == 0:
                self.save()
        return res

    @abstractmethod
    def get_completion_with_llm(self, messages):
        pass

    def get_completion_with_cache(self, messages):
        text = str(messages)
        if text in self.cache.keys():
            res = self.cache.get(text)
            return res
        else:
            return None

    def load(self):
        if self.cache_path is None:
            return
        if not os.path.exists(self.cache_path):
            return
        with open(self.cache_path, 'r', encoding='utf-8') as f:
            self.cache = json.load(f)

    def save(self):
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False)


class ChatGPTLLM(LLM):
    def __init__(self, model="gpt-3.5-turbo-0125", temperature=0, max_tokens=500, cache_path=None):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.temperature = temperature
        self.max_tokens = max_tokens
        load_dotenv()
        self.client = openai.OpenAI(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY")
        )
        self.cache_path = cache_path
        self.load()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_completion_with_llm(self, messages):
        self.total_counter += 1
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            frequency_penalty=0,
            presence_penalty=0,
            timeout=20
        )
        self.total_time += time.time() - start_time
        self.total_tokens = self.total_tokens + response.usage.total_tokens
        res = response.choices[0].message.content
        return res
