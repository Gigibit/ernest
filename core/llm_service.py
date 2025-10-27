# core/llm_service.py
import logging, hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LLMService:
    def __init__(self, profiles):
        self.profiles = profiles
        self._pipes = {}

    def _get_pipe(self, name):
        if name in self._pipes:
            return self._pipes[name]
        prof = self.profiles[name]
        logging.info(f"Loading model {prof['id']} for {name}")
        tok = AutoTokenizer.from_pretrained(prof["id"], use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            prof["id"], device_map="auto", torch_dtype="auto", trust_remote_code=True
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tok)
        self._pipes[name] = (pipe, prof)
        return self._pipes[name]

    def invoke(self, name: str, prompt: str, **overrides) -> str:
        pipe, prof = self._get_pipe(name)
        max_tokens = overrides.get("max_new_tokens", prof.get("max", 512))
        out = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=prof.get("temp", 0.0),
            top_p=prof.get("top_p", 1.0),
        )
        return out[0]["generated_text"][len(prompt):].strip()

    def prompt_hash(self, name: str, prompt: str) -> str:
        return hashlib.sha256((name + "::" + prompt).encode()).hexdigest()
