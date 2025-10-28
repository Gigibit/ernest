# core/llm_service.py
import hashlib
import logging
import os
import shutil
from collections import defaultdict
from typing import Dict

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from core.env import load_env_file

# Ensure environment variables from a local .env file are available before any
# Hugging Face models are downloaded. This allows users to provide HF tokens
# without relying on ``huggingface-cli login``.
load_env_file()

class LLMService:
    def __init__(self, profiles):
        self.profiles = profiles
        self._pipes = {}
        self._usage = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "invocations": 0}
        )
        self._log_hardware_state()

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
        self._pipes[name] = {"pipe": pipe, "profile": prof, "tokenizer": tok}
        return self._pipes[name]

    def invoke(self, name: str, prompt: str, **overrides) -> str:
        pipe_bundle = self._get_pipe(name)
        pipe = pipe_bundle["pipe"]
        prof = pipe_bundle["profile"]
        tokenizer = pipe_bundle["tokenizer"]
        max_tokens = overrides.get("max_new_tokens", prof.get("max", 512))
        prompt_tokens = self._count_tokens(tokenizer, prompt)
        temperature = overrides.get("temperature", prof.get("temp", 0.0))
        do_sample = overrides.get("do_sample")
        if do_sample is None:
            do_sample = temperature > 0

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "top_p": overrides.get("top_p", prof.get("top_p", 1.0)),
        }
        if do_sample and temperature:
            generation_kwargs["temperature"] = temperature
        out = pipe(
            prompt,
            **generation_kwargs,
        )
        generated = out[0]["generated_text"]
        completion = generated[len(prompt) :]
        completion_tokens = self._count_tokens(tokenizer, completion)
        self._register_usage(name, prompt_tokens, completion_tokens)
        return completion.strip()

    def prompt_hash(self, name: str, prompt: str) -> str:
        return hashlib.sha256((name + "::" + prompt).encode()).hexdigest()

    # ------------------------------------------------------------------
    # Accounting helpers
    # ------------------------------------------------------------------
    def reset_usage(self) -> None:
        """Reset the internal token accounting counters."""

        self._usage.clear()

    def get_usage_summary(self) -> Dict[str, Dict[str, int]]:
        """Return the aggregated token usage per profile and totals."""

        summary: Dict[str, Dict[str, int]] = {}
        total_prompt = 0
        total_completion = 0
        total_calls = 0
        for profile, data in self._usage.items():
            summary[profile] = dict(data)
            total_prompt += data["prompt_tokens"]
            total_completion += data["completion_tokens"]
            total_calls += data["invocations"]
        summary["__totals__"] = {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "invocations": total_calls,
        }
        return summary

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _count_tokens(tokenizer, text: str) -> int:
        if not text:
            return 0
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return int(encoded.input_ids.shape[-1])

    def _register_usage(
        self, profile: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        record = self._usage[profile]
        record["prompt_tokens"] += prompt_tokens
        record["completion_tokens"] += completion_tokens
        record["invocations"] += 1

    def _log_hardware_state(self) -> None:
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                devices = ", ".join(torch.cuda.get_device_name(i) for i in range(device_count))
                logging.info("CUDA devices detected: %s", devices)
                if shutil.which("nvcc") is None:
                    logging.warning(
                        "CUDA runtime available but nvcc compiler missing. Install NVIDIA drivers/toolkit for optimal performance."
                    )
            else:
                gpu_hint = os.environ.get("CUDA_VISIBLE_DEVICES")
                if gpu_hint and gpu_hint not in {"", "-1"}:
                    logging.warning(
                        "CUDA_VISIBLE_DEVICES=%s but torch reports no GPU. Missing drivers?",
                        gpu_hint,
                    )
                elif shutil.which("nvidia-smi"):
                    logging.warning(
                        "nvidia-smi detected without CUDA support in torch. Install or update NVIDIA drivers."
                    )
                else:
                    logging.info("CUDA not available; running migrations on CPU.")
        except Exception as exc:  # noqa: BLE001
            logging.warning("Unable to determine CUDA support: %s", exc)
