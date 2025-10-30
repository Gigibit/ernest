# core/llm_service.py
import hashlib
import json
import logging
import os
import shutil
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, Mapping

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
        self._shared_models = {}
        self._usage = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "invocations": 0}
        )
        self.mode = (
            (os.environ.get("ERNEST_LLM_MODE") or os.environ.get("CHRISTOPHE_LLM_MODE") or "live")
            .strip()
            .lower()
        )
        self._replay_records = {}
        self._replay_profiles = {}
        self._replay_default = deque()
        self._replay_path = None
        if self.mode == "mock":
            self._initialise_mock_store()
        self._log_hardware_state()

    def _get_pipe(self, name):
        if name in self._pipes:
            return self._pipes[name]
        prof = self.profiles[name]
        model_id = prof["id"]

        if model_id in self._shared_models:
            logging.info("Reusing loaded model %s for profile %s", model_id, name)
            shared = self._shared_models[model_id]
        else:
            logging.info("Loading model %s for profile %s", model_id, name)
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True
            )
            pipe = pipeline("text-generation", model=model, tokenizer=tok)
            shared = {"pipe": pipe, "tokenizer": tok}
            self._shared_models[model_id] = shared

        self._pipes[name] = {"pipe": shared["pipe"], "profile": prof, "tokenizer": shared["tokenizer"]}
        return self._pipes[name]

    def invoke(self, name: str, prompt: str, **overrides) -> str:
        if self.mode == "mock":
            completion = self._mock_completion(name, prompt)
            self._register_mock_usage(name, prompt, completion)
            return completion

        pipe_bundle = self._get_pipe(name)
        pipe = pipe_bundle["pipe"]
        prof = pipe_bundle["profile"]
        tokenizer = pipe_bundle["tokenizer"]
        max_tokens = overrides.get("max_new_tokens", prof.get("max", 512))
        prompt_tokens = self._count_tokens(tokenizer, prompt)

        prompt, prompt_tokens, max_tokens = self._enforce_context_window(
            tokenizer, prompt, prompt_tokens, max_tokens
        )
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

    def _enforce_context_window(
        self, tokenizer, prompt: str, prompt_tokens: int, max_tokens: int
    ):
        """Ensure prompt + generation do not exceed the model context window."""

        try:
            max_context = int(getattr(tokenizer, "model_max_length", 0))
        except (TypeError, ValueError):  # noqa: BLE001
            max_context = 0

        # Some tokenizers advertise extremely large context lengths (e.g. GPT-2
        # returns 1000000000000000019884624838656). Clamp to a reasonable
        # default in that case so our math stays numerically stable.
        if not max_context or max_context > 32768:
            max_context = 32768

        if prompt_tokens >= max_context:
            # Trim the prompt so that there is at least one token left for
            # generation. We keep the most recent portion of the prompt because
            # the models primarily operate autoregressively on the tail of the
            # context window.
            target_prompt_tokens = max(1, max_context - max(1, max_tokens))
            prompt = self._tail_truncate(tokenizer, prompt, target_prompt_tokens)
            prompt_tokens = target_prompt_tokens

        # Ensure the planned completion fits inside the remaining context.
        available_for_completion = max_context - prompt_tokens
        if available_for_completion <= 0:
            max_tokens = 1
        else:
            max_tokens = max(1, min(max_tokens, available_for_completion))

        return prompt, prompt_tokens, max_tokens

    @staticmethod
    def _tail_truncate(tokenizer, prompt: str, target_tokens: int) -> str:
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        ids = encoded.input_ids[0]
        if ids.shape[-1] <= target_tokens:
            return prompt
        truncated_ids = ids[-target_tokens:]
        return tokenizer.decode(truncated_ids, skip_special_tokens=False)

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

    # ------------------------------------------------------------------
    # Mock replay helpers
    # ------------------------------------------------------------------
    def _initialise_mock_store(self) -> None:
        replay_env = os.environ.get("ERNEST_LLM_REPLAY_PATH") or os.environ.get(
            "CHRISTOPHE_LLM_REPLAY_PATH"
        )
        if replay_env:
            path = Path(replay_env).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                try:
                    payload = json.loads(path.read_text())
                except json.JSONDecodeError as exc:
                    logging.warning(
                        "Unable to decode mock LLM transcript %s: %s", path, exc
                    )
                else:
                    self._replay_path = path
                    self._load_mock_payload(payload)
                    logging.info(
                        "LLM mock mode active; replaying completions from %s", path
                    )
            else:
                logging.warning(
                    "LLM mock mode requested but replay file %s was not found", path
                )
        if not self._replay_records and not self._replay_profiles and not self._replay_default:
            logging.info(
                "LLM mock mode active without transcript; falling back to stub completions."
            )

    def _load_mock_payload(self, payload) -> None:
        if isinstance(payload, Mapping):
            records = payload.get("records")
            if isinstance(records, Mapping):
                for key, value in records.items():
                    self._replay_records[str(key)] = self._normalise_values(value)
            # Allow top-level hashes without explicit "records" wrapper.
            for key, value in payload.items():
                if key in {"records", "profiles", "default"}:
                    continue
                if self._looks_like_hash(key):
                    self._replay_records.setdefault(str(key), self._normalise_values(value))
            profiles = payload.get("profiles")
            if isinstance(profiles, Mapping):
                for profile, value in profiles.items():
                    self._replay_profiles[str(profile)] = deque(
                        self._normalise_values(value)
                    )
            default = payload.get("default")
            if default is not None:
                self._replay_default = deque(self._normalise_values(default))
        elif isinstance(payload, Iterable):
            for entry in payload:
                if isinstance(entry, Mapping):
                    key = entry.get("key") or entry.get("hash")
                    if key:
                        self._replay_records[str(key)] = self._normalise_values(
                            entry.get("completion")
                            or entry.get("text")
                            or entry.get("value")
                            or entry.get("response")
                            or ""
                        )
                    profile = entry.get("profile")
                    if profile:
                        self._replay_profiles.setdefault(str(profile), deque()).extend(
                            self._normalise_values(
                                entry.get("completion")
                                or entry.get("text")
                                or entry.get("value")
                                or entry.get("response")
                                or ""
                            )
                        )

    @staticmethod
    def _looks_like_hash(key: str) -> bool:
        return len(key) >= 16 and all(ch in "0123456789abcdef" for ch in key.lower())

    @staticmethod
    def _normalise_values(raw) -> deque:
        values = []
        if raw is None:
            values.append("")
        elif isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, Mapping):
            if "values" in raw and isinstance(raw["values"], Iterable):
                for item in raw["values"]:
                    values.extend(LLMService._normalise_values(item))
            elif "completion" in raw:
                values.extend(LLMService._normalise_values(raw["completion"]))
            elif "text" in raw:
                values.extend(LLMService._normalise_values(raw["text"]))
            else:
                values.append(json.dumps(raw))
        elif isinstance(raw, Iterable):
            for item in raw:
                values.extend(LLMService._normalise_values(item))
        else:
            values.append(str(raw))
        if not values:
            values.append("")
        return deque(values)

    def _cycle(self, values: deque) -> str:
        if not values:
            return ""
        result = values[0]
        if len(values) > 1:
            values.rotate(-1)
        return result

    def _mock_completion(self, profile: str, prompt: str) -> str:
        cache_key = self.prompt_hash(profile, prompt)
        record = self._replay_records.get(cache_key)
        if record:
            return self._cycle(record)
        profile_seq = self._replay_profiles.get(profile)
        if profile_seq:
            return self._cycle(profile_seq)
        if self._replay_default:
            return self._cycle(self._replay_default)
        # Fallback stub embeds prompt snippet for debugging but avoids newline-heavy output.
        snippet = prompt.strip().splitlines()
        snippet = snippet[0][:120] if snippet else ""
        return json.dumps(
            {
                "mode": "mock",
                "profile": profile,
                "detail": "No mock transcript found; returning stub response.",
                "prompt_preview": snippet,
            }
        )

    def _register_mock_usage(self, profile: str, prompt: str, completion: str) -> None:
        record = self._usage[profile]
        record["prompt_tokens"] += len(prompt.split())
        record["completion_tokens"] += len(completion.split())
        record["invocations"] += 1
