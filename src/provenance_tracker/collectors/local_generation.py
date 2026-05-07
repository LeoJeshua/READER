from __future__ import annotations

import gc
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from provenance_tracker.datasets.schemas import ProbeSample, ResponseRecord
from provenance_tracker.utils.io import write_jsonl


@dataclass(slots=True)
class GenerationConfig:
    model_name_or_path: str
    label: str
    run_id: str
    max_input_length: int = 512
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_chat_template: bool = True
    system_prompt: str | None = None
    batch_size: int = 4


class LocalResponseCollector:
    """Load a local causal LM once and generate responses for a probe iterable."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Pad on the left for decoder-only batched generation.
        self.tokenizer.padding_side = "left"

        dtype = getattr(torch, config.dtype, torch.bfloat16)
        if config.device == "auto":
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                dtype=dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            self._input_device = next(self.model.parameters()).device
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                dtype=dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            ).to(config.device)
            self._input_device = torch.device(config.device)
        self.model.eval()

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _format_prompt(self, prompt: str) -> str:
        if not self.config.use_chat_template or not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return prompt

    def collect(self, probes: Iterable[ProbeSample]) -> list[ResponseRecord]:
        torch.manual_seed(self.config.seed)
        probes = list(probes)
        records: list[ResponseRecord] = []

        bs = max(1, self.config.batch_size)
        for i in range(0, len(probes), bs):
            chunk = probes[i : i + bs]
            texts = [self._format_prompt(p.prompt) for p in chunk]
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_input_length,
                padding=True,
            ).to(self._input_device)
            input_lens = inputs["attention_mask"].sum(dim=1).tolist()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature if self.config.do_sample else 1.0,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            timestamp = datetime.now(timezone.utc).isoformat()
            for probe, output_ids, in_len in zip(chunk, outputs, input_lens):
                # With left padding, the prompt tokens are at the tail of input_ids;
                # generated continuation follows after the full input window.
                gen_ids = output_ids[inputs["input_ids"].shape[1] :]
                response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                records.append(
                    ResponseRecord(
                        sample_id=probe.sample_id,
                        prompt=probe.prompt,
                        response=response,
                        label=self.config.label,
                        model_name=self.config.model_name_or_path,
                        run_id=self.config.run_id,
                        timestamp=timestamp,
                        probe_category=probe.probe_category,
                    )
                )
        return records


def build_mock_records(
    probes: Iterable[ProbeSample],
    *,
    label: str,
    model_name: str,
    run_id: str,
    responder: Callable[[str], str],
) -> list[ResponseRecord]:
    timestamp = datetime.now(timezone.utc).isoformat()
    return [
        ResponseRecord(
            sample_id=probe.sample_id,
            prompt=probe.prompt,
            response=responder(probe.prompt),
            label=label,
            model_name=model_name,
            run_id=run_id,
            timestamp=timestamp,
            probe_category=probe.probe_category,
        )
        for probe in probes
    ]


def save_records(records: list[ResponseRecord], output_path: str) -> str:
    write_jsonl(output_path, [record.to_dict() for record in records])
    return output_path
