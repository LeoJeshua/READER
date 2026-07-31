#!/usr/bin/env python3
"""Evaluate MMLU-Pro through an OpenAI-compatible Chat Completions API.

The benchmark prompt and generation parameters follow
TIGER-AI-Lab/MMLU-Pro's ``evaluate_from_api.py``. Transport controls such as
concurrency, retries, and timeout do not change the evaluation prompt.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

CHOICES = "ABCDEFGHIJ"
OFFICIAL_RANDOM_SEED = 12345
OFFICIAL_MAX_TOKENS = 4000
OFFICIAL_TEMPERATURE = 0.0
OFFICIAL_TOP_P = 1.0
OFFICIAL_FREQUENCY_PENALTY = 0.0
OFFICIAL_PRESENCE_PENALTY = 0.0


def preprocess(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        example = dict(raw)
        example["options"] = [
            option for option in example["options"] if option != "N/A"
        ]
        grouped[example["category"]].append(example)
    return dict(grouped)


def format_example(
    question: str,
    options: list[str],
    cot_content: str = "",
) -> str:
    if not cot_content:
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]

    example = f"Question: {question}\nOptions: "
    for index, option in enumerate(options):
        example += f"{CHOICES[index]}. {option}\n"
    example += f"Answer: {cot_content}\n\n"
    return example


def build_prompt(
    question: dict[str, Any],
    cot_examples: list[dict[str, Any]],
) -> str:
    category = question["category"]
    instruction = (
        "The following are multiple choice questions (with answers) about "
        f"{category}.\nThink step by step and then output the answer in the "
        'format of "The answer is (X)" at the end.\n\n'
    )
    examples = "".join(
        format_example(
            example["question"],
            example["options"],
            example.get("cot_content", ""),
        )
        for example in cot_examples
    )
    test_input = format_example(question["question"], question["options"])
    return instruction + examples + test_input


def extract_answer(text: str) -> str | None:
    match = re.search(r"answer is \(?([A-J])\)?", text)
    if match:
        return match.group(1)

    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)

    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return match.group(0) if match else None


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized += "/v1"
    return normalized


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def usage_to_dict(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


async def call_api(
    client: AsyncOpenAI,
    *,
    model_name: str,
    prompt: str,
    max_tokens: int,
    max_retries: int,
    retry_delay: float,
    disable_thinking: bool = False,
) -> tuple[str | None, dict[str, int] | None, str | None, float]:
    last_error: str | None = None
    started = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            request: dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": OFFICIAL_TEMPERATURE,
                "max_tokens": max_tokens,
                "top_p": OFFICIAL_TOP_P,
                "frequency_penalty": OFFICIAL_FREQUENCY_PENALTY,
                "presence_penalty": OFFICIAL_PRESENCE_PENALTY,
            }
            if disable_thinking:
                request["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            completion = await client.chat.completions.create(**request)
            content = completion.choices[0].message.content
            if content is None:
                raise RuntimeError("API returned an empty message content")
            return (
                content.replace("**", ""),
                usage_to_dict(completion.usage),
                None,
                time.perf_counter() - started,
            )
        except Exception as exc:  # API clients expose several exception types.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (2**attempt))
    return None, None, last_error, time.perf_counter() - started


async def evaluate_question(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    *,
    model_name: str,
    question: dict[str, Any],
    cot_examples: list[dict[str, Any]],
    max_tokens: int,
    max_retries: int,
    retry_delay: float,
    disable_thinking: bool = False,
) -> dict[str, Any]:
    prompt = build_prompt(question, cot_examples)
    async with semaphore:
        response, usage, error, latency = await call_api(
            client,
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_delay=retry_delay,
            disable_thinking=disable_thinking,
        )

    result = dict(question)
    result["pred"] = extract_answer(response) if response is not None else None
    result["model_outputs"] = response
    result["usage"] = usage
    result["latency_sec"] = round(latency, 3)
    result["error"] = error
    return result


def result_key(example: dict[str, Any]) -> tuple[Any, str]:
    return example.get("question_id"), example["question"]


def load_existing_results(path: Path) -> dict[tuple[Any, str], dict[str, Any]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {result_key(row): row for row in rows}


def ordered_results(
    questions: list[dict[str, Any]],
    result_map: dict[tuple[Any, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    return [result_map[result_key(question)] for question in questions]


def score_results(
    results_by_subject: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    rng = random.Random(OFFICIAL_RANDOM_SEED)
    per_subject: dict[str, dict[str, Any]] = {}
    total_correct = 0
    total_count = 0
    total_unparsed = 0

    for subject in sorted(results_by_subject):
        correct = 0
        unparsed = 0
        rows = results_by_subject[subject]
        for row in rows:
            prediction = row.get("pred")
            if prediction is None:
                unparsed += 1
                prediction = CHOICES[rng.randrange(len(row["options"]))]
            if prediction == row["answer"]:
                correct += 1

        count = len(rows)
        per_subject[subject] = {
            "correct": correct,
            "wrong": count - correct,
            "total": count,
            "unparsed_randomly_scored": unparsed,
            "accuracy": correct / count if count else 0.0,
        }
        total_correct += correct
        total_count += count
        total_unparsed += unparsed

    return {
        "overall": {
            "correct": total_correct,
            "wrong": total_count - total_correct,
            "total": total_count,
            "unparsed_randomly_scored": total_unparsed,
            "accuracy": total_correct / total_count if total_count else 0.0,
        },
        "per_subject": per_subject,
    }


def selected_subjects(
    requested: str,
    available: set[str],
) -> list[str]:
    if requested == "all":
        return sorted(available)
    selected = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = sorted(set(selected) - available)
    if unknown:
        raise ValueError(f"unknown subjects: {', '.join(unknown)}")
    return sorted(dict.fromkeys(selected))


async def run(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    validation = json.loads(
        (data_dir / "validation.json").read_text(encoding="utf-8")
    )
    test = json.loads((data_dir / "test.json").read_text(encoding="utf-8"))
    validation_by_subject = preprocess(validation)
    test_by_subject = preprocess(test)
    subjects = selected_subjects(args.assigned_subjects, set(test_by_subject))

    base_url = normalize_base_url(args.base_url)
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=base_url,
        timeout=args.timeout,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=args.use_env_proxy),
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "implementation_reference": (
            "TIGER-AI-Lab/MMLU-Pro evaluate_from_api.py"
        ),
        "base_url": base_url,
        "model_name": args.model_name,
        "data_dir": str(data_dir.resolve()),
        "assigned_subjects": subjects,
        "max_per_subject": args.max_per_subject,
        "request_parameters": {
            "endpoint": "/v1/chat/completions",
            "max_tokens": args.max_tokens,
            "temperature": OFFICIAL_TEMPERATURE,
            "top_p": OFFICIAL_TOP_P,
            "frequency_penalty": OFFICIAL_FREQUENCY_PENALTY,
            "presence_penalty": OFFICIAL_PRESENCE_PENALTY,
            "stop": None,
            "chat_template_kwargs": (
                {"enable_thinking": False}
                if args.disable_thinking
                else None
            ),
        },
        "prompt_policy": {
            "shots": "all validation examples in the same subject",
            "validation_examples_per_subject": {
                subject: len(validation_by_subject.get(subject, []))
                for subject in subjects
            },
            "input_token_limit": None,
            "input_truncation": False,
        },
        "transport": {
            "concurrency": args.concurrency,
            "timeout_sec": args.timeout,
            "max_retries": args.max_retries,
            "use_env_proxy": args.use_env_proxy,
        },
        "official_unparsed_answer_seed": OFFICIAL_RANDOM_SEED,
    }
    atomic_write_json(output_dir / "run_config.json", run_config)

    all_results: dict[str, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    for subject in subjects:
        questions = test_by_subject[subject]
        if args.max_per_subject is not None:
            questions = questions[: args.max_per_subject]
        cot_examples = validation_by_subject.get(subject, [])
        result_path = output_dir / f"{subject}_result.json"
        existing = {} if args.overwrite else load_existing_results(result_path)
        pending = [
            question
            for question in questions
            if result_key(question) not in existing
            or existing[result_key(question)].get("model_outputs") is None
        ]
        completed_count = len(questions) - len(pending)
        print(
            f"[{subject}] total={len(questions)} resume={completed_count} "
            f"pending={len(pending)} shots={len(cot_examples)}",
            flush=True,
        )

        for start in range(0, len(pending), args.concurrency):
            batch = pending[start : start + args.concurrency]
            completed = await asyncio.gather(
                *[
                    evaluate_question(
                        client,
                        semaphore,
                        model_name=args.model_name,
                        question=question,
                        cot_examples=cot_examples,
                        max_tokens=args.max_tokens,
                        max_retries=args.max_retries,
                        retry_delay=args.retry_delay,
                        disable_thinking=args.disable_thinking,
                    )
                    for question in batch
                ]
            )
            for result in completed:
                existing[result_key(result)] = result
            atomic_write_json(
                result_path,
                ordered_results(
                    [
                        question
                        for question in questions
                        if result_key(question) in existing
                    ],
                    existing,
                ),
            )
            print(
                f"[{subject}] completed "
                f"{min(start + len(batch), len(pending))}/{len(pending)}",
                flush=True,
            )

        all_results[subject] = ordered_results(questions, existing)
        subject_score = score_results({subject: all_results[subject]})
        atomic_write_json(
            output_dir / f"{subject}_summary.json",
            subject_score["per_subject"][subject],
        )

    await client.close()
    summary = score_results(all_results)
    summary["elapsed_sec"] = round(time.perf_counter() - started, 3)
    summary["model_name"] = args.model_name
    summary["base_url"] = base_url
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official-style MMLU-Pro evaluation over an OpenAI API"
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible server, with or without the trailing /v1.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent / "data"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--assigned-subjects", default="all")
    parser.add_argument("--max-per-subject", type=int)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-tokens", type=int, default=OFFICIAL_MAX_TOKENS)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=false to the server chat template.",
    )
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Honor HTTP(S)_PROXY for external APIs; disabled for local vLLM.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    if args.max_per_subject is not None and args.max_per_subject <= 0:
        parser.error("--max-per-subject must be positive")
    return args


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run(args))
    overall = summary["overall"]
    print(
        f"[done] accuracy={overall['accuracy']:.6f} "
        f"({overall['correct']}/{overall['total']}) "
        f"unparsed={overall['unparsed_randomly_scored']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
