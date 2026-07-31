from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from evaluate import (
    OFFICIAL_FREQUENCY_PENALTY,
    OFFICIAL_PRESENCE_PENALTY,
    OFFICIAL_TEMPERATURE,
    OFFICIAL_TOP_P,
    build_prompt,
    call_api,
    extract_answer,
)


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Reasoning. **The answer is (C).**"
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
        )


class EvaluateTests(unittest.TestCase):
    def test_prompt_uses_official_api_format(self) -> None:
        validation = {
            "question": "Training question?",
            "options": ["one", "two"],
            "cot_content": "A: Let's think step by step. The answer is (B).",
            "category": "math",
        }
        question = {
            "question": "Test question?",
            "options": ["left", "right"],
            "category": "math",
        }
        prompt = build_prompt(question, [validation])
        self.assertIn("about math.\nThink step by step", prompt)
        self.assertIn("Options: A. one\nB. two\n", prompt)
        self.assertIn("Answer: Let's think step by step.", prompt)
        self.assertTrue(prompt.endswith("Answer: Let's think step by step.\n\n"))

    def test_answer_extraction_matches_official_fallbacks(self) -> None:
        self.assertEqual(extract_answer("The answer is (D)."), "D")
        self.assertEqual(extract_answer("work\nAnswer: E"), "E")
        self.assertEqual(extract_answer("A or B, finally C"), "C")
        self.assertIsNone(extract_answer("no option selected"))

    def test_api_payload_matches_official_openai_branch(self) -> None:
        completions = _FakeCompletions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        response, usage, error, _ = asyncio.run(
            call_api(
                client,
                model_name="test-model",
                prompt="prompt",
                max_tokens=4000,
                max_retries=0,
                retry_delay=0,
            )
        )
        self.assertEqual(response, "Reasoning. The answer is (C).")
        self.assertIsNone(error)
        self.assertEqual(usage["total_tokens"], 120)
        self.assertEqual(completions.kwargs["max_tokens"], 4000)
        self.assertEqual(
            completions.kwargs["temperature"], OFFICIAL_TEMPERATURE
        )
        self.assertEqual(completions.kwargs["top_p"], OFFICIAL_TOP_P)
        self.assertEqual(
            completions.kwargs["frequency_penalty"],
            OFFICIAL_FREQUENCY_PENALTY,
        )
        self.assertEqual(
            completions.kwargs["presence_penalty"],
            OFFICIAL_PRESENCE_PENALTY,
        )
        self.assertNotIn("stop", completions.kwargs)
        self.assertNotIn("extra_body", completions.kwargs)

    def test_api_payload_can_disable_thinking(self) -> None:
        completions = _FakeCompletions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        asyncio.run(
            call_api(
                client,
                model_name="test-model",
                prompt="prompt",
                max_tokens=4000,
                max_retries=0,
                retry_delay=0,
                disable_thinking=True,
            )
        )
        self.assertEqual(
            completions.kwargs["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )


if __name__ == "__main__":
    unittest.main()
