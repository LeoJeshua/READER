# MMLU-Pro API evaluation

This directory is independent of the earlier Transformers and embedded-vLLM
evaluators. It follows the OpenAI-compatible branch in the official
`TIGER-AI-Lab/MMLU-Pro` `evaluate_from_api.py`.

The bundled `data/` directory contains the validation demonstrations and test
questions used by the evaluator. Local response files are written under
`results/`, which is intentionally excluded from Git because responses can be
large and may record machine-specific endpoint metadata.

## Official evaluation settings

- All validation examples from the matching category (5-shot in the bundled
  dataset)
- One user message through `/v1/chat/completions`
- `temperature=0`
- `max_tokens=4000`
- `top_p=1`
- `frequency_penalty=0`
- `presence_penalty=0`
- No stop sequence
- No input-token limit or prompt truncation
- Unparsed answers are randomly scored with the official seed `12345`

Concurrency, timeout, and retry settings only control API transport.
Environment HTTP proxies are ignored by default so local serving endpoints
work directly. Add `--use-env-proxy` when calling an external API that needs
the configured proxy.

## Run

```bash
bash mmlu_pro_api/run.sh \
  --base-url http://SERVER_IP:8000/v1 \
  --model-name ministral-3-8b-instruct-2512 \
  --output-dir mmlu_pro_api/results/ministral3_8b_instruct
```

Small smoke test:

```bash
bash mmlu_pro_api/run.sh \
  --base-url http://SERVER_IP:8000/v1 \
  --model-name ministral-3-8b-instruct-2512 \
  --output-dir mmlu_pro_api/results/ministral3_8b_instruct_smoke \
  --assigned-subjects math \
  --max-per-subject 2
```

The evaluator resumes successful responses from existing per-category result
files unless `--overwrite` is supplied. Records left by failed API requests
are retried automatically. API usage fields are recorded when the server
returns them.
