# Canonical protocol

## Dynamic attribution

The canonical paper result uses the 100-way Agent500 roster and all 500 prompts
per source. Split assignment is grouped by prompt: all sources' responses to a
prompt belong to the same outer fold. Five folds use split seed 42.

For each prompt-response pair, the proxy input is:

```text
Prompt:
{prompt}

Response:
{response}
```

The DCT span begins at the first response token and ends at its final effective
token after truncation. For a trajectory of length `N`, READER computes an
orthonormal DCT-II and divides its coefficients by `sqrt(N)`. This makes mode
zero equal to the arithmetic mean. The source fingerprint concatenates mode
zero (DC) and mode one (first AC), giving dimension `2d` for proxy hidden size
`d`. No learned projection, PCA, GRP, or SVD follows the DCT.

Each outer fold fits its own `StandardScaler` and 100-way PyTorch linear probe
on 400 prompts per source. The full-batch Adam optimizer runs 40 steps at
learning rate `1e-3`; cosine decay uses horizon 100 and minimum ratio `0.01`.
The L2 coefficient is `1/(2 C N_train)` with `C=1`, applied to the weight matrix
only.

At evaluation time, query groups are sampled independently within each held-out
fold for seeds 42, 43, and 44. Budgets are `1, 5, 10, 20, 50, 100`. Evidence is
the sum of prior-corrected response log posteriors. The uniform enrolled prior
makes the correction constant across classes, while the implementation retains
the general rule.

Budget curves pool OOF predictions for each grouping seed and then average the
three metrics. Endpoint tables report the mean and population standard
deviation across the five prompt folds. These two macro-F1 summaries need not
be numerically identical.

## No-retraining stress tests

Controlled response-length panels reuse the fold model that excluded the same
Agent500 prompt. The paper groups all row-level OOF predictions source-wise
after this prompt-matched scoring step. Reports also retain a stricter
within-fold grouping diagnostic.

Math100 has unseen prompts. Every one of the five Agent500 fold probes scores
the complete Math100 panel. The endpoint table reports the mean and standard
deviation of the five fold-model metrics. Curves additionally report an
ensemble formed by averaging fold log posteriors before source-wise evidence
accumulation. No probe, standardizer, or proxy weight is retrained.

## Static relationship auditing

The balanced task derived from Bench-A contains 116 pairs over 67 models and
600 aligned prompts. Positive pairs connect a parent and a derived model.
Negative pairs connect different parent families.

For each candidate pair, READER computes aligned-prompt cosine similarity in
the DC and first-AC blocks separately, then averages each score over all
prompts. A split-local standardizer and linear SVM operate on this two-score
feature. The 20 pair splits use a fixed 4:1 train/test ratio. Model-disjoint and
leave-two-family-out definitions are included under `data/bench_a/splits/`.

Static layer validation selected L12/L5/L16/L18 for the four main readers;
dynamic Agent500 uses L2/L5/L17/L18. `configs/proxies.yaml` records both fields.
The feature construction remains identical, while layer selection and linear
readout are task-side fitted choices.

## Reproducibility artifacts

`results/manifest.json` records the SHA-256 and byte size of every released
paper artifact. `results/paper_map.json` maps paper experiment families to
their report directories. `tools/validate_release.py` verifies both manifests
and a set of headline numerical invariants.
