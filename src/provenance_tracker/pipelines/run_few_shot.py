"""Run prompt-level few-shot evaluation on a feature .npz.

Supports the same view specs as ``evaluate.py``:
  - ``flat`` for already-2D features (mpnet)
  - ``last_layer`` / ``layer=<i>`` / ``best_layer`` for proxy multi-layer
  - ``intra_m=<m>`` for intra-position features (mean-pooled)

Writes a JSON dump with one entry per ``--k-shots`` value.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from provenance_tracker.evaluation.few_shot import run_few_shot
from provenance_tracker.pipelines.evaluate import _resolve_view
from provenance_tracker.utils.io import load_feature_batch


def main() -> None:
    p = argparse.ArgumentParser(description="Prompt-level few-shot evaluation")
    p.add_argument("--features-path", required=True)
    p.add_argument("--view", default="flat")
    p.add_argument("--method-name", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--k-shots", type=int, nargs="+", default=[5, 10, 20, 30])
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--no-inner-cv", action="store_true",
                   help="Disable LR C tuning via inner CV (use C=0.1)")
    p.add_argument("--pca-dim", type=int, default=128,
                   help="PCA bottleneck before LR; set to 0 to disable")
    args = p.parse_args()

    batch = load_feature_batch(args.features_path)
    X = _resolve_view(batch.features, batch.labels, args.view)
    print(f"[few_shot] method={args.method_name} view={args.view} "
          f"X.shape={X.shape} classes={len(set(batch.labels))} "
          f"prompts={len(set(batch.sample_ids))}")

    results = []
    for k in args.k_shots:
        r = run_few_shot(
            X, batch.labels, batch.sample_ids,
            k_shots_per_prompt=k, n_seeds=args.n_seeds,
            inner_cv=not args.no_inner_cv,
            seed_base=args.seed_base,
            method_name=args.method_name,
            pca_dim=args.pca_dim if args.pca_dim > 0 else None,
        )
        results.append(asdict(r))
        print(f"  k={k:3d}  acc={r.accuracy_mean:.3f}±{r.accuracy_std:.3f}  "
              f"f1={r.macro_f1_mean:.3f}±{r.macro_f1_std:.3f}  "
              f"N_tr={r.n_train_samples} N_te={r.n_test_samples}")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "method": args.method_name,
        "view": args.view,
        "features_path": args.features_path,
        "k_shots_results": results,
    }, indent=2))
    print(f"[few_shot] wrote {out}")


if __name__ == "__main__":
    main()
