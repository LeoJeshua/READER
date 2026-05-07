"""Run mean-LR baseline + 3 attention poolers on (N, M, D) feature files.

Usage::

    python -m provenance_tracker.pipelines.run_attention_pool \\
        --features-path A.npz --features-path B.npz \\
        --output-json out.json [--device cuda] [--epochs 60]

Each input ``.npz`` must contain ``features (N,M,D)`` and ``labels``; if the
file is the all-positions extractor output, ``valid_counts (N,)`` is read and
used as a length mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from provenance_tracker.evaluation.attention_pool import (
    MultiHeadAttn,
    SingleQueryAttn,
    TransformerEncoderCls,
    evaluate_pooled_embedding,
    kfold_evaluate,
    mean_pool_lr_baseline,
)


def _load(path: Path) -> tuple[np.ndarray, list[str], np.ndarray | None, str]:
    with np.load(path, allow_pickle=True) as d:
        features = np.asarray(d["features"], dtype=np.float32)
        labels = [str(x) for x in d["labels"].tolist()]
        valid_counts = (
            np.asarray(d["valid_counts"], dtype=np.int32)
            if "valid_counts" in d.files else None
        )
        feature_kind = str(d["feature_kind"])
    return features, labels, valid_counts, feature_kind


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-path", action="append", required=True, type=Path)
    ap.add_argument("--proxy-name", action="append", required=True,
                    help="Display name aligned with each --features-path")
    ap.add_argument("--output-json", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--ff-dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if len(args.features_path) != len(args.proxy_name):
        raise SystemExit("--features-path and --proxy-name must be paired 1:1")

    out: dict = {"runs": []}
    for path, name in zip(args.features_path, args.proxy_name):
        features, labels, valid_counts, kind = _load(path)
        N, M, D = features.shape
        n_classes = len(set(labels))
        print(f"\n=== {name} :: {path.name} :: features={features.shape} kind={kind} ===")
        print(f"    classes={n_classes} valid_counts={'yes' if valid_counts is not None else 'no'}")

        run = {
            "proxy": name,
            "feature_path": str(path),
            "feature_kind": kind,
            "shape": [int(N), int(M), int(D)],
            "n_classes": n_classes,
            "methods": {},
        }

        # baseline
        print("[mean-pool LR] training")
        base = mean_pool_lr_baseline(features, labels, valid_counts=valid_counts, seed=args.seed)
        emb_metrics = evaluate_pooled_embedding(base.pooled_emb, labels)
        run["methods"]["mean_pool_lr"] = {
            "accuracy_mean": float(np.mean(base.accuracies)),
            "accuracy_std": float(np.std(base.accuracies)),
            "macro_f1_mean": float(np.mean(base.macro_f1s)),
            **emb_metrics,
        }
        print(f"    acc={np.mean(base.accuracies):.4f}±{np.std(base.accuracies):.4f}  "
              f"pair-AUC={emb_metrics['mean_pairwise_auc']:.4f}")

        common_kwargs = dict(
            valid_counts=valid_counts,
            seed=args.seed,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
            batch_size=args.batch_size,
        )

        factories = {
            "single_query_attn": lambda: SingleQueryAttn(D, n_classes, dropout=args.dropout),
            "multi_head_attn": lambda: MultiHeadAttn(
                D, n_classes, n_heads=args.n_heads, dropout=args.dropout
            ),
            "transformer_cls": lambda: TransformerEncoderCls(
                D, n_classes, n_heads=args.n_heads, ff_dim=args.ff_dim,
                dropout=args.dropout, n_layers=1,
            ),
        }
        for name_method, fac in factories.items():
            print(f"[{name_method}] training")
            r = kfold_evaluate(fac, features, labels, **common_kwargs)
            emb_metrics = evaluate_pooled_embedding(r.pooled_emb, labels)
            run["methods"][name_method] = {
                "accuracy_mean": float(np.mean(r.accuracies)),
                "accuracy_std": float(np.std(r.accuracies)),
                "macro_f1_mean": float(np.mean(r.macro_f1s)),
                **emb_metrics,
            }
            print(f"    acc={np.mean(r.accuracies):.4f}±{np.std(r.accuracies):.4f}  "
                  f"pair-AUC={emb_metrics['mean_pairwise_auc']:.4f}")
            torch.cuda.empty_cache()

        out["runs"].append(run)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))
    print(f"\n[done] -> {args.output_json}")


if __name__ == "__main__":
    main()
