"""Log-posterior cross-K evaluation on flat baseline encoder features.

Single-stage aggregation (no intra-M): each baseline (mpnet / bge /
qwen3-emb) emits one ``(N, D)`` vector per trajectory. For each K:

* Trains one ``(scaler, PCA, LR)`` per fold on the per-trajectory features;
  evaluation pools log-probs across K trajectories per fingerprint
  (the ``logposterior_flat`` aggregator).
* Builds K-mean-pool fingerprint features for the geometric metrics
  (Pair-AUC / mAP@5 / mAP@10 / ARI / NMI) — same protocol as ``reports_agg``.

Cache namespace is ``<tag>_logposterior_flat_fold{f}.joblib``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from provenance_tracker.evaluation.logposterior_metrics import (
    _build_fp_features_flat,
    assemble_lp_report,
)
from provenance_tracker.evaluation.multi_traj import (
    run_logposterior_sweep_flat,
)
from provenance_tracker.utils.io import load_feature_batch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-path", required=True,
                   help="Flat (N, D) baseline encoder feature .npz "
                        "(mpnet / bge / qwen3-emb).")
    p.add_argument("--method-tag", required=True,
                   help="Encoder tag, e.g. mpnet / bge / qwen3emb.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ks", type=int, nargs="+",
                   default=[1, 5, 10, 20, 50, 100])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    batch = load_feature_batch(args.features_path)
    x = batch.features
    if x.ndim != 2:
        raise ValueError(f"expected flat (N, D), got {x.shape}")
    labels = batch.labels
    print(f"[agg_lp] tag={args.method_tag} x={x.shape} "
          f"classes={len(set(labels))}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[agg_lp] running logposterior_sweep_flat ...", flush=True)
    sweep, raws = run_logposterior_sweep_flat(
        x, labels, list(args.ks),
        seed=args.seed, n_splits=args.n_splits,
        cache_dir=cache_dir, cache_tag=args.method_tag,
        cache_method="logposterior_flat",
    )
    for K in args.ks:
        if K not in raws:
            print(f"[agg_lp] K={K} skipped (no fingerprints)")
            continue
        fp_x, fp_y = _build_fp_features_flat(x, labels, K, args.seed)
        method_name = f"{args.method_tag}_crossK{K}"
        report = assemble_lp_report(
            method=method_name,
            aggregator="logposterior_flat",
            K=K, M=None,
            raw=raws[K],
            fp_features=fp_x, fp_labels=fp_y,
            n_splits=args.n_splits,
            random_state=args.seed,
            clf="lr",
            n_jobs=args.n_jobs,
            n_samples=fp_x.shape[0],
        )
        out_path = out_dir / f"{args.method_tag}_crossK{K}.json"
        out_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"  [K={K}] acc={report['classification_accuracy']:.3f} "
              f"f1={report['classification_macro_f1']:.3f} "
              f"pairAUC={report['mean_pairwise_auc']:.3f} "
              f"map@5={report['retrieval_map_at_5']:.3f} "
              f"ARI={report['clustering_ari']:.3f} "
              f"NMI={report['clustering_nmi']:.3f} "
              f"NLL={report['nll']:.3f} ECE={report['ece']:.3f} "
              f"-> {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
