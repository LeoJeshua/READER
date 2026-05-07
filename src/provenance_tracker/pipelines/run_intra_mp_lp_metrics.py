"""Mean-pool intra-M + log-posterior cross-K evaluation, 9-metric panel.

For each (M, K) pair:

* **Stage 1 (intra-M)**: pre-mean-pools the M intra-position tokens into a
  single ``(N, D)`` per-trajectory feature using uniform position sampling
  (matches ``_intra_positions``). The classifier head sees mean-pooled
  features as inputs.
* **Stage 2 (cross-K)**: trains one ``(scaler, PCA, LR)`` per fold on the
  mean-pooled per-trajectory features; evaluation pools log-probs across K
  trajectories per fingerprint.
* Builds a separate K-mean-pool fingerprint matrix for the geometric
  metrics (Pair-AUC / mAP@5 / mAP@10 / ARI / NMI), same protocol as
  ``reports_intra``.

This is the ``meanpool_intra_logpost`` aggregator: classifier sees
mean-pooled per-trajectory features (matches what the user wants for
"intra-M = mean-pool"), but cross-K samples are combined via log-posterior
accumulation (Bayesian evidence) rather than another mean-pool layer.

Cache namespace is ``<tag>_meanpool_intra_logpost_M{M}_fold{f}.joblib``
(distinct from the per-token ``logposterior_intra`` cache used by
``run_intra_lp_metrics``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from provenance_tracker.evaluation.logposterior_metrics import (
    _build_fp_features_intra,
    assemble_lp_report,
)
from provenance_tracker.evaluation.multi_traj import (
    _intra_positions,
    run_logposterior_sweep_flat,
)
from provenance_tracker.utils.io import load_feature_batch


def _mean_pool_intra(x_intra: np.ndarray, M: int) -> np.ndarray:
    """Pre-pool (N, M_max, D) to (N, D) by averaging M uniformly-sampled
    intra positions (matches ``evaluate.py`` ``intra_m=`` view)."""
    pos = _intra_positions(M, x_intra.shape[1])
    return x_intra[:, pos, :].mean(axis=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--intra-features-path", required=True)
    p.add_argument("--method-tag", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ks", type=int, nargs="+",
                   default=[1, 5, 10, 20, 50, 100])
    p.add_argument("--ms", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    batch = load_feature_batch(args.intra_features_path)
    x = batch.features
    if x.ndim != 3:
        raise ValueError(f"expected (N, M_max, D), got {x.shape}")
    labels = batch.labels
    M_max = x.shape[1]
    print(f"[intra_mp_lp] tag={args.method_tag} x={x.shape} "
          f"classes={len(set(labels))} M_max={M_max}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for M in args.ms:
        if M > M_max:
            print(f"[warn] M={M} > M_max={M_max}, skipping")
            continue
        x_flat = _mean_pool_intra(x, M)
        print(f"[intra_mp_lp] M={M} pre-pooled -> {x_flat.shape}; "
              f"running logposterior_sweep_flat ...", flush=True)
        cache_method = f"meanpool_intra_logpost_M{M}"
        sweep, raws = run_logposterior_sweep_flat(
            x_flat, labels, list(args.ks),
            seed=args.seed, n_splits=args.n_splits,
            cache_dir=cache_dir, cache_tag=args.method_tag,
            cache_method=cache_method,
        )
        for K in args.ks:
            if K not in raws:
                print(f"[intra_mp_lp] M={M} K={K} skipped (no fingerprints)")
                continue
            fp_x, fp_y = _build_fp_features_intra(x, labels, K, M, args.seed)
            method_name = f"{args.method_tag}_intraM{M}_crossK{K}"
            report = assemble_lp_report(
                method=method_name,
                aggregator="meanpool_intra_logpost",
                K=K, M=M,
                raw=raws[K],
                fp_features=fp_x, fp_labels=fp_y,
                n_splits=args.n_splits,
                random_state=args.seed,
                clf="lr",
                n_jobs=args.n_jobs,
                n_samples=fp_x.shape[0],
            )
            out_path = out_dir / f"{args.method_tag}_intraM{M}_crossK{K}.json"
            out_path.write_text(json.dumps(report, indent=2, default=str))
            print(f"  [M={M} K={K}] acc={report['classification_accuracy']:.3f} "
                  f"f1={report['classification_macro_f1']:.3f} "
                  f"pairAUC={report['mean_pairwise_auc']:.3f} "
                  f"map@5={report['retrieval_map_at_5']:.3f} "
                  f"ARI={report['clustering_ari']:.3f} "
                  f"NMI={report['clustering_nmi']:.3f} "
                  f"NLL={report['nll']:.3f} ECE={report['ece']:.3f} "
                  f"-> {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
