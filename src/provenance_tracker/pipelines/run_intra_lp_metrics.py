"""Pure log-posterior intra-M + cross-K evaluation with the full 9-metric panel.

For each (M, K) pair:

* Trains one ``(scaler, PCA, LR)`` per fold on per-token features (intra-M
  positions uniformly sampled from ``M_max``); evaluation pools per-token
  log-probs across K samples × M tokens (the ``logposterior_intra``
  aggregator).
* Builds a separate K-mean-pool fingerprint matrix for the geometric
  metrics (Pair-AUC / mAP@5 / mAP@10 / ARI / NMI) — same protocol as the
  existing ``reports_intra`` evaluator.

Output JSON schema mirrors ``reports_intra/<tag>_intraM<M>_crossK<K>.json``
so downstream plotting code keeps working; adds ``aggregator``,
``classification_accuracy_fold_*``, ``nll``, ``ece``.

Cache key reuses the existing
``<cache_dir>/<tag>_logposterior_intra_M{M}_fold{f}.joblib`` namespace from
``run_multi_traj_intra_eval`` so prior fits are reused for free.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from provenance_tracker.evaluation.logposterior_metrics import (
    _build_fp_features_intra,
    assemble_lp_report,
)
from provenance_tracker.evaluation.multi_traj import (
    run_logposterior_intra_sweep_full,
)
from provenance_tracker.utils.io import load_feature_batch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--intra-features-path", required=True)
    p.add_argument("--method-tag", required=True,
                   help="e.g. qwen3_8b_L23 — used in 'method' field and cache.")
    p.add_argument("--output-dir", required=True,
                   help="Directory for per-(M,K) JSON files.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ks", type=int, nargs="+",
                   default=[1, 5, 10, 20, 50, 100])
    p.add_argument("--ms", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--cache-dir", default=None,
                   help="Joblib cache dir for per-fold (sc, pca, LR). Reuses "
                        "existing logposterior_intra cache from "
                        "run_multi_traj_intra_eval if shared.")
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    batch = load_feature_batch(args.intra_features_path)
    x = batch.features
    if x.ndim != 3:
        raise ValueError(f"expected (N, M_max, D), got {x.shape}")
    labels = batch.labels
    M_max = x.shape[1]
    print(f"[intra_lp] tag={args.method_tag} x={x.shape} "
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
        print(f"[intra_lp] M={M} logposterior_intra_sweep ...", flush=True)
        sweep, raws = run_logposterior_intra_sweep_full(
            x, labels, list(args.ks), M, args.seed,
            n_splits=args.n_splits,
            cache_dir=cache_dir, cache_tag=args.method_tag,
        )
        for K in args.ks:
            if K not in raws:
                print(f"[intra_lp] M={M} K={K} skipped (no fingerprints)")
                continue
            fp_x, fp_y = _build_fp_features_intra(x, labels, K, M, args.seed)
            method_name = f"{args.method_tag}_intraM{M}_crossK{K}"
            report = assemble_lp_report(
                method=method_name,
                aggregator="logposterior_intra_lp",
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
