"""Run multi-trajectory aggregator comparison on intra-position features.

Mirrors run_multi_traj_eval but for ``intra_features/<tag>_L<L>_M16.npz``
(shape ``(N, M_max, D)``). Sweeps M ∈ {4, 8, 16} × K ∈ args.ks and runs three
aggregators per (M, K): mean-pool + LR, log-posterior, Gaussian discriminant.
Optionally also evaluates each fold's classifier on a held-out math intra
feature set (cross-domain robustness) and caches the per-fold (sc, pca, clf)
to ``cache_dir`` for reuse.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from provenance_tracker.evaluation.multi_traj import (
    run_gaussian_disc_intra_sweep,
    run_logposterior_intra_sweep,
    run_meanpool_lr_intra,
)
from provenance_tracker.utils.io import load_feature_batch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--intra-features-path", required=True,
                   help="Path to intra_features/<tag>_L<L>_M16.npz (N, M_max, D).")
    p.add_argument("--method-tag", required=True,
                   help="Human tag for this feature source (e.g. qwen3_8b_L23).")
    p.add_argument("--output-json", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ks", type=int, nargs="+",
                   default=[1, 5, 10, 20, 50, 100])
    p.add_argument("--ms", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--math-intra-features-path", default=None,
                   help="Optional held-out intra features for cross-domain "
                        "(math) evaluation. Same shape as --intra-features-path.")
    p.add_argument("--cache-dir", default=None,
                   help="Optional dir for joblib-pickled per-fold (sc, pca, "
                        "clf) — speeds up repeat runs.")
    args = p.parse_args()

    batch = load_feature_batch(args.intra_features_path)
    x = batch.features
    if x.ndim != 3:
        raise ValueError(f"expected (N, M_max, D), got {x.shape}")
    labels = batch.labels
    M_max = x.shape[1]
    print(f"[multi_traj_intra] tag={args.method_tag} x={x.shape} "
          f"classes={len(set(labels))} M_max={M_max}")

    math_x = math_labels = None
    if args.math_intra_features_path:
        mb = load_feature_batch(args.math_intra_features_path)
        math_x = mb.features
        math_labels = mb.labels
        if math_x.ndim != 3:
            raise ValueError(f"math expected (N, M_max, D), got {math_x.shape}")
        print(f"[multi_traj_intra] math={math_x.shape} "
              f"classes={len(set(math_labels))}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "tag": args.method_tag,
        "seed": args.seed,
        "M_max": int(M_max),
        "ks": list(args.ks),
        "ms": list(args.ms),
        "math_features_path": args.math_intra_features_path,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "aggregators": [],
        "aggregators_math": [],
    }

    def _emit(metrics_in, metrics_math, name: str, M: int, K: int) -> None:
        d_in = asdict(metrics_in); d_in["M"] = int(M)
        results["aggregators"].append(d_in)
        msg = (f"    [{name} M={M} K={K}] indist acc={metrics_in.accuracy:.3f} "
               f"f1={metrics_in.macro_f1:.3f} nll={metrics_in.nll:.3f} "
               f"ece={metrics_in.ece:.3f} N={metrics_in.n_fingerprints}")
        if metrics_math is not None:
            d_m = asdict(metrics_math); d_m["M"] = int(M)
            results["aggregators_math"].append(d_m)
            msg += (f"  | math acc={metrics_math.accuracy:.3f} "
                    f"f1={metrics_math.macro_f1:.3f} nll={metrics_math.nll:.3f} "
                    f"ece={metrics_math.ece:.3f} N={metrics_math.n_fingerprints}")
        print(msg, flush=True)

    for M in args.ms:
        if M > M_max:
            print(f"[warn] M={M} > M_max={M_max}, skipping")
            continue

        # logposterior + gaussian_disc: train once per (M, fold), sweep K cheaply.
        print(f"[multi_traj_intra] M={M} logposterior_intra_sweep ...", flush=True)
        sweep_lp = run_logposterior_intra_sweep(
            x, labels, list(args.ks), M, args.seed,
            n_splits=args.n_splits,
            cache_dir=cache_dir, cache_tag=args.method_tag,
            math_x_intra=math_x, math_labels=math_labels,
        )
        for K in args.ks:
            if K in sweep_lp:
                m_in, m_math = sweep_lp[K]
                _emit(m_in, m_math, "logposterior_intra", M, K)

        print(f"[multi_traj_intra] M={M} gaussian_disc_intra_sweep ...", flush=True)
        sweep_gd = run_gaussian_disc_intra_sweep(
            x, labels, list(args.ks), M, args.seed,
            n_splits=args.n_splits,
            cache_dir=cache_dir, cache_tag=args.method_tag,
            math_x_intra=math_x, math_labels=math_labels,
        )
        for K in args.ks:
            if K in sweep_gd:
                m_in, m_math = sweep_gd[K]
                _emit(m_in, m_math, "gaussian_disc_intra", M, K)

        # mean-pool: training depends on K (fingerprint pooling), keep per-K loop.
        for K in args.ks:
            print(f"[multi_traj_intra] M={M} K={K} meanpool_lr_intra ...",
                  flush=True)
            m_in, m_math = run_meanpool_lr_intra(
                x, labels, K, M, args.seed,
                n_splits=args.n_splits,
                cache_dir=cache_dir, cache_tag=args.method_tag,
                math_x_intra=math_x, math_labels=math_labels,
            )
            _emit(m_in, m_math, "meanpool_lr_intra", M, K)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"[multi_traj_intra] wrote {out}")


if __name__ == "__main__":
    main()
