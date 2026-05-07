"""Run multi-trajectory aggregator comparison on a proxy feature NPZ.

Iterates K ∈ {1,2,4,8,16,32} for three aggregators (mean-pool+LR,
log-posterior, Gaussian discriminant) and runs MSPRT once with K_max=32.
Saves a JSON dump of all results.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from provenance_tracker.evaluation.multi_traj import (
    run_gaussian_disc,
    run_logposterior,
    run_meanpool_lr,
    run_msprt,
)
from provenance_tracker.utils.io import load_feature_batch


def _resolve_layer(x3: np.ndarray, layer: int) -> np.ndarray:
    if x3.ndim == 2:
        return x3.astype(np.float32)
    if x3.ndim == 3:
        return x3[:, layer, :].astype(np.float32)
    raise ValueError(f"unexpected features shape {x3.shape}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-path", required=True)
    p.add_argument("--math-features-path", default=None,
                   help="Optional held-out math features (same view); each "
                        "fold's classifier scores math fingerprints, log-probs "
                        "are ensembled across folds.")
    p.add_argument("--layer", type=int, default=23)
    p.add_argument("--method-tag", required=True,
                   help="Human tag for this feature source (e.g. 8b_L23)")
    p.add_argument("--output-json", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--sprt-k-max", type=int, default=32)
    p.add_argument("--sprt-streams", type=int, default=50)
    p.add_argument("--skip-sprt", action="store_true")
    args = p.parse_args()

    batch = load_feature_batch(args.features_path)
    x = _resolve_layer(batch.features, args.layer)
    labels = batch.labels
    print(f"[multi_traj] tag={args.method_tag} x={x.shape} classes={len(set(labels))}")

    math_x = math_labels = None
    if args.math_features_path:
        m_batch = load_feature_batch(args.math_features_path)
        math_x = _resolve_layer(m_batch.features, args.layer)
        math_labels = m_batch.labels
        agent_classes = set(labels)
        keep = [i for i, l in enumerate(math_labels) if l in agent_classes]
        if len(keep) != len(math_labels):
            print(f"[multi_traj] math: dropping {len(math_labels)-len(keep)} "
                  f"rows whose labels are not in agent")
            math_x = math_x[keep]
            math_labels = [math_labels[i] for i in keep]
        print(f"[multi_traj] math={math_x.shape} classes={len(set(math_labels))}")

    results = {"tag": args.method_tag, "seed": args.seed,
               "layer": args.layer,
               "math_features_path": args.math_features_path,
               "aggregators": [], "aggregators_math": [], "sprt": None}

    for K in args.ks:
        for fn, name in [
            (run_meanpool_lr, "meanpool_lr"),
            (run_logposterior, "logposterior"),
            (run_gaussian_disc, "gaussian_disc"),
        ]:
            print(f"  [{name} K={K}] ...", flush=True)
            m_in, m_math = fn(x, labels, K, args.seed,
                              math_x=math_x, math_labels=math_labels)
            results["aggregators"].append(asdict(m_in))
            msg = (f"    acc={m_in.accuracy:.3f} f1={m_in.macro_f1:.3f} "
                   f"nll={m_in.nll:.3f} ece={m_in.ece:.3f} N={m_in.n_fingerprints}")
            if m_math is not None:
                results["aggregators_math"].append(asdict(m_math))
                msg += (f"  | math acc={m_math.accuracy:.3f} "
                        f"f1={m_math.macro_f1:.3f} nll={m_math.nll:.3f} "
                        f"ece={m_math.ece:.3f} N={m_math.n_fingerprints}")
            print(msg)

    if not args.skip_sprt:
        print(f"  [MSPRT K_max={args.sprt_k_max} streams={args.sprt_streams}] ...", flush=True)
        results["sprt"] = run_msprt(
            x, labels, args.seed,
            K_max=args.sprt_k_max,
            n_streams_per_class=args.sprt_streams,
        )
        for a, s in results["sprt"]["per_alpha"].items():
            print(f"    α={a}: E[τ]={s['E_tau']:.2f} decided={s['decided_frac']:.2f} "
                  f"acc={s['overall_acc']:.3f} (when decided {s['acc_when_decided']:.3f})")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"[multi_traj] wrote {out}")


if __name__ == "__main__":
    main()
