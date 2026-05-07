"""Calibrated multi-hypothesis SPRT comparison.

For one proxy feature NPZ this runs:
- per-sample calibration (acc / NLL / ECE) for three log-prob sources:
  vanilla LR, temperature-scaled LR, Gaussian-discriminant LDA;
- MSPRT on each source with the same K_max and α grid.

Goal: validate the report's claim that SPRT's α-bound violation is a
per-sample calibration problem upstream of the aggregator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from provenance_tracker.evaluation.multi_traj import run_calibrated_msprt_panel
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
    p.add_argument("--layer", type=int, default=23)
    p.add_argument("--method-tag", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sprt-k-max", type=int, default=32)
    p.add_argument("--sprt-streams", type=int, default=50)
    args = p.parse_args()

    batch = load_feature_batch(args.features_path)
    x = _resolve_layer(batch.features, args.layer)
    labels = batch.labels
    print(f"[multi_traj_calib] tag={args.method_tag} x={x.shape} "
          f"classes={len(set(labels))}")

    panel = run_calibrated_msprt_panel(
        x, labels, args.seed,
        K_max=args.sprt_k_max,
        n_streams_per_class=args.sprt_streams,
    )
    panel["tag"] = args.method_tag
    panel["layer"] = args.layer
    panel["seed"] = args.seed

    for src, blob in panel["sources"].items():
        cal = blob["calibration"]
        print(f"  [{src}] per-sample acc={cal['accuracy']:.3f} "
              f"nll={cal['nll']:.3f} ece={cal['ece']:.3f}")
        if blob.get("T_per_fold"):
            Ts = blob["T_per_fold"]
            print(f"    temperatures per fold: "
                  f"{[round(float(t), 3) for t in Ts]}")
        for a, s in blob["sprt"]["per_alpha"].items():
            print(f"    α={a}: E[τ]={s['E_tau']:.2f} "
                  f"decided={s['decided_frac']:.2f} "
                  f"acc={s['overall_acc']:.3f} "
                  f"(when decided {s['acc_when_decided']:.3f})")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(panel, indent=2, default=str))
    print(f"[multi_traj_calib] wrote {out}")


if __name__ == "__main__":
    main()
