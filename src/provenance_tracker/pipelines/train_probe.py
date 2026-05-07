"""Fit a linear probe on the best-layer features and save it as an .npz bundle.

The bundle (StandardScaler mean/std + multinomial LR weights/bias) is reused by
activation-patching and attribution-patching pipelines to score intermediate
hidden states on the fly.
"""
from __future__ import annotations

import argparse

import numpy as np

from provenance_tracker.analysis.probe_utils import fit_probe, save_probe
from provenance_tracker.utils.io import ensure_parent_dir, load_feature_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear probe on best layer")
    parser.add_argument("--proxy-features", required=True)
    parser.add_argument("--best-layer", type=int, required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    batch = load_feature_batch(args.proxy_features)
    feats = batch.features
    if feats.ndim != 3:
        raise SystemExit(f"expected (N, L, D); got {feats.shape}")
    x = feats[:, args.best_layer, :].astype(np.float32)
    bundle = fit_probe(x, batch.labels, C=args.C)
    out = ensure_parent_dir(args.output_path)
    save_probe(bundle, str(out))
    print(f"[probe] layer={args.best_layer} train_acc={bundle.train_accuracy:.4f} -> {out}")


if __name__ == "__main__":
    main()
