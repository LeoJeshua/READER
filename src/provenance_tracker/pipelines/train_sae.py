"""Train sparse autoencoder on best-layer proxy features."""
from __future__ import annotations

import argparse
import json

import numpy as np

from provenance_tracker.analysis.sae import SAEConfig, train_sae
from provenance_tracker.utils.io import ensure_parent_dir, load_feature_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAE on best-layer hidden states")
    parser.add_argument("--proxy-features", required=True)
    parser.add_argument("--best-layer", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-means", required=True,
                        help="path to save (C, d_hidden) feature-class-mean array (.npy)")
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1-coef", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    batch = load_feature_batch(args.proxy_features)
    feats = batch.features
    if feats.ndim != 3:
        raise SystemExit(f"expected (N, L, D); got {feats.shape}")
    x = feats[:, args.best_layer, :].astype(np.float32)
    print(f"[sae] input shape={x.shape} best_layer={args.best_layer}")

    cfg = SAEConfig(
        input_dim=x.shape[1],
        expansion=args.expansion,
        l1_coef=args.l1_coef,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
    )
    _model, result = train_sae(x, batch.labels, cfg)

    payload = {
        "classes": result.classes,
        "top_feature_idx": result.top_feature_idx[:128].tolist(),
        "feature_importance_top": result.feature_importance[result.top_feature_idx[:128]].tolist(),
        "recon_loss": result.recon_loss,
        "sparsity": result.sparsity,
        "d_hidden": int(result.feature_class_mean.shape[1]),
        "best_layer": args.best_layer,
    }
    out = ensure_parent_dir(args.output_json)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    np.save(args.output_means, result.feature_class_mean)
    print(f"[sae] wrote {out} sparsity={result.sparsity:.3f} recon={result.recon_loss:.4f}")


if __name__ == "__main__":
    main()
