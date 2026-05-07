"""CKA-style fingerprint evaluation.

Drops the LR head and replaces it with a parameter-free *covariance-cosine*
classifier (see ``provenance_tracker.analysis.cka_fingerprint`` for the math
and the deviation from standard linear CKA).

Two modes:

* ``--input-kind multi_layer``: the file holds ``features (N, L, D)`` (e.g.
  ``proxy_features/all_targets_*_response.npz`` with single position per
  record). For each layer in ``--layers`` the script does the (M=1, K) sweep.

* ``--input-kind intra_single``: the file holds ``features (N, M_max, D)``
  for a *single* layer (e.g. ``intra_features/8b_L25_M16.npz``). The script
  does the (M, K) sweep at that fixed layer, where M positions are the first
  ``m`` consecutive ones (stride-Δ=1).

For each configuration we build per-class fingerprints C_c = X_c^T X_c on the
training pool, compare each test fingerprint via ``cov_cosine`` and report
classification accuracy / macro-F1 / per-class F1 / mean pairwise AUC.

Multi-GPU: pass ``WORLD_SIZE`` and ``RANK`` env vars (or
``--world-size / --rank``) to shard the (layer, M, K) configuration list
across replicas. Each rank writes its own JSON; merge after the run.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from provenance_tracker.analysis.cka_fingerprint import (
    compute_gram,
    cov_cosine_batch,
    fit_top_k_pcs,
    whiten,
)
from provenance_tracker.utils.io import ensure_parent_dir, load_feature_batch


@dataclass(slots=True)
class CkaConfigResult:
    method: str
    layer: int
    intra_m: int
    cross_k: int
    whiten_top_k: int
    n_samples: int
    n_classes: int
    n_fingerprints: int
    classification_accuracy: float
    classification_macro_f1: float
    mean_pairwise_auc: float
    retrieval_map_at_5: float
    retrieval_map_at_10: float
    per_class_f1: dict[str, float] = field(default_factory=dict)


def _select_intra_positions(features: np.ndarray, m: int) -> np.ndarray:
    """Return the last m positions (stride-Δ=1, ending at M_max-1).

    Anchoring at the end makes ``m=1`` reproduce the existing last-token
    baseline used by ``provenance_tracker.pipelines.evaluate``: the very
    final response token. ``m > 1`` adds the m-1 immediately preceding
    positions (consecutive, no skipping).
    """
    if features.ndim == 2:
        if m != 1:
            raise ValueError(f"got 2D features but m={m}")
        return features[:, None, :]  # (N, 1, D)
    if features.ndim != 3:
        raise ValueError(f"expected 2D or 3D features; got {features.shape}")
    if m < 1 or m > features.shape[1]:
        raise ValueError(f"m={m} out of range [1, {features.shape[1]}]")
    return features[:, features.shape[1] - m :, :]


def _build_fingerprints(
    feats: np.ndarray, labels: list[str], k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Group records per class into fingerprint chunks of size K.

    Returns:
      groups:  (G, K) int64 — record indices per fingerprint
      fp_lab:  (G,) object — class label per fingerprint
    """
    by_cls: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_cls.setdefault(lab, []).append(i)
    groups: list[np.ndarray] = []
    fp_lab: list[str] = []
    for lab, idxs in by_cls.items():
        idxs_arr = np.array(idxs, dtype=np.int64)
        rng.shuffle(idxs_arr)
        n_g = len(idxs_arr) // k
        if n_g == 0:
            continue
        trimmed = idxs_arr[: n_g * k].reshape(n_g, k)
        for row in trimmed:
            groups.append(row)
            fp_lab.append(lab)
    return np.stack(groups, axis=0), np.array(fp_lab, dtype=object)


def _evaluate_layer_mk(
    feats_full: np.ndarray,        # (N, M_used, D)
    labels: list[str],
    *,
    layer: int,
    m: int,
    k: int,
    whiten_top_k: int,
    n_splits: int,
    seed: int,
    device: torch.device,
) -> CkaConfigResult:
    n_records, m_used, d = feats_full.shape
    rng = np.random.default_rng(seed)

    # 1. Group records into fingerprints of size K (per-class shuffle).
    groups, fp_lab = _build_fingerprints(feats_full, labels, k, rng)
    n_fp = groups.shape[0]
    classes = sorted(set(fp_lab.tolist()))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    fp_y = np.array([cls_to_idx[c] for c in fp_lab], dtype=np.int64)
    if n_fp < n_splits or min((fp_y == c).sum() for c in range(len(classes))) < n_splits:
        n_splits = max(2, int(min(n_fp, min((fp_y == c).sum() for c in range(len(classes))))))

    # 2. Pre-load all features to GPU once. Shape (N, M_used, D).
    feats_t = torch.from_numpy(feats_full).to(device=device, dtype=torch.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_true: list[int] = []
    all_pred: list[int] = []
    all_scores: list[np.ndarray] = []          # (n_test, C) per fold

    for tr_idx, te_idx in skf.split(np.zeros(n_fp), fp_y):
        # --- training tokens per class ---
        # Collect every record-index used by training fingerprints, then per
        # class flatten those records' M tokens into a (N_c_tokens, D) matrix.
        train_records_per_class: dict[int, np.ndarray] = {}
        for fp in tr_idx:
            cls_id = fp_y[fp]
            recs = groups[fp]
            train_records_per_class.setdefault(cls_id, []).append(recs)
        train_pool_records: list[int] = []  # for global SVD
        train_tokens_per_class: dict[int, torch.Tensor] = {}
        for cls_id, lst in train_records_per_class.items():
            recs = np.concatenate(lst)
            tok = feats_t[recs].reshape(-1, d)  # (n_records*M_used, D)
            train_tokens_per_class[cls_id] = tok
            train_pool_records.extend(recs.tolist())

        # --- optional whitening from pooled training tokens only ---
        whiten_spec = None
        if whiten_top_k > 0:
            pool_tokens = feats_t[np.array(train_pool_records)].reshape(-1, d)
            whiten_spec = fit_top_k_pcs(pool_tokens, top_k=whiten_top_k)

        # --- per-class training prototype C_c ---
        protos: list[torch.Tensor] = [None] * len(classes)  # type: ignore[list-item]
        for cls_id, tok in train_tokens_per_class.items():
            tok_w = whiten(tok, whiten_spec) if whiten_spec is not None else tok
            protos[cls_id] = compute_gram(tok_w, center=True)
        # Stack into (C, D, D) for batch cosine.
        C_class = torch.stack(protos, dim=0)

        # --- per-test-fingerprint C_q + predict ---
        scores_fold = np.zeros((len(te_idx), len(classes)), dtype=np.float32)
        for j, fp in enumerate(te_idx):
            recs = groups[fp]
            tok = feats_t[recs].reshape(-1, d)
            tok_w = whiten(tok, whiten_spec) if whiten_spec is not None else tok
            C_q = compute_gram(tok_w, center=True)
            sims = cov_cosine_batch(C_q, C_class)
            scores_fold[j] = sims.detach().cpu().numpy()
            all_true.append(int(fp_y[fp]))
            all_pred.append(int(sims.argmax().item()))
        all_scores.append(scores_fold)

    # --- aggregate metrics ---
    all_true_arr = np.array(all_true)
    all_pred_arr = np.array(all_pred)
    acc = float((all_true_arr == all_pred_arr).mean())

    # macro F1
    f1_per_class: dict[str, float] = {}
    for c_idx, c_name in enumerate(classes):
        tp = int(((all_true_arr == c_idx) & (all_pred_arr == c_idx)).sum())
        fp = int(((all_true_arr != c_idx) & (all_pred_arr == c_idx)).sum())
        fn = int(((all_true_arr == c_idx) & (all_pred_arr != c_idx)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1_per_class[c_name] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    macro_f1 = float(np.mean(list(f1_per_class.values()))) if f1_per_class else 0.0

    # mean pairwise AUC: per (a, b), score = sim_a - sim_b, label = 1 if class==a
    scores_mat = np.concatenate(all_scores, axis=0)
    auc_pair: list[float] = []
    for ia, ib in combinations(range(len(classes)), 2):
        mask = (all_true_arr == ia) | (all_true_arr == ib)
        if mask.sum() < 2:
            continue
        y_pair = (all_true_arr[mask] == ia).astype(int)
        if y_pair.sum() == 0 or y_pair.sum() == mask.sum():
            continue
        s = scores_mat[mask, ia] - scores_mat[mask, ib]
        auc_pair.append(float(roc_auc_score(y_pair, s)))
    mean_auc = float(np.mean(auc_pair)) if auc_pair else 0.0

    # retrieval mAP: rank fingerprints by class-score on the *true class* row.
    # Use leave-one-out cosine over fingerprint feature vectors approximated
    # by their score row (a (C,) signature). This is an information-cheap
    # surrogate that lets render_report.py plot side-by-side without
    # implementing fp-vs-fp Gram comparison.
    map_5 = _retrieval_map_from_scores(scores_mat, all_true_arr, k=5)
    map_10 = _retrieval_map_from_scores(scores_mat, all_true_arr, k=10)

    return CkaConfigResult(
        method=f"cka_primal_L{layer}_M{m}_K{k}_W{whiten_top_k}",
        layer=layer,
        intra_m=m,
        cross_k=k,
        whiten_top_k=whiten_top_k,
        n_samples=n_records,
        n_classes=len(classes),
        n_fingerprints=int(n_fp),
        classification_accuracy=acc,
        classification_macro_f1=macro_f1,
        mean_pairwise_auc=mean_auc,
        retrieval_map_at_5=map_5,
        retrieval_map_at_10=map_10,
        per_class_f1=f1_per_class,
    )


def _retrieval_map_from_scores(scores: np.ndarray, y: np.ndarray, *, k: int) -> float:
    """Cheap mAP@k surrogate: each fingerprint's class-score row is its
    'embedding'; rank its neighbours by cosine to it; relevance = same y."""
    if scores.shape[0] == 0:
        return 0.0
    norms = np.linalg.norm(scores, axis=1, keepdims=True) + 1e-12
    z = scores / norms
    sims = z @ z.T
    np.fill_diagonal(sims, -np.inf)
    n = scores.shape[0]
    aps: list[float] = []
    for i in range(n):
        topk = np.argpartition(-sims[i], min(k, n - 1))[:k]
        topk = topk[np.argsort(-sims[i][topk])]
        rel = (y[topk] == y[i]).astype(int)
        if rel.sum() == 0:
            aps.append(0.0)
            continue
        cum = np.cumsum(rel)
        prec_at = cum / (np.arange(len(rel)) + 1)
        aps.append(float((prec_at * rel).sum() / rel.sum()))
    return float(np.mean(aps))


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CKA-fingerprint sweep evaluator")
    parser.add_argument("--features-path", required=True)
    parser.add_argument(
        "--input-kind", choices=["multi_layer", "intra_single"], required=True,
        help="multi_layer: features (N, L, D), sweeps layers + K (M=1). "
             "intra_single: features (N, M_max, D) at one fixed layer, sweeps M+K.",
    )
    parser.add_argument(
        "--fixed-layer", type=int, default=-1,
        help="(intra_single only) layer index to record in the result; defaults "
             "to -1 if unknown.",
    )
    parser.add_argument(
        "--layers", type=str, default="",
        help="(multi_layer only) comma-separated layer indices to evaluate. "
             "Empty -> all layers.",
    )
    parser.add_argument(
        "--ms", type=str, default="1",
        help="comma-separated intra-M values. multi_layer mode requires '1'.",
    )
    parser.add_argument(
        "--ks", type=str, default="1,2,4,8,16,32,48,64,96",
        help="comma-separated cross-K values.",
    )
    parser.add_argument(
        "--whiten-top-ks", type=str, default="0",
        help="comma-separated top-K PC removal sizes (0 = no whitening).",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", required=True,
                        help="Path to write the array of CkaConfigResult dicts.")
    parser.add_argument(
        "--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")),
        help="number of ranks to shard configurations across.",
    )
    parser.add_argument(
        "--rank", type=int, default=int(os.environ.get("RANK", "0")),
        help="this rank index (0..world_size-1).",
    )
    args = parser.parse_args()

    batch = load_feature_batch(args.features_path)
    feats_full = batch.features  # (N, A, D) where A is layer count or M_max
    labels = batch.labels

    if args.input_kind == "multi_layer":
        if feats_full.ndim != 3:
            raise ValueError(f"multi_layer expects 3D features; got {feats_full.shape}")
        n, l_total, d = feats_full.shape
        layers = _parse_int_list(args.layers) if args.layers else list(range(l_total))
        ms = _parse_int_list(args.ms)
        if ms != [1]:
            raise ValueError("multi_layer mode supports only --ms 1")
    else:  # intra_single
        if feats_full.ndim != 3:
            raise ValueError(f"intra_single expects 3D features; got {feats_full.shape}")
        n, m_max, d = feats_full.shape
        layers = [args.fixed_layer]
        ms = _parse_int_list(args.ms)

    ks = _parse_int_list(args.ks)
    whitens = _parse_int_list(args.whiten_top_ks)

    # Build full configuration grid then shard by rank.
    configs: list[tuple[int, int, int, int]] = []  # (layer, m, k, w)
    for layer in layers:
        for m in ms:
            for k in ks:
                for w in whitens:
                    configs.append((layer, m, k, w))
    my_configs = configs[args.rank :: args.world_size]
    print(f"[evaluate_cka] rank={args.rank}/{args.world_size} configs={len(my_configs)}/{len(configs)}")

    device = torch.device(args.device)
    results: list[dict] = []
    for cfg_i, (layer, m, k, w) in enumerate(my_configs):
        t0 = time.time()
        # Slice features down to (N, M_used, D) for this configuration.
        if args.input_kind == "multi_layer":
            slice_full = feats_full[:, layer, :][:, None, :]   # (N, 1, D)
        else:
            slice_full = _select_intra_positions(feats_full, m)
        result = _evaluate_layer_mk(
            slice_full, labels,
            layer=layer, m=m, k=k,
            whiten_top_k=w, n_splits=args.n_splits, seed=args.seed,
            device=device,
        )
        elapsed = time.time() - t0
        rec = asdict(result)
        rec["elapsed_sec"] = round(elapsed, 3)
        results.append(rec)
        print(
            f"  [{cfg_i + 1}/{len(my_configs)}] L={layer} M={m} K={k} W={w} "
            f"acc={result.classification_accuracy:.3f} macroF1={result.classification_macro_f1:.3f} "
            f"AUC={result.mean_pairwise_auc:.3f} N_fp={result.n_fingerprints} t={elapsed:.1f}s"
        )

    out_path = ensure_parent_dir(args.output_json)
    if args.world_size > 1:
        # Add rank suffix so concurrent ranks don't clobber each other.
        out_path = out_path.with_name(out_path.stem + f".rank{args.rank}" + out_path.suffix)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[evaluate_cka] wrote {out_path}")


if __name__ == "__main__":
    main()
