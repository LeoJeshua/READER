"""Quantitative validation of the superposition / orthogonality hypothesis.

Three proofs over a (C, P, D) tensor built from per-(model, prompt) proxy
hidden states at one layer (or aggregated over M intra positions):

  A. Variance ratio (Fisher ratio) -- raw vs marginalized.
     Sweeps M (intra-position averaging) and K (cross-prompt averaging).
  B. Semantic-center shift test -- Dist_S_shift / Dist_A.
  C. Principal angles between authorship subspace U_A and semantic
     subspace U_S.

Two input formats supported:
  * ``response`` mode: multi-layer FeatureBatch ``(N, L+1, D)``
    (from ``extract_proxy``). Pick a layer with ``--layer``.
    Each (c, p) is a single last-token vector; only K-grid is meaningful.
  * ``intra``   mode: single-layer FeatureBatch ``(N, M_max, D)``
    (from ``extract_intra``). The M dimension is averaged per (c, p);
    sweep M to demonstrate Intra-M LLN.

``--mode auto`` infers from feature_kind.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from provenance_tracker.utils.io import ensure_parent_dir, load_feature_batch


# ---------- per-row feature reduction ----------------------------------------

def _reduce_response(features: np.ndarray, layer: int) -> np.ndarray:
    """response mode: (N, L+1, D) -> (N, D) at chosen layer."""
    L = features.shape[1]
    if not (-L <= layer < L):
        raise SystemExit(f"layer {layer} out of range for L={L}")
    return features[:, layer, :]


def _reduce_intra(features: np.ndarray, m: int) -> np.ndarray:
    """intra mode: (N, M_max, D) -> (N, D) by averaging the first m positions."""
    M_max = features.shape[1]
    m = max(1, min(m, M_max))
    return features[:, :m, :].mean(axis=1)


def _build_cpd_from_2d(
    H: np.ndarray, labels: list[str], sample_ids: list[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Reshape (N, D) into (C, P, D) keeping only prompts common to all models."""
    by_model: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for i, (lab, sid) in enumerate(zip(labels, sample_ids)):
        by_model[lab][sid].append(H[i])

    models = sorted(by_model.keys())
    prompt_sets = [set(by_model[m].keys()) for m in models]
    common = sorted(set.intersection(*prompt_sets)) if prompt_sets else []
    if not common:
        raise SystemExit("no common prompt ids across all models")

    C, P, D = len(models), len(common), H.shape[1]
    H3 = np.empty((C, P, D), dtype=np.float64)
    for ci, m in enumerate(models):
        m_dict = by_model[m]
        for pi, sid in enumerate(common):
            stack = m_dict[sid]
            H3[ci, pi] = stack[0] if len(stack) == 1 else np.mean(stack, axis=0)
    return H3, models, common


# ---------- proofs -----------------------------------------------------------

def _variance_ratio(H3: np.ndarray) -> dict[str, float]:
    mu_c = H3.mean(axis=1)                          # (C, D)
    mu_g = mu_c.mean(axis=0, keepdims=True)         # (1, D)
    var_within = float(((H3 - mu_c[:, None, :]) ** 2).sum(-1).mean())
    var_between = float(((mu_c - mu_g) ** 2).sum(-1).mean())
    ratio = var_between / var_within if var_within > 0 else float("inf")
    return {
        "var_between": var_between,
        "var_within": var_within,
        "fisher_ratio": ratio,
    }


def proof_a_cross_K(H3: np.ndarray, K_grid: list[int]) -> list[dict[str, float]]:
    """Sweep K = how many prompts to average together inside each model."""
    C, P, D = H3.shape
    rows: list[dict[str, float]] = []
    for K in K_grid:
        if K < 1 or K > P:
            continue
        if K == 1:
            rows.append({"K": 1, **_variance_ratio(H3)})
            continue
        rng = np.random.default_rng(0)
        perm = rng.permutation(P)
        n_groups = P // K
        if n_groups < 2:
            continue
        idx = perm[: n_groups * K].reshape(n_groups, K)
        averaged = H3[:, idx, :].mean(axis=2)        # (C, n_groups, D)
        rows.append({"K": K, **_variance_ratio(averaged)})
    return rows


def proof_a_intra_M(
    features: np.ndarray,
    labels: list[str],
    sample_ids: list[str],
    M_grid: list[int],
) -> list[dict[str, float]]:
    """Sweep M = how many intra-response positions to average. Only intra mode."""
    rows: list[dict[str, float]] = []
    for M in M_grid:
        H = _reduce_intra(features, M)
        H3, _, _ = _build_cpd_from_2d(H, labels, sample_ids)
        rows.append({"M": int(M), **_variance_ratio(H3)})
    return rows


def proof_b_semantic_shift(H3: np.ndarray) -> dict[str, float]:
    C, P, D = H3.shape
    mu_c = H3.mean(axis=1)
    mask = ~np.eye(C, dtype=bool)

    diff_mu = mu_c[:, None, :] - mu_c[None, :, :]
    dist_A = float(np.linalg.norm(diff_mu, axis=-1)[mask].mean())

    e = H3 - mu_c[:, None, :]
    total = 0.0
    count = 0
    for p in range(P):
        ep = e[:, p, :]
        dp = np.linalg.norm(ep[:, None, :] - ep[None, :, :], axis=-1)
        total += float(dp[mask].sum())
        count += int(mask.sum())
    dist_S_shift = total / max(count, 1)

    return {
        "dist_A": dist_A,
        "dist_S_shift": dist_S_shift,
        "shift_ratio": dist_S_shift / dist_A if dist_A > 0 else float("inf"),
    }


def _principal_angles(U_A: np.ndarray, U_S: np.ndarray, k1: int, k2: int) -> dict:
    k1e = min(k1, U_A.shape[1])
    k2e = min(k2, U_S.shape[1])
    sv = np.linalg.svd(U_A[:, :k1e].T @ U_S[:, :k2e], compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(sv)).tolist()
    return {
        "k1": k1e,
        "k2": k2e,
        "cos_theta": sv.tolist(),
        "angles_deg": angles_deg,
        "min_angle_deg": float(min(angles_deg)) if angles_deg else None,
        "mean_angle_deg": float(np.mean(angles_deg)) if angles_deg else None,
    }


def proof_c_principal_angles(
    H3: np.ndarray, k_pairs: list[tuple[int, int]]
) -> dict[str, object]:
    """Compute principal angles for one or more (k1, k2) configurations.

    The expensive step (the two SVDs over X_A and X_S) is shared across pairs.
    """
    mu_c = H3.mean(axis=1)
    mu_g = mu_c.mean(axis=0, keepdims=True)
    X_A = (mu_c - mu_g).T                             # (D, C)
    U_A_full, _, _ = np.linalg.svd(X_A, full_matrices=False)

    bar_h_p = H3.mean(axis=0)
    bar_h_g = bar_h_p.mean(axis=0, keepdims=True)
    X_S = (bar_h_p - bar_h_g).T                       # (D, P)
    U_S_full, _, _ = np.linalg.svd(X_S, full_matrices=False)

    by_pair = [
        {"requested_k1": int(k1), "requested_k2": int(k2),
         **_principal_angles(U_A_full, U_S_full, k1, k2)}
        for (k1, k2) in k_pairs
    ]
    return {"by_pair": by_pair}


# ---------- markdown ---------------------------------------------------------

def _format_markdown(payload: dict) -> str:
    head = [
        "# Geometric Proofs",
        "",
        f"- features : `{payload['feature_path']}`",
        f"- mode     : `{payload['mode']}`  layer=`{payload.get('layer', 'n/a')}`  "
        f"M_max=`{payload.get('M_max', 'n/a')}`",
        f"- shape    : C={payload['C']} P={payload['P']} D={payload['D']}",
        "",
        "## Proof A — Variance Ratio (Fisher)",
    ]

    a = payload["proof_a"]
    if "intra_M" in a:
        head.append("\n### sweep M (intra-position averaging)\n")
        head.append("| M | Var_between | Var_within | R |")
        head.append("| --: | --: | --: | --: |")
        for r in a["intra_M"]:
            head.append(
                f"| {r['M']} | {r['var_between']:.3e} | "
                f"{r['var_within']:.3e} | {r['fisher_ratio']:.4f} |"
            )

    head.append("\n### sweep K (cross-prompt averaging, at chosen M / layer)\n")
    head.append("| K | Var_between | Var_within | R |")
    head.append("| --: | --: | --: | --: |")
    for r in a["cross_K"]:
        head.append(
            f"| {r['K']} | {r['var_between']:.3e} | "
            f"{r['var_within']:.3e} | {r['fisher_ratio']:.4f} |"
        )

    b = payload["proof_b"]
    head += [
        "",
        "## Proof B — Semantic Center Shift",
        f"- Dist_A          = {b['dist_A']:.4f}",
        f"- Dist_S_shift    = {b['dist_S_shift']:.4f}",
        f"- shift_ratio     = **{b['shift_ratio']:.4f}**  (want « 1)",
    ]

    c = payload["proof_c"]
    head += ["", "## Proof C — Principal Angles"]
    if "by_pair" in c:
        head.append("")
        head.append("| k1 | k2 | min(deg) | mean(deg) | angles_deg |")
        head.append("| --: | --: | --: | --: | --- |")
        for r in c["by_pair"]:
            head.append(
                f"| {r['k1']} | {r['k2']} | {r['min_angle_deg']:.2f} | "
                f"{r['mean_angle_deg']:.2f} | "
                f"{[round(a, 2) for a in r['angles_deg']]} |"
            )
    else:
        head += [
            f"- k1={c.get('k1')}, k2={c.get('k2')}",
            f"- min angle  = **{c['min_angle_deg']:.2f}°**, "
            f"mean angle = {c['mean_angle_deg']:.2f}°",
            f"- angles_deg = {[round(a, 2) for a in c['angles_deg']]}",
        ]
    return "\n".join(head) + "\n"


# ---------- main -------------------------------------------------------------

def _infer_mode(batch) -> str:
    kind = (batch.feature_kind or "").lower()
    if "intra" in kind:
        return "intra"
    if "layerwise" in kind or "last_token" in kind:
        return "response"
    # heuristic on shape: response files have many "layers" (>~10)
    return "intra" if batch.features.shape[1] <= 32 else "response"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three geometric proofs (variance ratio, semantic shift, principal angles)"
    )
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--mode", default="auto",
                        choices=["auto", "response", "intra"])
    parser.add_argument("--layer", default="last",
                        help="response mode only: int or 'last'")
    parser.add_argument("--m-grid", default="1,2,4,8,16",
                        help="intra mode only: M values to sweep")
    parser.add_argument("--k-grid", default="1,2,4,8,16,32,64",
                        help="cross-prompt K to sweep (both modes)")
    parser.add_argument("--k1", type=int, default=10, help="(legacy) dim of U_A in Proof C")
    parser.add_argument("--k2", type=int, default=10, help="(legacy) dim of U_S in Proof C")
    parser.add_argument("--k-pairs", default=None,
                        help="Proof C k1:k2 pairs, e.g. '5:5,10:10,15:15'. "
                             "If omitted, falls back to single (--k1, --k2).")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    batch = load_feature_batch(args.features_path)
    mode = args.mode
    if mode == "auto":
        mode = _infer_mode(batch)
    print(f"[prove] mode={mode} kind={batch.feature_kind} feat.shape={batch.features.shape}")

    K_grid = [int(x) for x in args.k_grid.split(",") if x.strip()]
    M_grid = [int(x) for x in args.m_grid.split(",") if x.strip()]
    layer_used: int | None = None

    if mode == "response":
        if args.layer == "last":
            layer_used = batch.features.shape[1] - 1
        else:
            layer_used = int(args.layer)
            if layer_used < 0:
                layer_used += batch.features.shape[1]
        H = _reduce_response(batch.features, layer_used)
        H3, models, _ = _build_cpd_from_2d(H, batch.labels, batch.sample_ids)
        intra_table = None
        M_max = 1
    else:  # intra
        M_max = int(batch.features.shape[1])
        intra_table = proof_a_intra_M(batch.features, batch.labels, batch.sample_ids, M_grid)
        H = _reduce_intra(batch.features, M_max)
        H3, models, _ = _build_cpd_from_2d(H, batch.labels, batch.sample_ids)

    C, P, D = H3.shape
    print(f"[prove] H3=({C}, {P}, {D}), models={C} prompts={P} dim={D}")

    proof_a = {"cross_K": proof_a_cross_K(H3, K_grid)}
    if intra_table is not None:
        proof_a["intra_M"] = intra_table

    if args.k_pairs:
        k_pairs = [
            (int(p.split(":")[0]), int(p.split(":")[1]))
            for p in args.k_pairs.split(",") if p.strip()
        ]
    else:
        k_pairs = [(args.k1, args.k2)]

    payload: dict = {
        "feature_path": str(args.features_path),
        "mode": mode,
        "layer": layer_used,
        "M_max": M_max,
        "C": C,
        "P": P,
        "D": D,
        "models": models,
        "proof_a": proof_a,
        "proof_b": proof_b_semantic_shift(H3),
        "proof_c": proof_c_principal_angles(H3, k_pairs),
    }

    out_json = ensure_parent_dir(args.output_json)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[prove] wrote {out_json}")

    if args.output_md:
        out_md = ensure_parent_dir(args.output_md)
        out_md.write_text(_format_markdown(payload), encoding="utf-8")
        print(f"[prove] wrote {out_md}")


if __name__ == "__main__":
    main()
