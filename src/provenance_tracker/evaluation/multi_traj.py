"""Multi-trajectory aggregators: mean-pool baseline, log-posterior accumulation,
Gaussian discriminant with K-scaled Mahalanobis, and MSPRT stopping-time.

All operate on a per-sample feature matrix ``(N, D)`` with labels; fingerprints
are built by per-class shuffling + chunking into groups of size K. Reproducible
via the ``seed`` argument.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PCA_DIM = 128  # per-fold PCA bottleneck; fit on train features only


@dataclass(slots=True)
class AggMetrics:
    method: str
    K: int
    n_fingerprints: int
    accuracy: float
    macro_f1: float
    nll: float
    ece: float
    alpha_calib: float | None = None
    nll_calib: float | None = None
    ece_calib: float | None = None


@dataclass(slots=True)
class LogPostRawSweep:
    """Raw per-fold output of a log-posterior K-sweep.

    Lets downstream callers compute additional metrics (Pair-AUC, mAP, ARI/NMI)
    on the same fingerprints used for AggMetrics scoring without retraining
    the per-fold LR. ``y_per_fold[i]`` is the int class index per fingerprint
    in fold *i*; ``logp_per_fold[i]`` is the matching ``(n_fp_i, C)`` log-prob
    matrix (un-renormalised; softmax-normalise to recover probabilities).
    """
    K: int
    classes: list[str]
    y_per_fold: list[np.ndarray]
    logp_per_fold: list[np.ndarray]
    alpha: float | None = None


def _group_per_class(labels: list[str], K: int, rng: np.random.Generator
                     ) -> list[tuple[str, np.ndarray]]:
    by_cls: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_cls.setdefault(lab, []).append(i)
    groups: list[tuple[str, np.ndarray]] = []
    for cls, idxs in by_cls.items():
        idx_arr = np.asarray(idxs)
        rng.shuffle(idx_arr)
        n_groups = len(idx_arr) // K
        if n_groups == 0:
            continue
        chunked = idx_arr[: n_groups * K].reshape(n_groups, K)
        for row in chunked:
            groups.append((cls, row))
    return groups


def expected_calibration_error(y_true_i: np.ndarray, probs: np.ndarray,
                               n_bins: int = 10) -> float:
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true_i).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(confs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs < hi) if i < n_bins - 1 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / N) * abs(correct[mask].mean() - confs[mask].mean())
    return float(ece)


def _score(y_true_i: np.ndarray, probs: np.ndarray, classes: list[str],
           method: str, K: int) -> AggMetrics:
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_true_i, preds)
    f1 = f1_score(y_true_i, preds, average="macro")
    nll = log_loss(y_true_i, probs, labels=list(range(len(classes))))
    ece = expected_calibration_error(y_true_i, probs)
    return AggMetrics(method=method, K=K, n_fingerprints=len(y_true_i),
                      accuracy=float(acc), macro_f1=float(f1), nll=float(nll), ece=ece)


def _fit_alpha(log_scores: np.ndarray, y_true_i: np.ndarray,
               bounds: tuple[float, float] = (0.05, 100.0)) -> float:
    """Fit positive evidence scale alpha for softmax(alpha * log_scores)."""
    from scipy.optimize import minimize_scalar

    def nll(log_alpha: float) -> float:
        alpha = float(np.exp(log_alpha))
        p = _normalise_logps(alpha * log_scores)
        return float(log_loss(y_true_i, p, labels=list(range(p.shape[1]))))

    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-3})
    return float(np.exp(res.x))


def _score_log_scores(y_true_i: np.ndarray, log_scores: np.ndarray,
                      classes: list[str], method: str, K: int,
                      alpha_calib: float | None = None) -> AggMetrics:
    """Score uncalibrated log-evidence, optionally with calibrated alpha."""
    probs = _normalise_logps(log_scores)
    metrics = _score(y_true_i, probs, classes, method, K)
    if alpha_calib is not None:
        probs_cal = _normalise_logps(alpha_calib * log_scores)
        metrics.alpha_calib = float(alpha_calib)
        metrics.nll_calib = float(log_loss(
            y_true_i, probs_cal, labels=list(range(len(classes)))))
        metrics.ece_calib = expected_calibration_error(y_true_i, probs_cal)
    return metrics


def _cv_folds(y_fp_i: np.ndarray, n_splits: int, seed: int
              ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    smallest = int(np.bincount(y_fp_i).min())
    n_splits = max(2, min(n_splits, smallest))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from skf.split(np.zeros(len(y_fp_i)), y_fp_i)


def _fit_pca_scaler(x_tr: np.ndarray) -> tuple[StandardScaler, PCA]:
    sc = StandardScaler().fit(x_tr)
    x_std = sc.transform(x_tr)
    n_comp = min(PCA_DIM, max(1, min(x_std.shape) - 1))
    pca = PCA(n_components=n_comp, random_state=0).fit(x_std)
    return sc, pca


def run_meanpool_lr(x: np.ndarray, labels: list[str], K: int, seed: int,
                    n_splits: int = 5,
                    *, math_x: np.ndarray | None = None,
                    math_labels: list[str] | None = None,
                    ) -> tuple[AggMetrics, AggMetrics | None]:
    """Baseline A: mean-pool K features → StandardScaler + PCA + LR.

    With ``math_x`` / ``math_labels``, also evaluates each fold's classifier on
    a fixed math fingerprint partition and ensembles log-probs across folds.
    Returns ``(in_domain, math_or_None)``.
    """
    rng = np.random.default_rng(seed)
    groups = _group_per_class(labels, K, rng)
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    C = len(classes)
    X_fp = np.stack([x[idxs].mean(axis=0) for _, idxs in groups]).astype(np.float64)
    y_fp_i = np.array([c2i[cls] for cls, _ in groups])

    math_pack = None
    if math_x is not None and math_labels is not None:
        rng_m = np.random.default_rng(seed + 1234)
        m_groups = _group_per_class(math_labels, K, rng_m)
        m_groups = [g for g in m_groups if g[0] in c2i]
        if m_groups:
            math_X_fp = np.stack([math_x[idxs].mean(axis=0)
                                  for _, idxs in m_groups]).astype(np.float64)
            math_y_fp_i = np.array([c2i[cls] for cls, _ in m_groups])
            math_logp_acc = np.zeros((len(m_groups), C))
            math_pack = (math_X_fp, math_y_fp_i, math_logp_acc)

    probs = np.zeros((len(groups), C))
    n_folds_done = 0
    for tr, te in _cv_folds(y_fp_i, n_splits, seed):
        sc, pca = _fit_pca_scaler(X_fp[tr])
        xtr = pca.transform(sc.transform(X_fp[tr]))
        xte = pca.transform(sc.transform(X_fp[te]))
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(xtr, y_fp_i[tr])
        probs[te] = _proba_padded(clf, xte, C)
        if math_pack is not None:
            math_X_fp, math_y_fp_i, math_logp_acc = math_pack
            mp = _proba_padded(
                clf, pca.transform(sc.transform(math_X_fp)), C
            )
            math_logp_acc += np.log(mp + 1e-12)
            n_folds_done += 1
    indist = _score(y_fp_i, probs, classes, "meanpool_lr", K)
    math_metrics: AggMetrics | None = None
    if math_pack is not None and n_folds_done > 0:
        _, math_y_fp_i, math_logp_acc = math_pack
        math_logp_acc /= n_folds_done
        math_p = _normalise_logps(math_logp_acc)
        math_metrics = _score(math_y_fp_i, math_p, classes,
                              "meanpool_lr_math", K)
    return indist, math_metrics


def _fit_single_sample_lr_by_fp(x: np.ndarray, labels: list[str],
                                groups: list[tuple[str, np.ndarray]],
                                seed: int, n_splits: int
                                ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run per-fold fit on unrolled train-fingerprint samples. Returns
    (logprobs_per_fp [n_groups, C], y_fp_i [n_groups], classes)."""
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    y_fp_i = np.array([c2i[cls] for cls, _ in groups])
    N_fp = len(groups)
    C = len(classes)

    # per-sample → which fingerprint it belongs to
    fp_of = -np.ones(len(labels), dtype=int)
    for fp_idx, (_, idxs) in enumerate(groups):
        fp_of[idxs] = fp_idx

    logps = np.zeros((N_fp, C))
    for tr_fp, te_fp in _cv_folds(y_fp_i, n_splits, seed):
        tr_sample_mask = np.isin(fp_of, tr_fp)
        tr_idx = np.where(tr_sample_mask)[0]
        x_tr = x[tr_idx]
        y_tr = np.array([c2i[labels[i]] for i in tr_idx])
        sc, pca = _fit_pca_scaler(x_tr)
        xtr = pca.transform(sc.transform(x_tr))
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(xtr, y_tr)

        for fp in te_fp:
            _, idxs = groups[fp]
            pp = clf.predict_proba(pca.transform(sc.transform(x[idxs])))
            logps[fp] = np.log(pp + 1e-12).mean(axis=0)
    return logps, y_fp_i, classes


def run_logposterior(x: np.ndarray, labels: list[str], K: int, seed: int,
                     n_splits: int = 5,
                     *, math_x: np.ndarray | None = None,
                     math_labels: list[str] | None = None,
                     ) -> tuple[AggMetrics, AggMetrics | None]:
    """Idea 1: fit LR on per-sample features, average log-probs across K.

    With math features, scores math fingerprints under each fold's LR and
    fold-averages the resulting log-probs. Returns ``(in_domain, math_or_None)``.
    """
    rng = np.random.default_rng(seed)
    groups = _group_per_class(labels, K, rng)
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    C = len(classes)
    y_fp_i = np.array([c2i[cls] for cls, _ in groups])
    fp_of = -np.ones(len(labels), dtype=int)
    for fp_idx, (_, idxs) in enumerate(groups):
        fp_of[idxs] = fp_idx

    math_pack = None
    if math_x is not None and math_labels is not None:
        rng_m = np.random.default_rng(seed + 1234)
        m_groups = _group_per_class(math_labels, K, rng_m)
        m_groups = [g for g in m_groups if g[0] in c2i]
        if m_groups:
            math_y_fp_i = np.array([c2i[cls] for cls, _ in m_groups])
            math_logp_acc = np.zeros((len(m_groups), C))
            math_pack = (m_groups, math_y_fp_i, math_logp_acc)

    logps = np.zeros((len(groups), C))
    logps_cal = np.zeros((len(groups), C))
    alpha_vals: list[float] = []
    n_folds_done = 0
    for tr_fp, te_fp in _cv_folds(y_fp_i, n_splits, seed):
        # Fit scalar evidence scale on a calibration split inside the train fold.
        tr_fp = np.asarray(tr_fp)
        y_tr_fp = y_fp_i[tr_fp]
        cal_splits = max(2, min(5, int(np.bincount(y_tr_fp).min())))
        cal_skf = StratifiedKFold(n_splits=cal_splits, shuffle=True,
                                  random_state=seed + 17)
        inner_rel, cal_rel = next(cal_skf.split(np.zeros(len(tr_fp)), y_tr_fp))
        inner_fp = tr_fp[inner_rel]
        cal_fp = tr_fp[cal_rel]
        inner_idx = np.where(np.isin(fp_of, inner_fp))[0]
        x_inner = x[inner_idx]
        y_inner = np.array([c2i[labels[i]] for i in inner_idx])
        sc_in, pca_in = _fit_pca_scaler(x_inner)
        clf_in = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf_in.fit(pca_in.transform(sc_in.transform(x_inner)), y_inner)
        cal_scores = np.zeros((len(cal_fp), C))
        cal_y = y_fp_i[cal_fp]
        for j, fp in enumerate(cal_fp):
            _, idxs = groups[int(fp)]
            pp = _proba_padded(clf_in, pca_in.transform(sc_in.transform(x[idxs])), C)
            cal_scores[j] = np.log(pp + 1e-12).mean(axis=0)
        alpha = _fit_alpha(cal_scores, cal_y)
        alpha_vals.append(alpha)

        tr_idx = np.where(np.isin(fp_of, tr_fp))[0]
        x_tr = x[tr_idx]
        y_tr = np.array([c2i[labels[i]] for i in tr_idx])
        sc, pca = _fit_pca_scaler(x_tr)
        xtr = pca.transform(sc.transform(x_tr))
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(xtr, y_tr)
        for fp in te_fp:
            _, idxs = groups[fp]
            pp = _proba_padded(clf, pca.transform(sc.transform(x[idxs])), C)
            logps[fp] = np.log(pp + 1e-12).mean(axis=0)
            logps_cal[fp] = alpha * logps[fp]
        if math_pack is not None:
            m_groups, _, math_logp_acc = math_pack
            for j, (_, idxs) in enumerate(m_groups):
                pp = _proba_padded(
                    clf, pca.transform(sc.transform(math_x[idxs])), C
                )
                math_logp_acc[j] += np.log(pp + 1e-12).mean(axis=0)
            n_folds_done += 1
    indist = _score_log_scores(y_fp_i, logps, classes, "logposterior", K)
    if alpha_vals:
        probs_cal = _normalise_logps(logps_cal)
        indist.alpha_calib = float(np.mean(alpha_vals))
        indist.nll_calib = float(log_loss(
            y_fp_i, probs_cal, labels=list(range(len(classes)))))
        indist.ece_calib = expected_calibration_error(y_fp_i, probs_cal)
    math_metrics: AggMetrics | None = None
    if math_pack is not None and n_folds_done > 0:
        _, math_y_fp_i, math_logp_acc = math_pack
        math_logp_acc /= n_folds_done
        math_p = _normalise_logps(math_logp_acc)
        math_metrics = _score(math_y_fp_i, math_p, classes,
                              "logposterior_math", K)
    return indist, math_metrics


def run_gaussian_disc(x: np.ndarray, labels: list[str], K: int, seed: int,
                      n_splits: int = 5,
                      *, math_x: np.ndarray | None = None,
                      math_labels: list[str] | None = None,
                      ) -> tuple[AggMetrics, AggMetrics | None]:
    """Idea 2: LDA with Ledoit-Wolf auto-shrinkage + K-scaled decision function.

    sklearn's ``LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')``
    fits shared covariance with auto LW shrinkage; its ``decision_function``
    gives ``log π_c + μ_c^T Σ^{-1} x - 0.5 μ_c^T Σ^{-1} μ_c``. For K samples
    pooled into ``x̄`` the correct log-posterior (up to x̄-independent const) is

        log π_c + K (μ_c^T Σ^{-1} x̄ - 0.5 μ_c^T Σ^{-1} μ_c)
        = K · decision_function(x̄) + (1 - K) · log π_c

    With math features, scores K-pooled math fingerprints under each fold's
    LDA and fold-averages the resulting log-probs.
    """
    rng = np.random.default_rng(seed)
    groups = _group_per_class(labels, K, rng)
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    y_fp_i = np.array([c2i[cls] for cls, _ in groups])
    C = len(classes)

    fp_of = -np.ones(len(labels), dtype=int)
    for fp_idx, (_, idxs) in enumerate(groups):
        fp_of[idxs] = fp_idx

    math_pack = None
    if math_x is not None and math_labels is not None:
        rng_m = np.random.default_rng(seed + 1234)
        m_groups = _group_per_class(math_labels, K, rng_m)
        m_groups = [g for g in m_groups if g[0] in c2i]
        if m_groups:
            math_y_fp_i = np.array([c2i[cls] for cls, _ in m_groups])
            math_logp_acc = np.zeros((len(m_groups), C))
            math_pack = (m_groups, math_y_fp_i, math_logp_acc)

    probs = np.zeros((len(groups), C))
    n_folds_done = 0
    for tr_fp, te_fp in _cv_folds(y_fp_i, n_splits, seed):
        tr_mask = np.isin(fp_of, tr_fp)
        tr_idx = np.where(tr_mask)[0]
        y_tr = np.array([c2i[labels[i]] for i in tr_idx])
        sc, pca = _fit_pca_scaler(x[tr_idx])
        x_tr = pca.transform(sc.transform(x[tr_idx]))

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(x_tr, y_tr)
        col_map = {int(c): k for k, c in enumerate(lda.classes_)}
        log_prior_full = np.full(C, -np.inf)
        for cls_i, k in col_map.items():
            log_prior_full[cls_i] = np.log(lda.priors_[k] + 1e-12)

        xb = np.stack([pca.transform(sc.transform(x[groups[fp][1]])).mean(axis=0) for fp in te_fp])
        df = lda.decision_function(xb)
        df_full = np.full((df.shape[0], C), -np.inf)
        for cls_i, k in col_map.items():
            df_full[:, cls_i] = df[:, k]

        lp_te = K * df_full + (1 - K) * log_prior_full[np.newaxis, :]
        p = _normalise_logps(lp_te)
        for i, fp in enumerate(te_fp):
            probs[fp] = p[i]

        if math_pack is not None:
            m_groups, _, math_logp_acc = math_pack
            mb = np.stack([
                pca.transform(sc.transform(math_x[idxs])).mean(axis=0)
                for _, idxs in m_groups
            ])
            df_m = lda.decision_function(mb)
            df_m_full = np.full((df_m.shape[0], C), -np.inf)
            for cls_i, k in col_map.items():
                df_m_full[:, cls_i] = df_m[:, k]
            lp_m = K * df_m_full + (1 - K) * log_prior_full[np.newaxis, :]
            p_m = _normalise_logps(lp_m)
            math_logp_acc += np.log(p_m + 1e-12)
            n_folds_done += 1

    indist = _score(y_fp_i, probs, classes, "gaussian_disc", K)
    math_metrics: AggMetrics | None = None
    if math_pack is not None and n_folds_done > 0:
        _, math_y_fp_i, math_logp_acc = math_pack
        math_logp_acc /= n_folds_done
        math_p = _normalise_logps(math_logp_acc)
        math_metrics = _score(math_y_fp_i, math_p, classes,
                              "gaussian_disc_math", K)
    return indist, math_metrics


# ---------------------------------------------------------------------------
# Intra-M variants: input x_intra is (N, M_max, D); per-fingerprint = K samples
# × M tokens. Aggregators below mirror the response-level versions but exploit
# the M independent token positions per sample. Optional kwargs:
#   cache_dir / cache_tag — joblib-pickle (sc, pca, clf|lda, ...) per fold so
#       repeated runs (e.g. re-rendering or extending K) skip the LR fit.
#   math_x_intra / math_labels — held-out cross-domain features. For each fold
#       we use that fold's classifier to score every math fingerprint, then
#       ensemble log-probs across folds. Returned as a second AggMetrics.
# ---------------------------------------------------------------------------
def _intra_positions(M: int, M_max: int) -> np.ndarray:
    """Pick M positions uniformly across [0, M_max-1], preserving the last
    endpoint. Matches ``evaluate.py``'s ``intra_m=`` view so the multi-traj
    intra path samples the same tokens as the single-trajectory baseline.

    M=1 returns just the last position (≡ last-token baseline).
    """
    if M < 1 or M > M_max:
        raise ValueError(f"M={M} must be in [1, {M_max}]")
    if M == 1:
        return np.array([M_max - 1], dtype=np.int64)
    idx = np.round(np.linspace(0, M_max - 1, M)).astype(np.int64)
    return np.unique(idx)


def _flatten_train_tokens(x_intra: np.ndarray, labels: list[str], M: int,
                          tr_idx: np.ndarray, c2i: dict[str, int]
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Stack tokens of train samples into (n_tr*M_eff, D); replicate labels per token."""
    D = x_intra.shape[2]
    pos = _intra_positions(M, x_intra.shape[1])
    x_tr = x_intra[np.ix_(tr_idx, pos)].reshape(-1, D)
    y_tr = np.repeat([c2i[labels[i]] for i in tr_idx], len(pos))
    return x_tr, y_tr


def _cache_path(cache_dir: Path | str | None, cache_tag: str | None,
                method: str, M: int, K: int, fold: int) -> Path | None:
    if cache_dir is None or cache_tag is None:
        return None
    return Path(cache_dir) / f"{cache_tag}_{method}_M{M}_K{K}_fold{fold}.joblib"


def _proba_padded(clf: LogisticRegression, x: np.ndarray,
                  C: int) -> np.ndarray:
    """clf.predict_proba padded to the full C-class layout (missing → 0)."""
    p = clf.predict_proba(x)
    if p.shape[1] == C:
        return p
    out = np.zeros((p.shape[0], C))
    for k, c in enumerate(clf.classes_):
        out[:, int(c)] = p[:, k]
    return out


def _lda_decision_padded(lda: LinearDiscriminantAnalysis, x: np.ndarray,
                         C: int) -> tuple[np.ndarray, np.ndarray]:
    """LDA decision function and log-prior padded to C, missing → -inf."""
    df = lda.decision_function(x)
    df_full = np.full((df.shape[0], C), -np.inf)
    log_prior = np.full(C, -np.inf)
    for k, c in enumerate(lda.classes_):
        df_full[:, int(c)] = df[:, k]
        log_prior[int(c)] = np.log(lda.priors_[k] + 1e-12)
    return df_full, log_prior


def _normalise_logps(logps: np.ndarray) -> np.ndarray:
    logps = logps - logps.max(axis=1, keepdims=True)
    p = np.exp(logps)
    p /= p.sum(axis=1, keepdims=True)
    return p


def _build_math(math_x_intra: np.ndarray | None, math_labels: list[str] | None,
                K: int, M: int, seed: int, c2i: dict[str, int]
                ) -> tuple[list, np.ndarray, int] | None:
    """Build (groups, y_fp_i, D) for a held-out math probe set. None if absent.

    Uses an independent RNG (seed + 1234) so the math partition is stable across
    folds — every fold's classifier scores the same math fingerprints.
    """
    if math_x_intra is None or math_labels is None:
        return None
    rng = np.random.default_rng(seed + 1234)
    groups = _group_per_class(math_labels, K, rng)
    if not groups:
        return None
    y_fp_i = np.array([c2i[cls] for cls, _ in groups if cls in c2i])
    if len(y_fp_i) != len(groups):
        groups = [g for g in groups if g[0] in c2i]
        y_fp_i = np.array([c2i[cls] for cls, _ in groups])
    return groups, y_fp_i, math_x_intra.shape[2]


def run_meanpool_lr_intra(x_intra: np.ndarray, labels: list[str], K: int,
                          M: int, seed: int, n_splits: int = 5,
                          *, cache_dir: Path | str | None = None,
                          cache_tag: str | None = None,
                          math_x_intra: np.ndarray | None = None,
                          math_labels: list[str] | None = None,
                          ) -> tuple[AggMetrics, AggMetrics | None]:
    """Mean-pool K samples × M tokens → (D,) per fingerprint, train LR.

    Returns ``(in_domain, math_or_None)``.
    """
    rng = np.random.default_rng(seed)
    groups = _group_per_class(labels, K, rng)
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    C = len(classes)
    D = x_intra.shape[2]
    pos = _intra_positions(M, x_intra.shape[1])
    X_fp = np.stack([
        x_intra[np.ix_(idxs, pos)].reshape(-1, D).mean(axis=0)
        for _, idxs in groups
    ]).astype(np.float64)
    y_fp_i = np.array([c2i[cls] for cls, _ in groups])

    math_pack = _build_math(math_x_intra, math_labels, K, M, seed, c2i)
    if math_pack is not None:
        math_groups, math_y_fp_i, _ = math_pack
        math_pos = _intra_positions(M, math_x_intra.shape[1])
        math_X_fp = np.stack([
            math_x_intra[np.ix_(idxs, math_pos)].reshape(-1, D).mean(axis=0)
            for _, idxs in math_groups
        ]).astype(np.float64)
        math_logp_acc = np.zeros((len(math_groups), C))

    probs = np.zeros((len(groups), C))
    n_folds_done = 0
    for fold_i, (tr, te) in enumerate(_cv_folds(y_fp_i, n_splits, seed)):
        cp = _cache_path(cache_dir, cache_tag, "meanpool_lr_intra", M, K, fold_i)
        if cp is not None and cp.exists():
            sc, pca, clf = joblib.load(cp)
        else:
            sc, pca = _fit_pca_scaler(X_fp[tr])
            xtr = pca.transform(sc.transform(X_fp[tr]))
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
            clf.fit(xtr, y_fp_i[tr])
            if cp is not None:
                cp.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump((sc, pca, clf), cp)
        xte = pca.transform(sc.transform(X_fp[te]))
        probs[te] = _proba_padded(clf, xte, C)

        if math_pack is not None:
            mp = _proba_padded(clf, pca.transform(sc.transform(math_X_fp)), C)
            math_logp_acc += np.log(mp + 1e-12)
            n_folds_done += 1

    indist = _score(y_fp_i, probs, classes, "meanpool_lr_intra", K)
    math_metrics: AggMetrics | None = None
    if math_pack is not None and n_folds_done > 0:
        math_logp_acc /= n_folds_done
        math_p = _normalise_logps(math_logp_acc)
        math_metrics = _score(math_y_fp_i, math_p, classes,
                              "meanpool_lr_intra_math", K)
    return indist, math_metrics


def run_logposterior_intra(x_intra: np.ndarray, labels: list[str], K: int,
                           M: int, seed: int, n_splits: int = 5,
                           *, cache_dir: Path | str | None = None,
                           cache_tag: str | None = None,
                           math_x_intra: np.ndarray | None = None,
                           math_labels: list[str] | None = None,
                           ) -> tuple[AggMetrics, AggMetrics | None]:
    """Train per-token LR on (n_tr*M, D); mean log-probs across K*M tokens.

    Returns ``(in_domain, math_or_None)``.
    """
    rng = np.random.default_rng(seed)
    groups = _group_per_class(labels, K, rng)
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    y_fp_i = np.array([c2i[cls] for cls, _ in groups])
    N = len(labels); D = x_intra.shape[2]; C = len(classes)
    fp_of = -np.ones(N, dtype=int)
    for fp_idx, (_, idxs) in enumerate(groups):
        fp_of[idxs] = fp_idx

    pos = _intra_positions(M, x_intra.shape[1])

    math_pack = _build_math(math_x_intra, math_labels, K, M, seed, c2i)
    if math_pack is not None:
        math_groups, math_y_fp_i, _ = math_pack
        math_pos = _intra_positions(M, math_x_intra.shape[1])
        math_logp_acc = np.zeros((len(math_groups), C))

    logps = np.zeros((len(groups), C))
    n_folds_done = 0
    for fold_i, (tr_fp, te_fp) in enumerate(_cv_folds(y_fp_i, n_splits, seed)):
        cp = _cache_path(cache_dir, cache_tag, "logposterior_intra", M, K, fold_i)
        if cp is not None and cp.exists():
            sc, pca, clf = joblib.load(cp)
        else:
            tr_idx = np.where(np.isin(fp_of, tr_fp))[0]
            x_tr, y_tr = _flatten_train_tokens(x_intra, labels, M, tr_idx, c2i)
            sc, pca = _fit_pca_scaler(x_tr)
            xtr = pca.transform(sc.transform(x_tr))
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
            clf.fit(xtr, y_tr)
            if cp is not None:
                cp.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump((sc, pca, clf), cp)
        for fp in te_fp:
            _, idxs = groups[fp]
            tokens = x_intra[np.ix_(idxs, pos)].reshape(-1, D)
            pp = _proba_padded(clf, pca.transform(sc.transform(tokens)), C)
            logps[fp] = np.log(pp + 1e-12).mean(axis=0)

        if math_pack is not None:
            for j, (_, idxs) in enumerate(math_groups):
                tokens = math_x_intra[np.ix_(idxs, math_pos)].reshape(-1, D)
                pp = _proba_padded(clf, pca.transform(sc.transform(tokens)), C)
                math_logp_acc[j] += np.log(pp + 1e-12).mean(axis=0)
            n_folds_done += 1

    probs = _normalise_logps(logps)
    indist = _score(y_fp_i, probs, classes, "logposterior_intra", K)
    math_metrics: AggMetrics | None = None
    if math_pack is not None and n_folds_done > 0:
        math_logp_acc /= n_folds_done
        math_p = _normalise_logps(math_logp_acc)
        math_metrics = _score(math_y_fp_i, math_p, classes,
                              "logposterior_intra_math", K)
    return indist, math_metrics


def run_gaussian_disc_intra(x_intra: np.ndarray, labels: list[str], K: int,
                            M: int, seed: int, n_splits: int = 5,
                            *, cache_dir: Path | str | None = None,
                            cache_tag: str | None = None,
                            math_x_intra: np.ndarray | None = None,
                            math_labels: list[str] | None = None,
                            ) -> tuple[AggMetrics, AggMetrics | None]:
    """Train LDA on per-token features. K_eff = K*M tokens → K_eff-scaled
    decision function on the K*M-mean. Statistically iffy (intra-tokens are
    correlated) but a fair baseline analogue of run_gaussian_disc.

    Returns ``(in_domain, math_or_None)``.
    """
    rng = np.random.default_rng(seed)
    groups = _group_per_class(labels, K, rng)
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    y_fp_i = np.array([c2i[cls] for cls, _ in groups])
    N = len(labels); D = x_intra.shape[2]; C = len(classes)
    fp_of = -np.ones(N, dtype=int)
    for fp_idx, (_, idxs) in enumerate(groups):
        fp_of[idxs] = fp_idx
    pos = _intra_positions(M, x_intra.shape[1])
    K_eff = K * len(pos)

    math_pack = _build_math(math_x_intra, math_labels, K, M, seed, c2i)
    if math_pack is not None:
        math_groups, math_y_fp_i, _ = math_pack
        math_pos = _intra_positions(M, math_x_intra.shape[1])
        math_logp_acc = np.zeros((len(math_groups), C))

    probs = np.zeros((len(groups), C))
    n_folds_done = 0
    for fold_i, (tr_fp, te_fp) in enumerate(_cv_folds(y_fp_i, n_splits, seed)):
        cp = _cache_path(cache_dir, cache_tag, "gaussian_disc_intra", M, K, fold_i)
        if cp is not None and cp.exists():
            sc, pca, lda = joblib.load(cp)
        else:
            tr_idx = np.where(np.isin(fp_of, tr_fp))[0]
            x_tr, y_tr = _flatten_train_tokens(x_intra, labels, M, tr_idx, c2i)
            sc, pca = _fit_pca_scaler(x_tr)
            x_tr_p = pca.transform(sc.transform(x_tr))
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda.fit(x_tr_p, y_tr)
            if cp is not None:
                cp.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump((sc, pca, lda), cp)

        xb = np.stack([
            pca.transform(sc.transform(
                x_intra[np.ix_(groups[fp][1], pos)].reshape(-1, D)
            )).mean(axis=0)
            for fp in te_fp
        ])
        df_full, log_prior = _lda_decision_padded(lda, xb, C)
        lp_te = K_eff * df_full + (1 - K_eff) * log_prior[np.newaxis, :]
        p = _normalise_logps(lp_te)
        for i, fp in enumerate(te_fp):
            probs[fp] = p[i]

        if math_pack is not None:
            mb = np.stack([
                pca.transform(sc.transform(
                    math_x_intra[np.ix_(idxs, math_pos)].reshape(-1, D)
                )).mean(axis=0)
                for _, idxs in math_groups
            ])
            df_m, log_prior_m = _lda_decision_padded(lda, mb, C)
            lp_m = K_eff * df_m + (1 - K_eff) * log_prior_m[np.newaxis, :]
            # store as log-probs (after normalisation) for fold-averaging
            p_m = _normalise_logps(lp_m)
            math_logp_acc += np.log(p_m + 1e-12)
            n_folds_done += 1

    indist = _score(y_fp_i, probs, classes, "gaussian_disc_intra", K)
    math_metrics: AggMetrics | None = None
    if math_pack is not None and n_folds_done > 0:
        math_logp_acc /= n_folds_done
        math_p = _normalise_logps(math_logp_acc)
        math_metrics = _score(math_y_fp_i, math_p, classes,
                              "gaussian_disc_intra_math", K)
    return indist, math_metrics


# ---------------------------------------------------------------------------
# K-sweep variants for logposterior_intra and gaussian_disc_intra. The trick:
# for both methods the trained classifier is K-independent (per-token training);
# K only affects how test fingerprints are pooled and (for LDA) the K_eff
# decision-function scaling. So we use a sample-level k-fold and train ONE
# classifier per (M, fold) — the K loop happens at evaluation time. ~6× faster
# than the per-K functions when sweeping K ∈ {1,5,10,20,50,100}.
# ---------------------------------------------------------------------------
def _sample_kfold(labels: list[str], n_splits: int, seed: int
                  ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stratified k-fold over sample indices."""
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[l] for l in labels])
    n_splits = max(2, min(n_splits, int(np.bincount(y).min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from skf.split(np.zeros(len(labels)), y)


def _group_subset(sub_idx: np.ndarray, labels: list[str], K: int,
                  rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """Group sub_idx samples per class, chunk into K-sized fingerprints."""
    by_cls: dict[str, list[int]] = {}
    for i in sub_idx:
        by_cls.setdefault(labels[int(i)], []).append(int(i))
    groups: list[tuple[str, np.ndarray]] = []
    for cls, idxs in by_cls.items():
        arr = np.asarray(idxs)
        rng.shuffle(arr)
        n_g = len(arr) // K
        if n_g == 0:
            continue
        chunked = arr[: n_g * K].reshape(n_g, K)
        for row in chunked:
            groups.append((cls, row))
    return groups


def _inner_train_cal_split(indices: np.ndarray, labels: list[str], seed: int,
                           cal_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split of an outer-train index set into train/calibration."""
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[labels[int(i)]] for i in indices])
    n_splits = max(2, int(round(1.0 / cal_frac)))
    n_splits = min(n_splits, int(np.bincount(y).min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_rel, cal_rel = next(skf.split(np.zeros(len(indices)), y))
    return indices[tr_rel], indices[cal_rel]


def _fit_intra_alpha_for_fold(
    x_intra: np.ndarray,
    labels: list[str],
    M: int,
    Ks: list[int],
    tr_idx: np.ndarray,
    c2i: dict[str, int],
    seed: int,
) -> dict[int, float]:
    """Fit one group-level evidence scale per K using only outer-train data."""
    D = x_intra.shape[2]
    C = len(c2i)
    pos = _intra_positions(M, x_intra.shape[1])
    inner_tr, cal_idx = _inner_train_cal_split(tr_idx, labels, seed)
    x_tr, y_tr = _flatten_train_tokens(x_intra, labels, M, inner_tr, c2i)
    sc, pca = _fit_pca_scaler(x_tr)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    clf.fit(pca.transform(sc.transform(x_tr)), y_tr)

    alphas: dict[int, float] = {}
    for K in Ks:
        rng = np.random.default_rng(seed + 5000 + K * 1543)
        groups = _group_subset(cal_idx, labels, K, rng)
        groups = [(c, idxs) for c, idxs in groups if c in c2i]
        if not groups:
            continue
        y_cal = np.array([c2i[c] for c, _ in groups])
        scores = np.zeros((len(groups), C))
        for j, (_, idxs) in enumerate(groups):
            tokens = x_intra[np.ix_(idxs, pos)].reshape(-1, D)
            pp = _proba_padded(clf, pca.transform(sc.transform(tokens)), C)
            scores[j] = np.log(pp + 1e-12).mean(axis=0)
        alphas[K] = _fit_alpha(scores, y_cal)
    return alphas


def _build_math_groups_per_K(math_x_intra: np.ndarray | None,
                             math_labels: list[str] | None,
                             Ks: list[int], seed: int,
                             c2i: dict[str, int]) -> dict:
    """For each K build a fixed math-fingerprint partition (used by every fold).

    Returns ``{K: (groups, y_fp_i)}`` for K values that yield ≥1 fingerprint.
    """
    out: dict[int, tuple[list, np.ndarray]] = {}
    if math_x_intra is None or math_labels is None:
        return out
    all_idx = np.arange(len(math_labels))
    for K in Ks:
        rng = np.random.default_rng(seed + 1234 + K)
        groups = _group_subset(all_idx, math_labels, K, rng)
        groups = [(c, idxs) for c, idxs in groups if c in c2i]
        if not groups:
            continue
        y_fp_i = np.array([c2i[c] for c, _ in groups])
        out[K] = (groups, y_fp_i)
    return out


def _fit_flat_alpha_for_fold(
    x: np.ndarray,
    labels: list[str],
    Ks: list[int],
    tr_idx: np.ndarray,
    c2i: dict[str, int],
    seed: int,
) -> dict[int, float]:
    """Fit one group-level evidence scale per K on flat (N, D) features.

    Mirrors :func:`_fit_intra_alpha_for_fold` but for per-trajectory features
    (no intra-token unrolling). Used by :func:`run_logposterior_sweep_flat`.
    """
    C = len(c2i)
    inner_tr, cal_idx = _inner_train_cal_split(tr_idx, labels, seed)
    x_tr = x[inner_tr]
    y_tr = np.array([c2i[labels[int(i)]] for i in inner_tr])
    sc, pca = _fit_pca_scaler(x_tr)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    clf.fit(pca.transform(sc.transform(x_tr)), y_tr)

    alphas: dict[int, float] = {}
    for K in Ks:
        rng = np.random.default_rng(seed + 5000 + K * 1543)
        groups = _group_subset(cal_idx, labels, K, rng)
        groups = [(c, idxs) for c, idxs in groups if c in c2i]
        if not groups:
            continue
        y_cal = np.array([c2i[c] for c, _ in groups])
        scores = np.zeros((len(groups), C))
        for j, (_, idxs) in enumerate(groups):
            pp = _proba_padded(clf, pca.transform(sc.transform(x[idxs])), C)
            scores[j] = np.log(pp + 1e-12).mean(axis=0)
        alphas[K] = _fit_alpha(scores, y_cal)
    return alphas


def _build_flat_math_groups_per_K(math_x: np.ndarray | None,
                                  math_labels: list[str] | None,
                                  Ks: list[int], seed: int,
                                  c2i: dict[str, int]) -> dict:
    """Flat-feature analogue of :func:`_build_math_groups_per_K`."""
    out: dict[int, tuple[list, np.ndarray]] = {}
    if math_x is None or math_labels is None:
        return out
    all_idx = np.arange(len(math_labels))
    for K in Ks:
        rng = np.random.default_rng(seed + 1234 + K)
        groups = _group_subset(all_idx, math_labels, K, rng)
        groups = [(c, idxs) for c, idxs in groups if c in c2i]
        if not groups:
            continue
        y_fp_i = np.array([c2i[c] for c, _ in groups])
        out[K] = (groups, y_fp_i)
    return out


def run_logposterior_sweep_flat(
    x: np.ndarray,
    labels: list[str],
    Ks: list[int],
    seed: int = 42,
    n_splits: int = 5,
    *,
    cache_dir: Path | str | None = None,
    cache_tag: str | None = None,
    cache_method: str = "logposterior_flat",
    math_x: np.ndarray | None = None,
    math_labels: list[str] | None = None,
) -> tuple[
    dict[int, tuple[AggMetrics, AggMetrics | None]],
    dict[int, LogPostRawSweep],
]:
    """K-sweep log-posterior on flat (N, D) per-trajectory features.

    Trains one (sc, pca, LR) per fold on per-trajectory features; the K loop
    happens at evaluation time by grouping each fold's test samples into
    K-sized fingerprints and averaging their log-probs. Mirrors
    :func:`run_logposterior_intra_sweep_full` but skips intra-token unrolling,
    so the same code path serves:

    - mean-pooled intra features (caller pre-pools M tokens into one feature
      per sample, baking ``M`` into ``cache_method``)
    - flat baseline encoder features (mpnet / bge / qwen3-emb)

    Cache key pattern: ``<cache_tag>_<cache_method>_fold{fold}.joblib`` —
    callers needing per-M caches must bake ``M`` into ``cache_method``
    (e.g. ``f"meanpool_intra_logpost_M{M}"``).
    """
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    C = len(classes)

    in_acc: dict[int, dict] = {K: {"y": [], "logp": [], "logp_cal": []} for K in Ks}
    alpha_acc: dict[int, list[float]] = {K: [] for K in Ks}
    math_groups = _build_flat_math_groups_per_K(math_x, math_labels, Ks,
                                                seed, c2i)
    math_acc: dict[int, np.ndarray] = {
        K: np.zeros((len(g), C)) for K, (g, _) in math_groups.items()
    }
    n_folds_done = 0

    for fold_i, (tr_idx, te_idx) in enumerate(
            _sample_kfold(labels, n_splits, seed)):
        fold_alphas = _fit_flat_alpha_for_fold(
            x, labels, Ks, tr_idx, c2i, seed + fold_i * 1009
        )
        cp = (Path(cache_dir) / f"{cache_tag}_{cache_method}_fold{fold_i}.joblib"
              if cache_dir is not None and cache_tag is not None else None)
        if cp is not None and cp.exists():
            sc, pca, clf = joblib.load(cp)
        else:
            x_tr = x[tr_idx]
            y_tr = np.array([c2i[labels[int(i)]] for i in tr_idx])
            sc, pca = _fit_pca_scaler(x_tr)
            xtr = pca.transform(sc.transform(x_tr))
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
            clf.fit(xtr, y_tr)
            if cp is not None:
                cp.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump((sc, pca, clf), cp)

        for K in Ks:
            rng_K = np.random.default_rng(seed + 100 + K * 7919 + fold_i * 31)
            te_groups = _group_subset(te_idx, labels, K, rng_K)
            te_groups = [(c, idxs) for c, idxs in te_groups if c in c2i]
            if not te_groups:
                continue
            te_y = np.array([c2i[c] for c, _ in te_groups])
            te_logp = np.zeros((len(te_groups), C))
            for j, (_, idxs) in enumerate(te_groups):
                pp = _proba_padded(clf, pca.transform(sc.transform(x[idxs])), C)
                te_logp[j] = np.log(pp + 1e-12).mean(axis=0)
            in_acc[K]["y"].append(te_y)
            in_acc[K]["logp"].append(te_logp)
            if K in fold_alphas:
                alpha_acc[K].append(fold_alphas[K])
                in_acc[K]["logp_cal"].append(fold_alphas[K] * te_logp)

        for K, (mg, _) in math_groups.items():
            for j, (_, idxs) in enumerate(mg):
                pp = _proba_padded(clf, pca.transform(sc.transform(math_x[idxs])), C)
                math_acc[K][j] += np.log(pp + 1e-12).mean(axis=0)
        n_folds_done += 1

    results: dict[int, tuple[AggMetrics, AggMetrics | None]] = {}
    raws: dict[int, LogPostRawSweep] = {}
    method_label = cache_method  # used as the AggMetrics.method tag
    for K in Ks:
        if not in_acc[K]["y"]:
            continue
        y_all = np.concatenate(in_acc[K]["y"])
        logp_all = np.concatenate(in_acc[K]["logp"], axis=0)
        m_in = _score_log_scores(y_all, logp_all, classes, method_label, K)
        if in_acc[K]["logp_cal"]:
            logp_cal = np.concatenate(in_acc[K]["logp_cal"], axis=0)
            probs_cal = _normalise_logps(logp_cal)
            m_in.alpha_calib = float(np.mean(alpha_acc[K]))
            m_in.nll_calib = float(log_loss(
                y_all, probs_cal, labels=list(range(len(classes)))))
            m_in.ece_calib = expected_calibration_error(y_all, probs_cal)
        m_math: AggMetrics | None = None
        if K in math_groups and n_folds_done > 0:
            mg, mg_y = math_groups[K]
            avg = math_acc[K] / n_folds_done
            p = _normalise_logps(avg)
            m_math = _score(mg_y, p, classes, f"{method_label}_math", K)
        results[K] = (m_in, m_math)
        raws[K] = LogPostRawSweep(
            K=int(K), classes=list(classes),
            y_per_fold=[y for y in in_acc[K]["y"]],
            logp_per_fold=[lp for lp in in_acc[K]["logp"]],
            alpha=(float(np.mean(alpha_acc[K])) if alpha_acc[K] else None),
        )
    return results, raws


def run_logposterior_intra_sweep(
    x_intra: np.ndarray, labels: list[str], Ks: list[int], M: int,
    seed: int, n_splits: int = 5, *,
    cache_dir: Path | str | None = None, cache_tag: str | None = None,
    math_x_intra: np.ndarray | None = None,
    math_labels: list[str] | None = None,
) -> dict[int, tuple[AggMetrics, AggMetrics | None]]:
    """Backward-compat wrapper: returns metrics dict only.

    For callers that also need the raw per-fold (y, logp) — e.g., to compute
    Pair-AUC / mAP / clustering on the fingerprints — use
    :func:`run_logposterior_intra_sweep_full` instead.
    """
    metrics, _raw = run_logposterior_intra_sweep_full(
        x_intra, labels, Ks, M, seed, n_splits=n_splits,
        cache_dir=cache_dir, cache_tag=cache_tag,
        math_x_intra=math_x_intra, math_labels=math_labels,
    )
    return metrics


def run_logposterior_intra_sweep_full(
    x_intra: np.ndarray, labels: list[str], Ks: list[int], M: int,
    seed: int, n_splits: int = 5, *,
    cache_dir: Path | str | None = None, cache_tag: str | None = None,
    math_x_intra: np.ndarray | None = None,
    math_labels: list[str] | None = None,
) -> tuple[
    dict[int, tuple[AggMetrics, AggMetrics | None]],
    dict[int, LogPostRawSweep],
]:
    """K-sweep version: trains one (sc, pca, LR) per (M, fold), evaluates at all K.

    Cache key is K-free: ``<tag>_logposterior_intra_M{M}_fold{f}.joblib``.
    Returns ``(metrics, raw)`` where ``raw[K]`` exposes the per-fold
    fingerprint labels and log-prob matrices used by ``_score_log_scores``,
    so downstream metrics can be computed on the same fingerprints.
    """
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    D = x_intra.shape[2]; C = len(classes)

    in_acc: dict[int, dict] = {K: {"y": [], "logp": [], "logp_cal": []} for K in Ks}
    alpha_acc: dict[int, list[float]] = {K: [] for K in Ks}
    math_groups = _build_math_groups_per_K(math_x_intra, math_labels, Ks,
                                           seed, c2i)
    math_acc: dict[int, np.ndarray] = {
        K: np.zeros((len(g), C)) for K, (g, _) in math_groups.items()
    }
    n_folds_done = 0
    pos = _intra_positions(M, x_intra.shape[1])
    math_pos = (_intra_positions(M, math_x_intra.shape[1])
                if math_x_intra is not None else None)

    for fold_i, (tr_idx, te_idx) in enumerate(
            _sample_kfold(labels, n_splits, seed)):
        fold_alphas = _fit_intra_alpha_for_fold(
            x_intra, labels, M, Ks, tr_idx, c2i, seed + fold_i * 1009
        )
        cp = (Path(cache_dir) / f"{cache_tag}_logposterior_intra_M{M}_fold{fold_i}.joblib"
              if cache_dir is not None and cache_tag is not None else None)
        if cp is not None and cp.exists():
            sc, pca, clf = joblib.load(cp)
        else:
            x_tr, y_tr = _flatten_train_tokens(x_intra, labels, M, tr_idx, c2i)
            sc, pca = _fit_pca_scaler(x_tr)
            xtr = pca.transform(sc.transform(x_tr))
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
            clf.fit(xtr, y_tr)
            if cp is not None:
                cp.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump((sc, pca, clf), cp)

        # In-domain: for each K, group THIS fold's test samples into fingerprints
        for K in Ks:
            rng_K = np.random.default_rng(seed + 100 + K * 7919 + fold_i * 31)
            te_groups = _group_subset(te_idx, labels, K, rng_K)
            te_groups = [(c, idxs) for c, idxs in te_groups if c in c2i]
            if not te_groups:
                continue
            te_y = np.array([c2i[c] for c, _ in te_groups])
            te_logp = np.zeros((len(te_groups), C))
            for j, (_, idxs) in enumerate(te_groups):
                tokens = x_intra[np.ix_(idxs, pos)].reshape(-1, D)
                pp = _proba_padded(clf, pca.transform(sc.transform(tokens)), C)
                te_logp[j] = np.log(pp + 1e-12).mean(axis=0)
            in_acc[K]["y"].append(te_y)
            in_acc[K]["logp"].append(te_logp)
            if K in fold_alphas:
                alpha_acc[K].append(fold_alphas[K])
                in_acc[K]["logp_cal"].append(fold_alphas[K] * te_logp)

        # Math: predict on fixed math fingerprints per K, accumulate log-probs
        for K, (mg, _) in math_groups.items():
            for j, (_, idxs) in enumerate(mg):
                tokens = math_x_intra[np.ix_(idxs, math_pos)].reshape(-1, D)
                pp = _proba_padded(clf, pca.transform(sc.transform(tokens)), C)
                math_acc[K][j] += np.log(pp + 1e-12).mean(axis=0)
        n_folds_done += 1

    results: dict[int, tuple[AggMetrics, AggMetrics | None]] = {}
    raws: dict[int, LogPostRawSweep] = {}
    for K in Ks:
        if not in_acc[K]["y"]:
            continue
        y_all = np.concatenate(in_acc[K]["y"])
        logp_all = np.concatenate(in_acc[K]["logp"], axis=0)
        m_in = _score_log_scores(y_all, logp_all, classes,
                                 "logposterior_intra", K)
        if in_acc[K]["logp_cal"]:
            logp_cal = np.concatenate(in_acc[K]["logp_cal"], axis=0)
            probs_cal = _normalise_logps(logp_cal)
            m_in.alpha_calib = float(np.mean(alpha_acc[K]))
            m_in.nll_calib = float(log_loss(
                y_all, probs_cal, labels=list(range(len(classes)))))
            m_in.ece_calib = expected_calibration_error(y_all, probs_cal)
        m_math: AggMetrics | None = None
        if K in math_groups and n_folds_done > 0:
            mg, mg_y = math_groups[K]
            avg = math_acc[K] / n_folds_done
            p = _normalise_logps(avg)
            m_math = _score(mg_y, p, classes, "logposterior_intra_math", K)
        results[K] = (m_in, m_math)
        raws[K] = LogPostRawSweep(
            K=int(K), classes=list(classes),
            y_per_fold=[y for y in in_acc[K]["y"]],
            logp_per_fold=[lp for lp in in_acc[K]["logp"]],
            alpha=(float(np.mean(alpha_acc[K])) if alpha_acc[K] else None),
        )
    return results, raws


def run_gaussian_disc_intra_sweep(
    x_intra: np.ndarray, labels: list[str], Ks: list[int], M: int,
    seed: int, n_splits: int = 5, *,
    cache_dir: Path | str | None = None, cache_tag: str | None = None,
    math_x_intra: np.ndarray | None = None,
    math_labels: list[str] | None = None,
) -> dict[int, tuple[AggMetrics, AggMetrics | None]]:
    """K-sweep version of gaussian_disc_intra. Train (sc, pca, LDA) once per
    (M, fold); sweep K with K_eff = K*M decision-function scaling at test."""
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    D = x_intra.shape[2]; C = len(classes)

    in_acc: dict[int, dict] = {K: {"y": [], "logp": []} for K in Ks}
    math_groups = _build_math_groups_per_K(math_x_intra, math_labels, Ks,
                                           seed, c2i)
    math_acc: dict[int, np.ndarray] = {
        K: np.zeros((len(g), C)) for K, (g, _) in math_groups.items()
    }
    n_folds_done = 0
    pos = _intra_positions(M, x_intra.shape[1])
    M_eff = len(pos)
    math_pos = (_intra_positions(M, math_x_intra.shape[1])
                if math_x_intra is not None else None)

    for fold_i, (tr_idx, te_idx) in enumerate(
            _sample_kfold(labels, n_splits, seed)):
        cp = (Path(cache_dir) / f"{cache_tag}_gaussian_disc_intra_M{M}_fold{fold_i}.joblib"
              if cache_dir is not None and cache_tag is not None else None)
        if cp is not None and cp.exists():
            sc, pca, lda = joblib.load(cp)
        else:
            x_tr, y_tr = _flatten_train_tokens(x_intra, labels, M, tr_idx, c2i)
            sc, pca = _fit_pca_scaler(x_tr)
            x_tr_p = pca.transform(sc.transform(x_tr))
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda.fit(x_tr_p, y_tr)
            if cp is not None:
                cp.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump((sc, pca, lda), cp)

        for K in Ks:
            K_eff = K * M_eff
            rng_K = np.random.default_rng(seed + 100 + K * 7919 + fold_i * 31)
            te_groups = _group_subset(te_idx, labels, K, rng_K)
            te_groups = [(c, idxs) for c, idxs in te_groups if c in c2i]
            if not te_groups:
                continue
            te_y = np.array([c2i[c] for c, _ in te_groups])
            xb = np.stack([
                pca.transform(sc.transform(
                    x_intra[np.ix_(idxs, pos)].reshape(-1, D))).mean(axis=0)
                for _, idxs in te_groups
            ])
            df_full, log_prior = _lda_decision_padded(lda, xb, C)
            lp_te = K_eff * df_full + (1 - K_eff) * log_prior[np.newaxis, :]
            p_te = _normalise_logps(lp_te)
            in_acc[K]["y"].append(te_y)
            in_acc[K]["logp"].append(np.log(p_te + 1e-12))

        for K, (mg, _) in math_groups.items():
            K_eff = K * M_eff
            mb = np.stack([
                pca.transform(sc.transform(
                    math_x_intra[np.ix_(idxs, math_pos)].reshape(-1, D))).mean(axis=0)
                for _, idxs in mg
            ])
            df_m, lp_m_prior = _lda_decision_padded(lda, mb, C)
            lp_m = K_eff * df_m + (1 - K_eff) * lp_m_prior[np.newaxis, :]
            p_m = _normalise_logps(lp_m)
            math_acc[K] += np.log(p_m + 1e-12)
        n_folds_done += 1

    results: dict[int, tuple[AggMetrics, AggMetrics | None]] = {}
    for K in Ks:
        if not in_acc[K]["y"]:
            continue
        y_all = np.concatenate(in_acc[K]["y"])
        logp_all = np.concatenate(in_acc[K]["logp"], axis=0)
        probs = _normalise_logps(logp_all)
        m_in = _score(y_all, probs, classes, "gaussian_disc_intra", K)
        m_math: AggMetrics | None = None
        if K in math_groups and n_folds_done > 0:
            mg, mg_y = math_groups[K]
            avg = math_acc[K] / n_folds_done
            p = _normalise_logps(avg)
            m_math = _score(mg_y, p, classes, "gaussian_disc_intra_math", K)
        results[K] = (m_in, m_math)
    return results


@dataclass(slots=True)
class SprtDecision:
    alpha: float
    tau: int               # step at which decision taken (≤ K_max; K_max if never)
    decided: bool          # whether threshold crossed before K_max
    class_true: int
    class_pred: int


def _oof_logprobs(x: np.ndarray, labels: list[str], seed: int,
                  n_splits: int = 5) -> tuple[np.ndarray, list[str]]:
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[l] for l in labels])
    N, C = len(x), len(classes)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    logps = np.zeros((N, C))
    for tr, te in skf.split(x, y):
        sc, pca = _fit_pca_scaler(x[tr])
        xtr = pca.transform(sc.transform(x[tr]))
        xte = pca.transform(sc.transform(x[te]))
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(xtr, y[tr])
        logps[te] = np.log(clf.predict_proba(xte) + 1e-12)
    return logps, classes


def _fit_temperature(logits: np.ndarray, y_true: np.ndarray,
                     bounds: tuple[float, float] = (0.05, 50.0)) -> float:
    """Fit a single positive scalar T minimising NLL of softmax(logits / T).

    Convex 1-D problem; ``scipy.minimize_scalar(method='bounded')`` solves it
    reliably. ``T > 1`` softens (the case for over-confident classifiers);
    ``T < 1`` sharpens.
    """
    from scipy.optimize import minimize_scalar

    def nll(T: float) -> float:
        s = logits / T
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s)
        p = p / p.sum(axis=1, keepdims=True)
        return float(-np.log(p[np.arange(len(y_true)), y_true] + 1e-12).mean())

    res = minimize_scalar(nll, bounds=bounds, method="bounded",
                          options={"xatol": 1e-3})
    return float(res.x)


def _oof_logprobs_calibrated(x: np.ndarray, labels: list[str], seed: int,
                             n_splits: int = 5, cal_frac: float = 0.2
                             ) -> tuple[np.ndarray, list[str], list[float]]:
    """Same as ``_oof_logprobs`` but applies per-fold temperature scaling.

    Inside each train fold we hold out a calibration split (``cal_frac``),
    fit LR on the remainder, fit ``T`` to minimise NLL on the cal split, then
    refit LR on the full train and divide test logits by ``T``.
    """
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[l] for l in labels])
    N, C = len(x), len(classes)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    logps = np.zeros((N, C))
    Ts: list[float] = []
    n_cal_splits = int(round(1.0 / cal_frac))
    for tr, te in skf.split(x, y):
        cal_skf = StratifiedKFold(n_splits=n_cal_splits, shuffle=True,
                                  random_state=seed + 1)
        tr_inner_rel, cal_rel = next(cal_skf.split(x[tr], y[tr]))
        tr_inner = tr[tr_inner_rel]
        cal = tr[cal_rel]

        sc_in, pca_in = _fit_pca_scaler(x[tr_inner])
        clf_in = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf_in.fit(pca_in.transform(sc_in.transform(x[tr_inner])), y[tr_inner])
        cal_logits = clf_in.decision_function(
            pca_in.transform(sc_in.transform(x[cal]))
        )
        if cal_logits.ndim == 1:  # 2-class edge case (not expected here)
            cal_logits = np.column_stack([-cal_logits, cal_logits])
        T = _fit_temperature(cal_logits, y[cal])
        Ts.append(T)

        sc, pca = _fit_pca_scaler(x[tr])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(pca.transform(sc.transform(x[tr])), y[tr])
        te_logits = clf.decision_function(
            pca.transform(sc.transform(x[te]))
        )
        if te_logits.ndim == 1:
            te_logits = np.column_stack([-te_logits, te_logits])
        col_map = {int(c): k for k, c in enumerate(clf.classes_)}
        te_full = np.full((te_logits.shape[0], C), -1e9)
        for cls_i, k in col_map.items():
            te_full[:, cls_i] = te_logits[:, k]
        scaled = te_full / T
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled)
        p = p / p.sum(axis=1, keepdims=True)
        logps[te] = np.log(p + 1e-12)
    return logps, classes, Ts


def _oof_logprobs_gd(x: np.ndarray, labels: list[str], seed: int,
                     n_splits: int = 5) -> tuple[np.ndarray, list[str]]:
    """Per-sample log-posteriors from LDA + auto Ledoit-Wolf shrinkage.

    Uses ``decision_function`` which equals log p(c|x) up to a c-independent
    constant (the data evidence). For a balanced 9-class panel that constant
    cancels inside the SPRT difference rule.
    """
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[l] for l in labels])
    N, C = len(x), len(classes)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.full((N, C), -1e9, dtype=np.float64)
    for tr, te in skf.split(x, y):
        sc, pca = _fit_pca_scaler(x[tr])
        xtr = pca.transform(sc.transform(x[tr]))
        xte = pca.transform(sc.transform(x[te]))
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(xtr, y[tr])
        df = lda.decision_function(xte)
        col_map = {int(c): k for k, c in enumerate(lda.classes_)}
        for cls_i, k in col_map.items():
            out[te, cls_i] = df[:, k]
    return out, classes


def per_sample_calibration(logps: np.ndarray, y_int: np.ndarray,
                           classes: list[str]) -> dict:
    """Return accuracy / NLL / ECE for per-sample class log-scores."""
    s = logps - logps.max(axis=1, keepdims=True)
    p = np.exp(s)
    p = p / p.sum(axis=1, keepdims=True)
    preds = p.argmax(axis=1)
    acc = float((preds == y_int).mean())
    nll = float(log_loss(y_int, p, labels=list(range(len(classes)))))
    ece = float(expected_calibration_error(y_int, p))
    return {"accuracy": acc, "nll": nll, "ece": ece, "n": int(len(y_int))}


def _msprt_from_logps(logps: np.ndarray, y: np.ndarray, classes: list[str],
                     seed: int, *, K_max: int, alphas: tuple[float, ...],
                     n_streams_per_class: int) -> dict:
    C = len(classes)
    rng = np.random.default_rng(seed)
    results: list[SprtDecision] = []
    thresholds = {a: np.log((C - 1) * (1 - a) / a) for a in alphas}
    for ci in range(C):
        cls_idx = np.where(y == ci)[0]
        for _ in range(n_streams_per_class):
            stream = rng.choice(cls_idx, size=min(K_max, len(cls_idx)),
                                replace=False)
            cum = np.cumsum(logps[stream], axis=0)
            for alpha, thr in thresholds.items():
                decided = False
                for t in range(1, cum.shape[0] + 1):
                    s = cum[t - 1]
                    top = s.max()
                    second = np.partition(s, -2)[-2]
                    if top - second >= thr:
                        results.append(SprtDecision(
                            alpha=alpha, tau=t, decided=True,
                            class_true=ci, class_pred=int(s.argmax()),
                        ))
                        decided = True
                        break
                if not decided:
                    results.append(SprtDecision(
                        alpha=alpha, tau=cum.shape[0], decided=False,
                        class_true=ci, class_pred=int(cum[-1].argmax()),
                    ))
    summary: dict = {"K_max": K_max, "classes": classes,
                     "n_streams_per_class": n_streams_per_class, "per_alpha": {}}
    for alpha in alphas:
        sub = [r for r in results if r.alpha == alpha]
        tau = np.array([r.tau for r in sub])
        decided = np.array([r.decided for r in sub])
        correct = np.array([r.class_pred == r.class_true for r in sub])
        summary["per_alpha"][alpha] = {
            "E_tau": float(tau.mean()),
            "median_tau": float(np.median(tau)),
            "p10_tau": float(np.quantile(tau, 0.1)),
            "p90_tau": float(np.quantile(tau, 0.9)),
            "decided_frac": float(decided.mean()),
            "overall_acc": float(correct.mean()),
            "acc_when_decided": float(correct[decided].mean()) if decided.any() else float("nan"),
            "target_threshold": float(np.log((C - 1) * (1 - alpha) / alpha)),
        }
    return summary


def run_calibrated_msprt_panel(x: np.ndarray, labels: list[str], seed: int,
                               *, K_max: int = 32,
                               alphas: tuple[float, ...] = (0.1, 0.05, 0.01, 0.001),
                               n_streams_per_class: int = 50,
                               n_splits: int = 5) -> dict:
    """Run per-sample calibration + SPRT for three increment sources:
    vanilla LR, temperature-scaled LR, Gaussian-discriminant LDA.
    Returns a dict ready to dump to JSON.
    """
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    y_int = np.array([c2i[l] for l in labels])

    sources: dict[str, dict] = {}

    # 1. vanilla LR
    lp_lr, _ = _oof_logprobs(x, labels, seed, n_splits=n_splits)
    sources["vanilla_lr"] = {
        "calibration": per_sample_calibration(lp_lr, y_int, classes),
        "sprt": _msprt_from_logps(lp_lr, y_int, classes, seed,
                                  K_max=K_max, alphas=alphas,
                                  n_streams_per_class=n_streams_per_class),
        "T_per_fold": None,
    }

    # 2. temperature-scaled LR
    lp_cal, _, Ts = _oof_logprobs_calibrated(x, labels, seed, n_splits=n_splits)
    sources["calibrated_lr"] = {
        "calibration": per_sample_calibration(lp_cal, y_int, classes),
        "sprt": _msprt_from_logps(lp_cal, y_int, classes, seed,
                                  K_max=K_max, alphas=alphas,
                                  n_streams_per_class=n_streams_per_class),
        "T_per_fold": Ts,
    }

    # 3. Gaussian discriminant
    lp_gd, _ = _oof_logprobs_gd(x, labels, seed, n_splits=n_splits)
    sources["gaussian_disc"] = {
        "calibration": per_sample_calibration(lp_gd, y_int, classes),
        "sprt": _msprt_from_logps(lp_gd, y_int, classes, seed,
                                  K_max=K_max, alphas=alphas,
                                  n_streams_per_class=n_streams_per_class),
        "T_per_fold": None,
    }

    return {"classes": classes, "sources": sources, "alphas": list(alphas),
            "K_max": K_max, "n_streams_per_class": n_streams_per_class}


def run_msprt(x: np.ndarray, labels: list[str], seed: int,
              *, K_max: int = 32, alphas: tuple[float, ...] = (0.1, 0.05, 0.01, 0.001),
              n_streams_per_class: int = 50,
              n_splits: int = 5) -> dict:
    """Idea 3: Multi-hypothesis SPRT on streams built from out-of-fold log-probs.

    Decision rule (generalised SPRT, Dragalin–Tartakovsky–Veeravalli 2000):
    at step t with cumulative log-posteriors S_c(t), decide
    c* = argmax_c S_c(t) when  S_{c*}(t) − max_{c ≠ c*} S_c(t) ≥ log((C-1)(1-α)/α).
    Stopping time τ is the first such t; if never, τ = K_max with no decision.
    """
    logps, classes = _oof_logprobs(x, labels, seed, n_splits=n_splits)
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[l] for l in labels])
    return _msprt_from_logps(logps, y, classes, seed,
                             K_max=K_max, alphas=alphas,
                             n_streams_per_class=n_streams_per_class)
