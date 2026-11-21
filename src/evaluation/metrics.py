"""Métricas de segmentação, temporais e intervalos de confiança.

Unidade estatística = trajetória (trajetórias são independentes; frames da mesma
trajetória são correlacionados). TP, FP e FN são somados nos frames de cada trajetória
(convenção G4, opção 3), e o IC de 95% é obtido por bootstrap sobre as trajetórias.
"""

from __future__ import annotations

import numpy as np


def confusion(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(TP, FP, FN, TN) somados sobre todos os eixos exceto o primeiro (frames)."""
    p, g = pred.astype(bool), gt.astype(bool)
    ax = tuple(range(1, p.ndim))
    tp = (p & g).sum(ax)
    fp = (p & ~g).sum(ax)
    fn = (~p & g).sum(ax)
    tn = (~p & ~g).sum(ax)
    return np.stack([tp, fp, fn, tn], -1).astype(np.int64)


def scores(c: np.ndarray) -> dict:
    """IoU, Dice, Precision e Recall a partir de (..., 4). Vazio/vazio = 1 (G4)."""
    tp, fp, fn = (c[..., i].astype(np.float64) for i in range(3))

    def div(a, b):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(b > 0, a / np.where(b > 0, b, 1), 1.0)

    return {"IoU": div(tp, tp + fp + fn), "Dice": div(2 * tp, 2 * tp + fp + fn),
            "Precision": div(tp, tp + fp), "Recall": div(tp, tp + fn)}


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Média e semi-amplitude do IC (percentil) de 95% por reamostragem com reposição."""
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float((hi - lo) / 2)


# ------------------------------------------------------------------ temporais
def _periodic_centroid(m: np.ndarray) -> np.ndarray | None:
    """Centroide circular (Bai e Breen, 2008) de uma máscara periódica, em pixels."""
    if m.sum() == 0:
        return None
    H, W = m.shape
    ys, xs = np.nonzero(m)
    out = []
    for v, n in ((ys, H), (xs, W)):
        th = 2 * np.pi * v / n
        out.append((np.arctan2(-np.sin(th).mean(), -np.cos(th).mean()) + np.pi) * n / (2 * np.pi))
    return np.array(out)


def _periodic_dist(a, b, shape):
    d = np.abs(a - b)
    d = np.minimum(d, np.asarray(shape) - d)
    return float(np.hypot(*d))


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def temporal(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Métricas temporais de uma trajetória (frames em ordem). Erros em relação ao GT:

    - ErroArea: média de |Δárea_pred - Δárea_GT| entre frames consecutivos (fração do domínio).
    - ErroCentroide: média da distância periódica entre os centroides predito e do GT (px).
    - ErroConsist: média de |IoU(M̂_t, M̂_t+1) - IoU(M_t, M_t+1)| (consistência temporal).
    """
    ap, ag = pred.reshape(len(pred), -1).mean(1), gt.reshape(len(gt), -1).mean(1)
    e_area = float(np.abs(np.diff(ap) - np.diff(ag)).mean()) if len(pred) > 1 else np.nan
    dists = []
    for p, g in zip(pred, gt):
        cp, cg = _periodic_centroid(p), _periodic_centroid(g)
        if cp is not None and cg is not None:
            dists.append(_periodic_dist(cp, cg, p.shape))
    e_cent = float(np.mean(dists)) if dists else np.nan
    if len(pred) > 1:
        tp = [_iou(pred[t], pred[t + 1]) for t in range(len(pred) - 1)]
        tg = [_iou(gt[t], gt[t + 1]) for t in range(len(gt) - 1)]
        e_cons = float(np.mean(np.abs(np.array(tp) - np.array(tg))))
    else:
        e_cons = np.nan
    return {"ErroArea": e_area, "ErroCentroide": e_cent, "ErroConsist": e_cons}
