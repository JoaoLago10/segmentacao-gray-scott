"""Pós-processamentos Q1--Q5 (PROTOCOLO.MD, Seção 8) aplicados às probabilidades salvas.

Entrada de cada método: ``P`` (T, H, W) com as probabilidades da classe "padrão" de
**uma trajetória**, em ordem temporal, e ``A`` (T, H, W) bruto (não usado; mantido na interface).
Saída: máscaras binárias (T, H, W) uint8.

- Q0 ("sem pós") = limiar fixo 0,5 (decisão C2).
- Os parâmetros de cada Qi são escolhidos na **validação** dentro de ``GRIDS``.
- Todas as operações espaciais respeitam o contorno periódico.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import ndimage

from ..ground_truth.candidates import periodic_label

FOOTPRINTS = {
    "cruz3": ndimage.generate_binary_structure(2, 1),
    "quadrado3": np.ones((3, 3), bool),
}


def _bin(P, thr=0.5):
    return (P > thr).astype(np.uint8)


def _per_frame(fn, M, *args):
    return np.stack([fn(m, *args) for m in M]).astype(np.uint8)


def _open(m, fp):
    return ndimage.grey_opening(m, footprint=FOOTPRINTS[fp], mode="wrap")


def _close(m, fp):
    return ndimage.grey_closing(m, footprint=FOOTPRINTS[fp], mode="wrap")


def _component_filter(m, keep):
    """Aplica ``keep(lab, n, areas) -> bool[n+1]`` aos componentes periódicos de ``m``."""
    lab, n = periodic_label(m)
    if n == 0:
        return m
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    k = keep(lab, n, areas)
    k[0] = False
    return k[lab].astype(np.uint8)


# ------------------------------------------------------------------ métodos
def q0(P, A, **_):
    return _bin(P)


def q1_limiar(P, A, thr):
    """Threshold optimization: limiar escolhido na validação."""
    return _bin(P, thr)


def q2_abertura(P, A, fp):
    return _per_frame(_open, _bin(P), fp)


def q3_fechamento(P, A, fp):
    return _per_frame(_close, _bin(P), fp)


def q4_componentes(P, A, area_min):
    """Connected-component filtering: remove componentes periódicos com área < area_min."""
    return _per_frame(lambda m: _component_filter(m, lambda l, n, a: a >= area_min), _bin(P))


def q5_temporal(P, A, janela):
    """Temporal consistency: média móvel centrada das probabilidades no tempo, seguida de 0,5."""
    Ps = ndimage.uniform_filter1d(P.astype(np.float32), size=janela, axis=0, mode="nearest")
    return _bin(Ps)


METHODS = {
    "Q0": ("Baseline (limiar fixo 0,5)", q0),
    "Q1": ("Threshold optimization", q1_limiar),
    "Q2": ("Morphological opening", q2_abertura),
    "Q3": ("Morphological closing", q3_fechamento),
    "Q4": ("Connected-component filtering", q4_componentes),
    "Q5": ("Temporal consistency filtering", q5_temporal),
}

GRIDS = {
    "Q0": {},
    "Q1": {"thr": [round(t, 2) for t in np.arange(0.05, 0.951, 0.05)]},
    "Q2": {"fp": ["cruz3", "quadrado3"]},
    "Q3": {"fp": ["cruz3", "quadrado3"]},
    "Q4": {"area_min": [2, 4, 8, 16, 32]},
    "Q5": {"janela": [3, 5]},
}

POS_IDS = list(METHODS)


def grid(qid: str) -> list[dict]:
    g = GRIDS[qid]
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]


def apply(qid: str, P: np.ndarray, A: np.ndarray, **params) -> np.ndarray:
    return METHODS[qid][1](P, A, **params)
