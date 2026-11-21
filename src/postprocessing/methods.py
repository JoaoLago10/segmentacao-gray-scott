"""Pós-processamentos Q1--Q10 aplicados aos mapas de probabilidade salvos.

Entrada de cada método: ``P`` (T, H, W) com as probabilidades da classe "padrão" de
**uma trajetória**, em ordem temporal, e ``A`` (T, H, W) bruto (usado só pelo CRF).
Saída: máscaras binárias (T, H, W) uint8.

- Q0 ("sem pós") = limiar fixo 0,5 (decisão C2).
- Os parâmetros de cada Qi são escolhidos na **validação** dentro de ``GRIDS``.
- Todas as operações espaciais respeitam o contorno periódico.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import ndimage
from skimage.filters import sobel
from skimage.segmentation import watershed

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
    return _bin(P, thr)


def q2_abertura(P, A, fp):
    return _per_frame(_open, _bin(P), fp)


def q3_fechamento(P, A, fp):
    return _per_frame(_close, _bin(P), fp)


def q4_abertura_fechamento(P, A, fp):
    return _per_frame(lambda m: _close(_open(m, fp), fp), _bin(P))


def q5_remover_pequenos(P, A, area_min):
    return _per_frame(lambda m: _component_filter(m, lambda l, n, a: a >= area_min), _bin(P))


def q6_preencher_buracos(P, A, area_max):
    """No toro não há "exterior": buraco = componente do fundo com área <= area_max."""
    def fill(m):
        holes = _component_filter(1 - m, lambda l, n, a: a <= area_max)
        return m | holes
    return _per_frame(fill, _bin(P))


def q7_componentes(P, A, p_min):
    """Mantém os componentes conexos cuja probabilidade média é >= p_min.

    O critério "maior componente" não serve aqui (Spots tem ~200 componentes).
    """
    def sel(m, p):
        def keep(lab, n, areas):
            s = np.bincount(lab.ravel(), weights=p.ravel(), minlength=n + 1)
            return s / np.maximum(areas, 1) >= p_min
        return _component_filter(m, keep)
    M = _bin(P)
    return np.stack([sel(m, p) for m, p in zip(M, P)]).astype(np.uint8)


def q8_watershed(P, A, alto, baixo, pad=16):
    """Watershed controlado por marcadores (padrão: P > alto; fundo: P < baixo) sobre |∇P|.

    O contorno periódico é tratado por preenchimento circular de ``pad`` pixels.
    """
    out = []
    for p in P:
        pp = np.pad(p, pad, mode="wrap")
        markers = np.zeros(pp.shape, np.int32)
        markers[pp < baixo] = 1
        markers[pp > alto] = 2
        lab = watershed(sobel(pp), markers)
        out.append((lab[pad:-pad, pad:-pad] == 2))
    return np.stack(out).astype(np.uint8)


def q9_crf(P, A, w_apar, theta_beta, w_suave=1.0, theta_alpha=3.0, theta_gamma=1.0,
           raio=3, iters=5):
    """CRF com inferência por campo médio (Krähenbühl e Koltun, 2011), modelo de Potts.

    Os kernels gaussianos (aparência: posição + intensidade de A; suavidade: posição) são
    truncados numa janela (2*raio+1)^2 periódica, em vez de totalmente conectados.
    """
    eps = 1e-6
    U1 = np.log(np.clip(P, eps, 1 - eps))
    U0 = np.log(np.clip(1 - P, eps, 1 - eps))
    Q1 = P.astype(np.float32).copy()
    offs = [(dy, dx) for dy in range(-raio, raio + 1) for dx in range(-raio, raio + 1)
            if (dy, dx) != (0, 0)]
    kernels = []
    for dy, dx in offs:
        d2 = dy * dy + dx * dx
        Ash = np.roll(A, (dy, dx), axis=(1, 2))
        k = (w_apar * np.exp(-d2 / (2 * theta_alpha ** 2)) * np.exp(-(A - Ash) ** 2 / (2 * theta_beta ** 2))
             + w_suave * np.exp(-d2 / (2 * theta_gamma ** 2)))
        kernels.append(k.astype(np.float32))
    for _ in range(iters):
        m1 = np.zeros_like(Q1)
        m0 = np.zeros_like(Q1)
        for (dy, dx), k in zip(offs, kernels):
            q = np.roll(Q1, (dy, dx), axis=(1, 2))
            m1 += k * q
            m0 += k * (1 - q)
        l1, l0 = U1 + m1, U0 + m0
        Q1 = 1.0 / (1.0 + np.exp(np.clip(l0 - l1, -50, 50)))
    return _bin(Q1)


def q10_temporal(P, A, janela):
    """Média móvel centrada das probabilidades ao longo do tempo, seguida do limiar 0,5."""
    Ps = ndimage.uniform_filter1d(P.astype(np.float32), size=janela, axis=0, mode="nearest")
    return _bin(Ps)


METHODS = {
    "Q0": ("Nenhum (limiar fixo 0,5)", q0),
    "Q1": ("Limiarização de probabilidades", q1_limiar),
    "Q2": ("Abertura morfológica", q2_abertura),
    "Q3": ("Fechamento morfológico", q3_fechamento),
    "Q4": ("Abertura + fechamento", q4_abertura_fechamento),
    "Q5": ("Remoção de componentes pequenos", q5_remover_pequenos),
    "Q6": ("Preenchimento de buracos", q6_preencher_buracos),
    "Q7": ("Seleção de componentes conexos", q7_componentes),
    "Q8": ("Watershed", q8_watershed),
    "Q9": ("CRF (campo médio)", q9_crf),
    "Q10": ("Filtragem temporal", q10_temporal),
}

GRIDS = {
    "Q0": {},
    "Q1": {"thr": [round(t, 2) for t in np.arange(0.05, 0.951, 0.05)]},
    "Q2": {"fp": ["cruz3", "quadrado3"]},
    "Q3": {"fp": ["cruz3", "quadrado3"]},
    "Q4": {"fp": ["cruz3", "quadrado3"]},
    "Q5": {"area_min": [2, 4, 8, 16, 32]},
    "Q6": {"area_max": [2, 4, 8, 16, 32]},
    "Q7": {"p_min": [0.6, 0.7, 0.8, 0.9]},
    "Q8": {"alto": [0.7, 0.8, 0.9], "baixo": [0.1, 0.2, 0.3]},
    "Q9": {"w_apar": [1.0, 3.0], "theta_beta": [0.05, 0.1]},
    "Q10": {"janela": [3, 5]},
}

POS_IDS = list(METHODS)


def grid(qid: str) -> list[dict]:
    g = GRIDS[qid]
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]


def apply(qid: str, P: np.ndarray, A: np.ndarray, **params) -> np.ndarray:
    return METHODS[qid][1](P, A, **params)
