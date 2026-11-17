"""Candidatos de ground truth para o Gray-Scott (Seção 0.4 e 0.9.4).

Nenhuma definição aqui é a oficial do projeto: a escolha depende da equipe e do
professor. Todas as operações espaciais respeitam o contorno periódico.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu


def otsu_threshold(values: np.ndarray, bins: int = 1024) -> float:
    """Limiar de Otsu sobre um conjunto de valores (ajustado só no treino)."""
    hist, edges = np.histogram(values.ravel(), bins=bins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return float(threshold_otsu(hist=(hist, centers)))


def binary_mask(B: np.ndarray, tau: float) -> np.ndarray:
    """Alternativa 1: padrão (1) onde B > tau, fundo (0) caso contrário."""
    return (B > tau).astype(np.uint8)


def trivial_state_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distância ao estado homogêneo trivial (A, B) = (1, 0), que é o fundo do Gray-Scott."""
    return np.sqrt((1.0 - A) ** 2 + B ** 2)


def periodic_dilate(mask: np.ndarray, size: int = 3) -> np.ndarray:
    return ndimage.maximum_filter(mask, size=(1,) * (mask.ndim - 2) + (size, size), mode="wrap")


def periodic_erode(mask: np.ndarray, size: int = 3) -> np.ndarray:
    return ndimage.minimum_filter(mask, size=(1,) * (mask.ndim - 2) + (size, size), mode="wrap")


def three_class_mask(B: np.ndarray, tau: float, width: int = 3) -> np.ndarray:
    """Alternativa 2: 0 = fundo, 1 = interior do padrão, 2 = interface.

    A interface é o gradiente morfológico (dilatação menos erosão) da máscara
    binária, com elemento estruturante quadrado ``width`` x ``width`` e wrap.
    """
    m = binary_mask(B, tau)
    interface = periodic_dilate(m, width) != periodic_erode(m, width)
    out = m.copy()
    out[interface] = 2
    return out


def periodic_gradient_magnitude(B: np.ndarray, dx: float = 2.0 / 127) -> np.ndarray:
    gx = (np.roll(B, -1, -2) - np.roll(B, 1, -2)) / (2 * dx)
    gy = (np.roll(B, -1, -1) - np.roll(B, 1, -1)) / (2 * dx)
    return np.sqrt(gx ** 2 + gy ** 2)


def periodic_label(mask: np.ndarray, connectivity: int = 1) -> tuple[np.ndarray, int]:
    """Alternativa 3: rotulação de componentes conexos num toro (2D).

    Rotula normalmente e depois une rótulos que se tocam através das bordas
    opostas, usando union-find.
    """
    structure = ndimage.generate_binary_structure(2, connectivity)
    lab, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return lab, 0
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pairs = [(lab[0, :], lab[-1, :]), (lab[:, 0], lab[:, -1])]
    if connectivity == 2:
        pairs += [(lab[0, :], np.roll(lab[-1, :], 1)), (lab[0, :], np.roll(lab[-1, :], -1)),
                  (lab[:, 0], np.roll(lab[:, -1], 1)), (lab[:, 0], np.roll(lab[:, -1], -1))]
    for a, b in pairs:
        for u, v in zip(a[(a > 0) & (b > 0)], b[(a > 0) & (b > 0)]):
            union(int(u), int(v))
    roots = np.array([find(i) for i in range(n + 1)])
    uniq, remap = np.unique(roots, return_inverse=True)
    return remap.reshape(-1)[lab], len(uniq) - 1


def temporal_mean_mask(B: np.ndarray, tau: float, t_axis: int = 0) -> np.ndarray:
    """Interpretação "imagem média de vários instantes": limiariza a média temporal de B."""
    return binary_mask(B.mean(axis=t_axis), tau)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a, b).sum()
    return float("nan") if union == 0 else float(np.logical_and(a, b).sum() / union)
