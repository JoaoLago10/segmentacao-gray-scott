"""Ground truth definitivo do projeto (decisão de 21/11/2026).

Interpretação (C) da ficha da Etapa 2: o "frame médio" define o critério e a
máscara é gerada em cada frame,

    M(x, y, t) = 1[B(x, y, t) > tau],   tau = Otsu global de B no treino (t > 0) = 0,1507.

Entrada da rede: só A (decisão G3). Frames excluídos (G4/G5): t = 0 e frames sem
padrão (max B < 1e-6, trajetórias estacionárias após a morte do padrão).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_CAND = Path(__file__).resolve().parents[2] / "results/gt/candidatos.json"
TAU = float(json.loads(_CAND.read_text())["limiares"]["otsu_B_global"])
DEAD_MAX_B = 1e-6


def gt_mask(B: np.ndarray, tau: float = TAU) -> np.ndarray:
    return (B > tau).astype(np.uint8)


def valid_frames(B: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Máscara booleana (..., n_t) dos frames usados no treino e na avaliação."""
    alive = B.reshape(*B.shape[:-2], -1).max(axis=-1) >= DEAD_MAX_B
    return alive & (np.asarray(times) > 0)
