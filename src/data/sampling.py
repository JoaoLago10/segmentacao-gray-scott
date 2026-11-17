"""Amostras locais do Gray-Scott usadas na auditoria e na ficha de ground truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .remote import FIELDS, read_frames_many, read_layout

LAYOUT_CACHE = Path("data/raw/meta/layouts")
SAMPLE_DIR = Path("data/processed/samples")


def sample_path(split: str, regime: str, tag: str) -> Path:
    return SAMPLE_DIR / f"{tag}_{split}_{regime}.npz"


def load_or_fetch(split: str, regime: str, trajs: list[int], times: list[int], tag: str) -> dict:
    """Baixa (ou lê do cache) os frames ``times`` das trajetórias ``trajs``.

    Retorna dict com ``A`` e ``B`` de forma (n_traj, n_t, H, W), ``trajs`` e ``times``.
    """
    path = sample_path(split, regime, tag)
    if path.exists():
        d = np.load(path)
        if list(d["trajs"]) == list(trajs) and list(d["times"]) == list(times):
            return {k: d[k] for k in d.files}
    layout = read_layout(split, regime, LAYOUT_CACHE)
    out = {f: read_frames_many(layout, f, trajs, times) for f in FIELDS}
    out["trajs"] = np.asarray(trajs)
    out["times"] = np.asarray(times)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out
