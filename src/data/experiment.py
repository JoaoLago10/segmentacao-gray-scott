"""Conjuntos do experimento e conversão frame -> nuvem de pontos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from .remote import REGIMES, read_frames_many, read_layout
from .sampling import LAYOUT_CACHE

ROOT = Path(__file__).resolve().parents[2]
N_TRAJ = {"train": 160, "valid": 20, "test": 20}


def load_config(path: str | Path = ROOT / "configs/experimento.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def frame_indices(spec) -> list[int]:
    if isinstance(spec, dict):
        return list(range(spec["inicio"], spec["fim"] + 1, spec["passo"]))
    return list(spec)


def split_trajs(split: str, k: int) -> list[int]:
    return sorted(set(np.linspace(0, N_TRAJ[split] - 1, k).round().astype(int).tolist()))


def split_path(cfg: dict, name: str) -> Path:
    return ROOT / cfg["dados"]["dir"] / f"{name}.npz"


def fetch_split(cfg: dict, name: str) -> Path:
    """Baixa (leitura parcial do Hugging Face) e salva ``{name}.npz`` com todos os regimes."""
    out = split_path(cfg, name)
    if out.exists():
        return out
    spec = cfg["dados"][name]
    split = spec.get("split", name)
    times = frame_indices(spec["frames"])
    parts = {"A": [], "B": [], "regime": [], "traj": []}
    for r in REGIMES:
        trajs = [spec["traj"]] if "traj" in spec else split_trajs(split, spec["trajs_por_regime"])
        lay = read_layout(split, r, ROOT / LAYOUT_CACHE)
        for f in ("A", "B"):
            parts[f].append(read_frames_many(lay, f, trajs, times))
        parts["regime"] += [r] * len(trajs)
        parts["traj"] += trajs
        print(f"  {name}/{r}: {len(trajs)} trajetórias x {len(times)} frames", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, A=np.concatenate(parts["A"]), B=np.concatenate(parts["B"]),
             regime=np.array(parts["regime"]), traj=np.array(parts["traj"]),
             times=np.array(times))
    return out


def load_split(cfg: dict, name: str) -> dict:
    d = np.load(split_path(cfg, name))
    return {k: d[k] for k in d.files}


# ------------------------------------------------------------ nuvem de pontos
def block_coords(size: int) -> np.ndarray:
    """Coordenadas (size*size, 2) normalizadas em [-1, 1] dos pixels de um bloco."""
    c = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], -1)


def periodic_crop(img: np.ndarray, i0: int, j0: int, size: int) -> np.ndarray:
    H, W = img.shape[-2:]
    rows = (np.arange(size) + i0) % H
    cols = (np.arange(size) + j0) % W
    return img[..., rows[:, None], cols[None, :]]


class BlockDataset(torch.utils.data.Dataset):
    """Um bloco periódico aleatório por frame e por época (aumento de dados incluído)."""

    def __init__(self, A_pre: np.ndarray, M: np.ndarray, size: int, train: bool = True):
        self.A, self.M, self.size, self.train = A_pre, M, size, train
        self.xy = block_coords(size)

    def __len__(self):
        return len(self.A)

    def __getitem__(self, i):
        # Semente derivada do gerador do torch: muda a cada época e a cada worker, e é
        # reprodutível a partir de torch.manual_seed.
        rng = (np.random.default_rng((int(torch.randint(1 << 31, (1,))), i))
               if self.train else None)
        H, W = self.A.shape[-2:]
        i0, j0 = (rng.integers(0, H), rng.integers(0, W)) if self.train else (0, 0)
        a = periodic_crop(self.A[i], i0, j0, self.size)
        m = periodic_crop(self.M[i], i0, j0, self.size)
        if self.train:
            k = int(rng.integers(4))
            a, m = np.rot90(a, k), np.rot90(m, k)
            if rng.random() < 0.5:
                a, m = a[:, ::-1], m[:, ::-1]
        pts = np.concatenate([self.xy, a.reshape(-1, 1)], 1).astype(np.float32)
        lab = m.reshape(-1).astype(np.float32)
        if self.train:
            perm = rng.permutation(len(lab))
            pts, lab = pts[perm], lab[perm]
        return torch.from_numpy(pts), torch.from_numpy(lab)


@torch.no_grad()
def predict_frames(model, A_pre: np.ndarray, size: int, stride: int, device, batch: int = 32) -> np.ndarray:
    """Probabilidades (N, H, W) por média de blocos periódicos sobrepostos."""
    N, H, W = A_pre.shape
    offs = [(i, j) for i in range(0, H, stride) for j in range(0, W, stride)]
    xy = torch.from_numpy(block_coords(size)).to(device)
    acc = np.zeros((N, H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    for i0, j0 in offs:
        rows = (np.arange(size) + i0) % H
        cols = (np.arange(size) + j0) % W
        cnt[np.ix_(rows, cols)] += 1
    jobs = [(n, i0, j0) for n in range(N) for i0, j0 in offs]
    for s in range(0, len(jobs), batch):
        chunk = jobs[s:s + batch]
        a = np.stack([periodic_crop(A_pre[n], i0, j0, size) for n, i0, j0 in chunk])
        a = torch.from_numpy(a.reshape(len(chunk), -1, 1).astype(np.float32)).to(device)
        pts = torch.cat([xy.expand(len(chunk), -1, -1), a], -1)
        p = torch.sigmoid(model(pts)).float().cpu().numpy().reshape(len(chunk), size, size)
        for (n, i0, j0), pb in zip(chunk, p):
            rows = (np.arange(size) + i0) % H
            cols = (np.arange(size) + j0) % W
            acc[n][np.ix_(rows, cols)] += pb
    return acc / cnt
