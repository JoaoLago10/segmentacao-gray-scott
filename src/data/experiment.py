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
    spec = cfg["dados"][name]
    split = spec.get("split", name)
    times = frame_indices(spec["frames"])
    n_traj = 1 if "traj" in spec else len(split_trajs(split, spec["trajs_por_regime"]))
    if out.exists():  # reaproveita só se as trajetórias e os frames coincidirem com a configuração
        with np.load(out) as d:
            if list(d["times"]) == times and len(d["traj"]) == n_traj * len(REGIMES):
                return out
        print(f"  {name}: configuração mudou, baixando de novo", flush=True)
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


# ---------------------------------------------------------------- frames
class FrameDataset(torch.utils.data.Dataset):
    """Frames inteiros (1, H, W) com máscara (H, W).

    Aumento de dados no treino (preserva a física: campo escalar, domínio periódico e
    isotrópico): rotação de 90 graus, reflexão e translação circular aleatórias.
    """

    def __init__(self, A_pre: np.ndarray, M: np.ndarray, train: bool = True):
        self.A, self.M, self.train = A_pre, M, train

    def __len__(self):
        return len(self.A)

    def __getitem__(self, i):
        a, m = self.A[i], self.M[i]
        if self.train:
            # semente derivada do torch: muda a cada época/worker e é reprodutível por torch.manual_seed
            rng = np.random.default_rng((int(torch.randint(1 << 31, (1,))), i))
            k = int(rng.integers(4))
            sh = (int(rng.integers(a.shape[0])), int(rng.integers(a.shape[1])))
            a, m = np.rot90(np.roll(a, sh, (0, 1)), k), np.rot90(np.roll(m, sh, (0, 1)), k)
            if rng.random() < 0.5:
                a, m = a[:, ::-1], m[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(a, np.float32))[None],
                torch.from_numpy(np.ascontiguousarray(m, np.float32)))


@torch.no_grad()
def predict_frames(model, A_pre: np.ndarray, device, batch: int = 32, amp: bool = False) -> np.ndarray:
    """Probabilidades (N, H, W) de P(padrão) por pixel, frame inteiro por vez."""
    out = []
    for s in range(0, len(A_pre), batch):
        x = torch.from_numpy(np.ascontiguousarray(A_pre[s:s + batch], np.float32))[:, None].to(device)
        with torch.autocast(device.type if hasattr(device, "type") else str(device),
                            enabled=amp and str(device).startswith("cuda")):
            p = model.predict_proba(x)
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)
