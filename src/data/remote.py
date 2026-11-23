"""Leitura parcial dos HDF5 do Gray-Scott hospedados no Hugging Face.

Os campos ``t0_fields/A`` e ``t0_fields/B`` são armazenados de forma
contígua, sem compressão, com forma (trajetória, tempo, x, y) em float32.
Isso permite baixar apenas os frames necessários por requisições HTTP de
intervalo de bytes, sem transferir os arquivos completos (21 GB no treino).
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

import fsspec
import h5py
import numpy as np
import requests

REPO_URL = "https://huggingface.co/datasets/polymathic-ai/gray_scott_reaction_diffusion/resolve/main"

REGIMES = {
    "gliders": (0.014, 0.054),
    "bubbles": (0.098, 0.057),
    "maze": (0.029, 0.057),
    "worms": (0.058, 0.065),
    "spirals": (0.018, 0.051),
    "spots": (0.03, 0.062),
}
SPLITS = ("train", "valid", "test")
FIELDS = ("A", "B")


def file_name(regime: str) -> str:
    f, k = REGIMES[regime]
    return f"gray_scott_reaction_diffusion_{regime}_F_{f:g}_k_{k:g}.hdf5"


def file_url(split: str, regime: str) -> str:
    return f"{REPO_URL}/data/{split}/{file_name(regime)}"


@dataclass
class FileLayout:
    split: str
    regime: str
    url: str
    shape: tuple[int, int, int, int]
    dtype: str
    offsets: dict[str, int]
    attrs: dict
    time: list[float]


def read_layout(split: str, regime: str, cache_dir: Path | None = None) -> FileLayout:
    """Lê metadados e deslocamentos em bytes dos campos A e B."""
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"layout_{split}_{regime}.json"
        if cache.exists():
            d = json.loads(cache.read_text())
            d["shape"] = tuple(d["shape"])
            return FileLayout(**d)

    url = file_url(split, regime)
    fs = fsspec.filesystem("https")
    with fs.open(url, "rb", block_size=2**20) as fh, h5py.File(fh, "r") as h:
        offsets = {}
        for name in FIELDS:
            ds = h[f"t0_fields/{name}"]
            if ds.chunks is not None or ds.compression is not None:
                raise RuntimeError(f"{split}/{regime}/{name} não é contíguo")
            offsets[name] = int(ds.id.get_offset())
            shape = tuple(int(s) for s in ds.shape)
            dtype = str(ds.dtype)
        attrs = {
            k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else
                [str(x) for x in v] if hasattr(v, "__len__") and not isinstance(v, str) else v)
            for k, v in h.attrs.items()
        }
        attrs["scalar_F"] = float(h["scalars/F"][()])
        attrs["scalar_k"] = float(h["scalars/k"][()])
        attrs["bc_x"] = h["boundary_conditions/x_periodic"].attrs["bc_type"]
        attrs["bc_y"] = h["boundary_conditions/y_periodic"].attrs["bc_type"]
        attrs["x_first_last"] = [float(h["dimensions/x"][0]), float(h["dimensions/x"][-1])]
        time = [float(t) for t in h["dimensions/time"][:]]

    layout = FileLayout(split, regime, url, shape, dtype, offsets, attrs, time)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(asdict(layout)))
    return layout


_SIGNED: dict[str, str] = {}
_LOCK = threading.Lock()
_SESSION = requests.Session()
_SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=64))


def _auth_headers() -> dict:
    tok = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _signed_url(url: str, refresh: bool = False) -> str:
    """Resolve uma vez o redirecionamento do Hugging Face para a URL assinada da CDN."""
    with _LOCK:
        if refresh or url not in _SIGNED:
            r = _SESSION.get(url, headers={"Range": "bytes=0-0", **_auth_headers()},
                             allow_redirects=False, timeout=60)
            _SIGNED[url] = r.headers.get("location", url)
        return _SIGNED[url]


def _fetch(url: str, start: int, nbytes: int, attempts: int = 10) -> bytes:
    """Lê um intervalo de bytes com espera exponencial (e Retry-After) entre tentativas.

    A URL assinada só é renovada quando expira (403/401); limites de taxa (429) e
    erros temporários (5xx, rede) apenas esperam e tentam de novo.
    """
    headers = {"Range": f"bytes={start}-{start + nbytes - 1}"}
    last = "?"
    for attempt in range(attempts):
        try:
            r = _SESSION.get(_signed_url(url), headers=headers, timeout=120)
            if r.status_code == 206 and len(r.content) == nbytes:
                return r.content
            last = f"HTTP {r.status_code}"
            if r.status_code in (401, 403):
                _signed_url(url, refresh=True)
            wait = float(r.headers.get("Retry-After", 0) or 0)
        except requests.RequestException as e:
            last, wait = type(e).__name__, 0.0
        time.sleep(max(wait, min(60.0, 2.0 ** attempt)) * (0.5 + random.random()))
    raise RuntimeError(f"falha ao ler {url} [{start}, {start + nbytes}) após {attempts} tentativas ({last})")


def _runs(times: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], 0
    for i in range(1, len(times) + 1):
        if i == len(times) or times[i] != times[i - 1] + 1:
            runs.append((int(times[start]), i - start))
            start = i
    return runs


def read_frames_many(layout: FileLayout, field: str, trajs: list[int],
                     times: list[int] | np.ndarray, workers: int = 16) -> np.ndarray:
    """Retorna (len(trajs), len(times), H, W) com os frames pedidos.

    Frames consecutivos viram uma única requisição; as requisições de todas as
    trajetórias são feitas em paralelo.
    """
    _, n_t, h, w = layout.shape
    frame_bytes = h * w * np.dtype(layout.dtype).itemsize
    times = np.asarray(sorted(set(int(t) for t in times)))
    runs = _runs(times)
    jobs = []
    for tr in trajs:
        base = layout.offsets[field] + int(tr) * n_t * frame_bytes
        jobs += [(base + t0 * frame_bytes, n * frame_bytes) for t0, n in runs]
    with ThreadPoolExecutor(workers) as ex:
        chunks = list(ex.map(lambda j: _fetch(layout.url, *j), jobs))
    data = np.frombuffer(b"".join(chunks), dtype=layout.dtype)
    return data.reshape(len(trajs), len(times), h, w).copy()


def read_frames(layout: FileLayout, field: str, traj: int, times: list[int] | np.ndarray,
                workers: int = 32) -> np.ndarray:
    """Retorna (len(times), H, W) com os frames pedidos de uma trajetória."""
    return read_frames_many(layout, field, [traj], times, workers)[0]
