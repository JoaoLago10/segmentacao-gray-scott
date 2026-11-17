"""Etapa 1: baixa as amostras usadas na auditoria (sem baixar os arquivos completos).

Amostras:
- ``final``: último frame (índice 1000) de todas as trajetórias de todas as partições,
  usado para conferir a lista oficial de trajetórias estacionárias.
- ``stride20``: frames 0, 20, ..., 1000 de todas as trajetórias da validação e de
  16 trajetórias do treino (0, 10, ..., 150), para estatísticas e curvas temporais.
- ``early``: frames 0 a 200 (todos) de trajetórias da validação de Gliders e Spirals,
  para localizar o instante de estabilização.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.remote import REGIMES, SPLITS  # noqa: E402
from src.data.sampling import load_or_fetch  # noqa: E402

N_TRAJ = {"train": 160, "valid": 20, "test": 20}
STRIDE20 = list(range(0, 1001, 20))
TRAIN_TRAJS = list(range(0, 160, 10))


def main() -> None:
    t0 = time.time()
    for split in SPLITS:
        for regime in REGIMES:
            load_or_fetch(split, regime, list(range(N_TRAJ[split])), [1000], "final")
            print(f"final {split} {regime} {time.time() - t0:.0f}s", flush=True)
    for regime in REGIMES:
        load_or_fetch("valid", regime, list(range(20)), STRIDE20, "stride20")
        print(f"stride20 valid {regime} {time.time() - t0:.0f}s", flush=True)
        load_or_fetch("train", regime, TRAIN_TRAJS, STRIDE20, "stride20")
        print(f"stride20 train {regime} {time.time() - t0:.0f}s", flush=True)
    for regime in ("gliders", "spirals"):
        load_or_fetch("valid", regime, list(range(20)), list(range(201)), "early")
        print(f"early valid {regime} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
