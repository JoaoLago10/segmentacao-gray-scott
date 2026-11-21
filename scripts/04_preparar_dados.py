"""Baixa os subconjuntos do experimento (configs/experimento.yaml) por leitura parcial dos HDF5.

Gera data/processed/exp/{train,valid,test,visual}.npz (~1,1 GB no total) e um resumo
com as frações de frames válidos e de pixels de padrão por regime.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.experiment import ROOT, fetch_split, load_config, load_split  # noqa: E402
from src.ground_truth.generation import TAU, gt_mask, valid_frames  # noqa: E402


def main():
    cfg = load_config()
    resumo = {"tau": TAU}
    for name in ("train", "valid", "test", "visual"):
        print(f"{name}:", flush=True)
        fetch_split(cfg, name)
        d = load_split(cfg, name)
        ok = valid_frames(d["B"], d["times"])
        M = gt_mask(d["B"])
        por_regime = {}
        for r in dict.fromkeys(d["regime"]):
            i = d["regime"] == r
            por_regime[str(r)] = {
                "trajetorias": [int(t) for t in d["traj"][i]],
                "frac_frames_validos": float(ok[i].mean()),
                "frac_pixels_padrao": float(M[i][ok[i]].mean()) if ok[i].any() else None,
            }
        resumo[name] = {"frames": [int(t) for t in d["times"]], "regimes": por_regime,
                        "forma_A": list(d["A"].shape)}
    out = ROOT / "results/exp/dados.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(resumo, indent=1, ensure_ascii=False))
    print(json.dumps({k: v["regimes"] if isinstance(v, dict) else v for k, v in resumo.items()},
                     indent=1)[:3000])


if __name__ == "__main__":
    main()
