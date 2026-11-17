"""Etapa 1: estatísticas e figuras da auditoria a partir das amostras locais.

Saídas em ``results/audit/`` (JSON) e ``docs/figs/audit/`` (PNG).
Nenhuma máscara ou limiar de ground truth é definido aqui.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.remote import REGIMES, SPLITS  # noqa: E402
from src.data.sampling import sample_path  # noqa: E402

OUT = ROOT / "results/audit"
FIG = ROOT / "docs/figs/audit"
DT = 10.0
STATIONARY_EPS = 1e-4


def load(split, regime, tag):
    d = np.load(ROOT / sample_path(split, regime, tag))
    return {k: d[k] for k in d.files}


def official_stationary() -> dict:
    """Extrai a lista oficial de trajetórias estacionárias do README do dataset."""
    text = (ROOT / "data/raw/meta/README.md").read_text()
    text = text[text.index("**Warning:**"):]
    split_map = {"Validation set": "valid", "Training set": "train", "Test set": "test"}
    out, split, regime = {}, None, None
    for line in text.splitlines():
        for k, v in split_map.items():
            if k in line:
                split = v
        m = re.search(r"f\s*=\s*([\d.]+)\s*,\s*k\s*=\s*([\d.]+)", line)
        if m:
            fk = (float(m.group(1)), float(m.group(2)))
            regime = next(r for r, p in REGIMES.items() if np.allclose(p, fk))
        m = re.search(r"Trajectory (\d+), time = (\d+)", line)
        if m:
            out.setdefault(split, {}).setdefault(regime, {})[int(m.group(1))] = int(m.group(2))
    return out


def spatial_std(x):
    return x.reshape(*x.shape[:-2], -1).std(-1)


def audit_stationary(report):
    official = official_stationary()
    found = {}
    finals = {}
    for split in SPLITS:
        for regime in REGIMES:
            d = load(split, regime, "final")
            sA = spatial_std(d["A"][:, 0])
            sB = spatial_std(d["B"][:, 0])
            finals[(split, regime)] = (sA, sB)
            idx = [int(t) for t in d["trajs"][sA < STATIONARY_EPS]]
            if idx:
                found.setdefault(split, {})[regime] = idx
    comparison = {}
    for split in SPLITS:
        for regime in REGIMES:
            off = sorted(official.get(split, {}).get(regime, {}).keys())
            fnd = found.get(split, {}).get(regime, [])
            if off or fnd:
                comparison[f"{split}/{regime}"] = {
                    "oficial": off, "detectado": fnd,
                    "so_oficial": sorted(set(off) - set(fnd)),
                    "so_detectado": sorted(set(fnd) - set(off)),
                }
    # Distribuição do desvio espacial final de A (separação entre estacionárias e não estacionárias)
    gl = finals[("train", "gliders")][0]
    report["estacionarias"] = {
        "criterio": f"desvio padrão espacial de A no frame 1000 < {STATIONARY_EPS}",
        "comparacao": comparison,
        "contagem_detectada": {s: {r: len(v) for r, v in found.get(s, {}).items()} for s in SPLITS},
        "contagem_oficial": {s: {r: len(v) for r, v in official.get(s, {}).items()} for s in SPLITS},
        "std_A_final_gliders_train_quantis": np.quantile(gl, [0, .25, .5, .75, 1]).tolist(),
        "valores_A_B_estacionarios": {},
    }
    for split in SPLITS:
        for regime, trajs in found.get(split, {}).items():
            d = load(split, regime, "final")
            sel = np.isin(d["trajs"], trajs)
            report["estacionarias"]["valores_A_B_estacionarios"][f"{split}/{regime}"] = {
                "A_medio": float(d["A"][sel].mean()), "B_medio": float(d["B"][sel].mean()),
                "B_max": float(d["B"][sel].max())}

    # Instante de estabilização (validação, frames 0..200)
    times_table = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, regime in zip(axes, ("gliders", "spirals")):
        d = load("valid", regime, "early")
        sA = spatial_std(d["A"])
        off = official.get("valid", {}).get(regime, {})
        frozen = np.abs(np.diff(d["A"], axis=1)).reshape(*sA.shape[:1], sA.shape[1] - 1, -1).max(-1) == 0
        maxB = np.abs(d["B"]).reshape(*sA.shape, -1).max(-1)
        for i, tr in enumerate(d["trajs"]):
            below = np.where(sA[i] < STATIONARY_EPS)[0]
            first = int(below[0]) if len(below) and np.all(sA[i, below[0]:] < STATIONARY_EPS) else None
            fz = np.where(frozen[i])[0]
            vb = np.where(maxB[i] < 1e-6)[0]
            if first is not None or int(tr) in off:
                times_table[f"valid/{regime}/traj{int(tr)}"] = {
                    "oficial": off.get(int(tr)),
                    "indice_A_congelado_float32": int(fz[0]) + 1 if len(fz) else None,
                    "indice_std_A_menor_eps": first,
                    "indice_max_B_menor_1e-6": int(vb[0]) if len(vb) else None}
            ax.semilogy(d["times"], np.maximum(sA[i], 1e-12),
                        color="C3" if int(tr) in off else "C0", lw=0.8, alpha=0.8)
        ax.axhline(STATIONARY_EPS, color="k", ls="--", lw=0.8)
        ax.set_title(f"{regime} (validação): vermelho = lista oficial")
        ax.set_xlabel("índice do frame (t = 10 x índice)")
        ax.set_ylabel("desvio padrão espacial de A")
    fig.tight_layout()
    fig.savefig(FIG / "estacionarias_std_A.png", dpi=130)
    plt.close(fig)
    report["estacionarias"]["instante_estabilizacao_validacao"] = times_table
    return found


def audit_values(report, stationary):
    stats = {}
    for split in ("train", "valid"):
        for regime in REGIMES:
            d = load(split, regime, "stride20")
            e = {}
            for f in ("A", "B"):
                x = d[f]
                e[f] = {"min": float(x.min()), "max": float(x.max()), "media": float(x.mean()),
                        "desvio": float(x.std()), "frac_zero": float((x == 0).mean()),
                        "frac_negativo": float((x < 0).mean()),
                        "frac_acima_1": float((x > 1).mean()),
                        "p01": float(np.quantile(x, 0.01)), "p99": float(np.quantile(x, 0.99)),
                        "nan_inf": int((~np.isfinite(x)).sum())}
            e["n_trajetorias_amostradas"] = int(len(d["trajs"]))
            e["n_frames_por_trajetoria"] = int(len(d["times"]))
            stats[f"{split}/{regime}"] = e
    report["valores"] = stats

    fig, axes = plt.subplots(2, 6, figsize=(20, 6))
    for j, regime in enumerate(REGIMES):
        d = load("train", regime, "stride20")
        for i, f in enumerate(("A", "B")):
            x = d[f][:, 1:].ravel()
            axes[i, j].hist(x, bins=200, color="C0" if f == "A" else "C1", log=True)
            axes[i, j].set_title(f"{regime}: {f} (treino, t > 0)")
    fig.tight_layout()
    fig.savefig(FIG / "histogramas_A_B_treino.png", dpi=110)
    plt.close(fig)


def audit_temporal(report, stationary):
    curves = {}
    fig, axes = plt.subplots(3, 6, figsize=(22, 9), sharex=True)
    for j, regime in enumerate(REGIMES):
        d = load("valid", regime, "stride20")
        t = d["times"] * DT
        stat = set(stationary.get("valid", {}).get(regime, []))
        keep = np.array([int(tr) not in stat for tr in d["trajs"]])
        B = d["B"][keep]
        sB = spatial_std(B)
        mB = B.reshape(*B.shape[:2], -1).mean(-1)
        dB = np.sqrt(((B[:, 1:] - B[:, :-1]) ** 2).reshape(B.shape[0], B.shape[1] - 1, -1).mean(-1))
        for ax, y, tt, lab in ((axes[0, j], mB, t, "média espacial de B"),
                               (axes[1, j], sB, t, "desvio espacial de B"),
                               (axes[2, j], dB, t[1:], "RMS(B_t - B_{t-200s})")):
            q = np.quantile(y, [0.1, 0.5, 0.9], axis=0)
            ax.fill_between(tt, q[0], q[2], alpha=0.3)
            ax.plot(tt, q[1])
            ax.set_ylabel(lab)
        axes[2, j].set_yscale("log")
        axes[0, j].set_title(f"{regime} (validação, {keep.sum()} traj. não estacionárias)")
        axes[2, j].set_xlabel("t (s)")
        med_d = np.median(dB, axis=0)
        curves[regime] = {
            "t": t.tolist(), "mediana_media_B": np.median(mB, 0).tolist(),
            "mediana_desvio_B": np.median(sB, 0).tolist(),
            "mediana_rms_variacao_B": med_d.tolist(),
            "t_variacao_abaixo_10pct_do_pico": float(t[1:][np.argmax(med_d < 0.1 * med_d.max())]),
        }
    fig.tight_layout()
    fig.savefig(FIG / "curvas_temporais_B_validacao.png", dpi=110)
    plt.close(fig)
    report["temporal"] = curves


def visual_table(report, stationary, idxs=(0, 20, 40, 100, 500, 1000)):
    """Rascunho da tabela visual (Seção 0.9.5): uma trajetória não estacionária por regime."""
    chosen = {}
    fig, axes = plt.subplots(2 * len(REGIMES), len(idxs), figsize=(2.1 * len(idxs), 2.2 * 2 * len(REGIMES)))
    for i, regime in enumerate(REGIMES):
        d = load("valid", regime, "stride20")
        stat = set(stationary.get("valid", {}).get(regime, []))
        tr_pos = next(p for p, tr in enumerate(d["trajs"]) if int(tr) not in stat)
        chosen[regime] = int(d["trajs"][tr_pos])
        times = list(d["times"])
        for j, idx in enumerate(idxs):
            k = times.index(idx)
            for r, f in enumerate(("A", "B")):
                ax = axes[2 * i + r, j]
                ax.imshow(d[f][tr_pos, k].T, origin="lower", cmap="viridis" if f == "A" else "magma",
                          vmin=0, vmax=1 if f == "A" else None)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0 and r == 0:
                    ax.set_title(f"t = {idx * DT:.0f} s")
                if j == 0:
                    ax.set_ylabel(f"{regime}\n{f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "tabela_visual_rascunho.png", dpi=110)
    plt.close(fig)
    report["tabela_visual"] = {"split": "valid", "trajetoria_por_regime": chosen,
                               "indices": list(idxs), "tempos_s": [i * DT for i in idxs]}


def initial_conditions(report):
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
    d = load("valid", "maze", "stride20")
    for j in range(8):
        tr = j * 2 + (j % 2)
        for r, f in enumerate(("A", "B")):
            axes[r, j].imshow(d[f][tr, 0].T, origin="lower")
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            axes[r, j].set_title(f"traj {int(d['trajs'][tr])} {f}(t=0)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "condicoes_iniciais_maze_validacao.png", dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    report = {}
    stationary = audit_stationary(report)
    audit_values(report, stationary)
    audit_temporal(report, stationary)
    visual_table(report, stationary)
    initial_conditions(report)
    (OUT / "auditoria.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "temporal"}, indent=1, ensure_ascii=False))
    for r, c in report["temporal"].items():
        print(r, "variação < 10% do pico em t =", c["t_variacao_abaixo_10pct_do_pico"])


if __name__ == "__main__":
    main()
