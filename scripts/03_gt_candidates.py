"""Etapa 2: análise dos candidatos de ground truth (nenhum é escolhido aqui).

Limiares ajustados apenas nas amostras de treino (t > 0); efeitos medidos na validação.
Saídas: ``results/gt/candidatos.json`` e ``docs/figs/gt/*.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.remote import REGIMES  # noqa: E402
from src.data.sampling import sample_path  # noqa: E402
from src.ground_truth import candidates as gt  # noqa: E402

OUT = ROOT / "results/gt"
FIG = ROOT / "docs/figs/gt"
DT = 10.0
MID = 500


def load(split, regime):
    d = np.load(ROOT / sample_path(split, regime, "stride20"))
    return {k: d[k] for k in d.files}


def alive(B):
    """Frames em que ainda existe padrão (máximo de B acima de 1e-6)."""
    return np.abs(B).reshape(*B.shape[:-2], -1).max(-1) > 1e-6


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rep = {}
    train = {r: load("train", r) for r in REGIMES}
    valid = {r: load("valid", r) for r in REGIMES}

    # 1. Limiares candidatos (treino, t > 0)
    B_all = np.concatenate([train[r]["B"][:, 1:].ravel() for r in REGIMES])
    B_alive = np.concatenate([train[r]["B"][:, 1:][alive(train[r]["B"][:, 1:])].ravel() for r in REGIMES])
    D_all = np.concatenate([gt.trivial_state_distance(train[r]["A"][:, 1:], train[r]["B"][:, 1:]).ravel()
                            for r in REGIMES])
    A_all = np.concatenate([train[r]["A"][:, 1:].ravel() for r in REGIMES])
    thr = {
        "otsu_B_global": gt.otsu_threshold(B_all),
        "otsu_B_global_sem_frames_mortos": gt.otsu_threshold(B_alive),
        "otsu_B_por_regime": {r: gt.otsu_threshold(train[r]["B"][:, 1:]) for r in REGIMES},
        "otsu_distancia_estado_trivial": gt.otsu_threshold(D_all),
        "otsu_A_global": gt.otsu_threshold(A_all),
    }
    rep["limiares"] = thr
    tg = thr["otsu_B_global"]

    # 2. Estatísticas das máscaras na validação (t > 0)
    stats = {}
    for r in REGIMES:
        A, B = valid[r]["A"][:, 1:], valid[r]["B"][:, 1:]
        tr = thr["otsu_B_por_regime"][r]
        mg, mr = gt.binary_mask(B, tg), gt.binary_mask(B, tr)
        md = (gt.trivial_state_distance(A, B) > thr["otsu_distancia_estado_trivial"]).astype(np.uint8)
        m3 = gt.three_class_mask(B, tg)
        mid = valid[r]["B"][:, list(valid[r]["times"]).index(MID)]
        ncomp = [gt.periodic_label(gt.binary_mask(b, tg))[1] for b in mid]
        stats[r] = {
            "frac_padrao_tau_global": float(mg.mean()),
            "frac_padrao_tau_regime": float(mr.mean()),
            "frac_padrao_distancia_trivial": float(md.mean()),
            "frac_mascaras_vazias_tau_global": float((mg.reshape(*mg.shape[:2], -1).max(-1) == 0).mean()),
            "frac_classes_3c": [float((m3 == c).mean()) for c in range(3)],
            "iou_tau_global_vs_tau_regime": gt.iou(mg, mr),
            "iou_tau_global_vs_distancia_trivial": gt.iou(mg, md),
            "componentes_t5000_media": float(np.mean(ncomp)),
            "componentes_t5000_min_max": [int(min(ncomp)), int(max(ncomp))],
        }
    rep["mascaras_validacao"] = stats

    # 3. Vazamento de rótulo: quanto um simples limiar em A (ou em B) reproduz a máscara de B
    leak = {}
    grid = np.linspace(0.2, 1.0, 81)
    for r in REGIMES:
        A, B = valid[r]["A"][:, 1:], valid[r]["B"][:, 1:]
        m = gt.binary_mask(B, tg)
        iou_otsuA = gt.iou(A < thr["otsu_A_global"], m)
        best = max((gt.iou(A < a, m), a) for a in grid)
        leak[r] = {"iou_A_otsu": iou_otsuA, "iou_A_melhor_limiar": best[0], "melhor_limiar_A": float(best[1])}
    rep["vazamento"] = leak

    # 4. "Frame médio": quanto uma máscara fixa representa os outros instantes
    fm = {}
    fig, axes = plt.subplots(1, 6, figsize=(22, 3.6), sharey=True)
    for ax, r in zip(axes, REGIMES):
        B = valid[r]["B"]
        times = list(valid[r]["times"])
        keep = alive(B[:, -1])
        Bk = B[keep]
        k_mid = times.index(MID)
        k_from = times.index(100)
        m_t = gt.binary_mask(Bk, tg)
        m_mid = m_t[:, k_mid]
        m_mean = gt.binary_mask(Bk[:, k_from:].mean(1), tg)
        iou_mid = np.array([[gt.iou(m_t[i, j], m_mid[i]) for j in range(len(times))] for i in range(len(Bk))])
        iou_mean = np.array([[gt.iou(m_t[i, j], m_mean[i]) for j in range(len(times))] for i in range(len(Bk))])
        t = np.array(times) * DT
        for y, lab in ((iou_mid, "máscara de t = 5000 s"), (iou_mean, "máscara da média de B (t >= 1000 s)")):
            q = np.nanquantile(y, [0.1, 0.5, 0.9], axis=0)
            ax.fill_between(t, q[0], q[2], alpha=0.25)
            ax.plot(t, q[1], label=lab)
        ax.set_title(f"{r} ({keep.sum()} traj.)")
        ax.set_xlabel("t (s)")
        fm[r] = {
            "iou_mediana_vs_t5000_em_t1000": float(np.nanmedian(iou_mid[:, times.index(100)])),
            "iou_mediana_vs_t5000_em_t10000": float(np.nanmedian(iou_mid[:, -1])),
            "iou_mediana_vs_media_em_t5000": float(np.nanmedian(iou_mean[:, k_mid])),
            "iou_mediana_vs_media_em_t10000": float(np.nanmedian(iou_mean[:, -1])),
            "frac_pixels_mascara_media": float(m_mean.mean()),
        }
    axes[0].set_ylabel("IoU(M_t, máscara fixa)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "frame_medio_iou_temporal.png", dpi=110)
    plt.close(fig)
    rep["frame_medio"] = fm

    # 5. Galeria de candidatos (validação, trajetória 0, t = 400 s e t = 5000 s)
    cols = ["B", "binária τ global", "binária τ regime", "dist. estado trivial", "3 classes",
            "componentes periódicos", "média temporal de B", "máscara da média"]
    for tidx in (40, MID):
        fig, axes = plt.subplots(6, len(cols), figsize=(2.2 * len(cols), 2.3 * 6))
        for i, r in enumerate(REGIMES):
            times = list(valid[r]["times"])
            A = valid[r]["A"][0, times.index(tidx)]
            B = valid[r]["B"][0, times.index(tidx)]
            Bmean = valid[r]["B"][0, times.index(100):].mean(0)
            lab, n = gt.periodic_label(gt.binary_mask(B, tg))
            imgs = [B, gt.binary_mask(B, tg), gt.binary_mask(B, thr["otsu_B_por_regime"][r]),
                    gt.trivial_state_distance(A, B) > thr["otsu_distancia_estado_trivial"],
                    gt.three_class_mask(B, tg), np.where(lab > 0, (lab * 37) % 255 + 1, 0),
                    Bmean, gt.binary_mask(Bmean, tg)]
            cmaps = ["magma", "gray", "gray", "gray", "viridis", "nipy_spectral", "magma", "gray"]
            for j, (im, cm) in enumerate(zip(imgs, cmaps)):
                axes[i, j].imshow(np.asarray(im, float).T, origin="lower", cmap=cm, interpolation="nearest")
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                if i == 0:
                    axes[i, j].set_title(cols[j], fontsize=9)
            axes[i, 0].set_ylabel(r)
            axes[i, 5].set_xlabel(f"{n} componentes", fontsize=8)
        fig.suptitle(f"Candidatos de GT, validação, trajetória 0, t = {tidx * DT:.0f} s "
                     f"(τ global = {tg:.3f})")
        fig.tight_layout()
        fig.savefig(FIG / f"galeria_candidatos_t{int(tidx * DT)}.png", dpi=100)
        plt.close(fig)

    # 6. Histograma de B com os limiares
    fig, axes = plt.subplots(1, 6, figsize=(22, 3.2))
    for ax, r in zip(axes, REGIMES):
        ax.hist(train[r]["B"][:, 1:].ravel(), bins=200, log=True, color="C1")
        ax.axvline(tg, color="k", label=f"Otsu global {tg:.3f}")
        ax.axvline(thr["otsu_B_por_regime"][r], color="C0", ls="--",
                   label=f"Otsu regime {thr['otsu_B_por_regime'][r]:.3f}")
        ax.set_title(f"{r}: B (treino, t > 0)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "histogramas_B_limiares.png", dpi=110)
    plt.close(fig)

    (OUT / "candidatos.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
