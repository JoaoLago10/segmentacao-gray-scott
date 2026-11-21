"""Figuras do artigo a partir dos resultados salvos.

- tabela_visual_{regime}.png: t = 0, dois instantes intermediários e o final; linhas com A bruto,
  A pré-processado, GT e as máscaras das 3 arquiteturas nas 4 condições (mapa de erros).
- heatmap_dice.png: Dice de teste para P0--P10 x Q0--Q10, por arquitetura (as 4 condições numa grade).
- curvas_treino.png: perda de treino e validação nas 20 épocas.
- dice_por_regime.png: Dice por regime na condição de referência e na melhor combinação.

As "melhores" condições de cada arquitetura são escolhidas pelo Dice de **validação**.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import postprocessing, preprocessing  # noqa: E402
from src.data.experiment import ROOT, load_config, load_split  # noqa: E402
from src.ground_truth.generation import TAU, gt_mask, valid_frames  # noqa: E402
from src.models import ARCH_NAMES  # noqa: E402

ARCH_COLORS = {"pointnet": "#2a78d6", "pointnet2": "#eb6834", "dgcnn": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e1e0d9"
DIVERGE = LinearSegmentedColormap.from_list("azul_cinza_vermelho", ["#2a78d6", "#f0efec", "#e34948"])
# Mapa de erros: TN, TP, FP, FN
ERR = ListedColormap(["#f0efec", "#1f1f1e", "#e34948", "#2a78d6"])
REG_PT = {"gliders": "Gliders", "bubbles": "Bubbles", "maze": "Maze", "worms": "Worms",
          "spirals": "Spirals", "spots": "Spots"}

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 8,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def best_choices(params: dict, arch: str) -> dict:
    """Escolhe pela validação: melhor pré (com Q0), melhor pós (com P0) e melhor par."""
    dv = {(t.split("_")[1], q): v["dice_valid"] for t, qs in params.items() if t.startswith(arch + "_")
          for q, v in qs.items()}
    pre = max((p for p, q in dv if q == "Q0" and p != "P0"), key=lambda p: dv[(p, "Q0")], default="P0")
    pos = max((q for p, q in dv if p == "P0" and q != "Q0"), key=lambda q: dv[("P0", q)], default="Q0")
    both = max(((p, q) for p, q in dv if p != "P0" and q != "Q0"), key=dv.get, default=("P0", "Q0"))
    return {"Sem pré e sem pós": ("P0", "Q0"), "Somente pré": (pre, "Q0"),
            "Somente pós": ("P0", pos), "Pré e pós": both}


def err_map(pred, gt):
    return np.select([pred & gt, pred & ~gt, ~pred & gt], [1, 2, 3], 0)


def visual_tables(cfg, out_dir, fig_dir, params, archs):
    vis = load_split(cfg, "visual")
    test = load_split(cfg, "test")
    times_v = list(vis["times"])
    train = load_split(cfg, "train")
    A_fit = train["A"][valid_frames(train["B"], train["times"])]
    # pré mostrado na 2ª linha: o pré mais escolhido entre as arquiteturas na condição "Pré e pós"
    choices = {a: best_choices(params, a) for a in archs}
    pres = [choices[a]["Pré e pós"][0] for a in archs]
    pre_show = max(set(pres), key=pres.count)
    pre = preprocessing.build(pre_show).fit(A_fit)
    cache = {}

    def probs(tag, split):
        if (tag, split) not in cache:
            cache[(tag, split)] = np.load(out_dir / "probs" / f"{tag}_{split}.npz")
        return cache[(tag, split)]

    for r in REG_PT:
        iv = int(np.nonzero(vis["regime"] == r)[0][0])
        traj = int(vis["traj"][iv])
        it = np.nonzero((test["regime"] == r) & (test["traj"] == traj))[0]
        A, B = vis["A"][iv], vis["B"][iv]
        M = gt_mask(B).astype(bool)
        rows = [("A bruto", A, "viridis"), (f"A após {pre_show}", pre(A), "viridis"),
                (f"GT: B > {TAU:.3f}", M, "gray_r")]
        for a in archs:
            for cond, (p, q) in choices[a].items():
                tag = f"{a}_{p}"
                Pv = probs(tag, "visual")["P"][iv].astype(np.float32)
                if len(it) and q != "Q0":
                    # sequência densa do teste (Q10 precisa de vizinhos no tempo) + t = 0 do conjunto visual
                    # sequência = frames do teste + frames do conjunto visual que faltarem, em ordem temporal
                    pt = probs(tag, "test")
                    jt = np.nonzero((pt["regime"] == r) & (pt["traj"] == traj))[0]
                    frames = {}
                    if len(jt):
                        At = test["A"][it[0]][np.searchsorted(test["times"], pt["times"])]
                        frames = {int(t): (pt["P"][jt[0]][k].astype(np.float32), At[k])
                                  for k, t in enumerate(pt["times"])}
                    for k, t in enumerate(times_v):
                        frames.setdefault(int(t), (Pv[k], A[k]))
                    tseq = sorted(frames)
                    seq = np.stack([frames[t][0] for t in tseq])
                    Aseq = np.stack([frames[t][1] for t in tseq])
                    out = postprocessing.apply(q, seq, Aseq, **params[tag][q]["params"])
                    pred = np.stack([out[tseq.index(int(t))] for t in times_v]).astype(bool)
                else:
                    pred = postprocessing.apply(q, Pv, A, **params[tag][q]["params"]).astype(bool)
                lab = f"{ARCH_NAMES[a]}\n{cond}\n({p if p != 'P0' else '-'} / {q if q != 'Q0' else '-'})"
                rows.append((lab, err_map(pred, M), ERR))
        fig, axes = plt.subplots(len(rows), len(times_v), figsize=(1.25 * len(times_v) + 1.3, 1.25 * len(rows)))
        for i, (lab, img, cmap) in enumerate(rows):
            for j, t in enumerate(times_v):
                ax = axes[i, j]
                if cmap is ERR:
                    ax.imshow(img[j], cmap=ERR, vmin=0, vmax=3, interpolation="nearest")
                else:
                    ax.imshow(img[j], cmap=cmap, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"t = {10 * t:g} s" + (" (cond. inicial)" if t == 0 else ""), color=INK)
                if j == 0:
                    ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=6.5, color=INK)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ERR.colors]
        fig.subplots_adjust(top=1 - 0.55 / fig.get_figheight(), bottom=0.45 / fig.get_figheight(),
                            hspace=0.08, wspace=0.05)
        fig.legend(handles, ["VN", "VP", "FP", "FN"], loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.55, 0.0))
        fig.suptitle(f"{REG_PT[r]} (trajetória de teste {traj})", color=INK, y=1 - 0.1 / fig.get_figheight())
        fig.savefig(fig_dir / f"tabela_visual_{r}.png")
        plt.close(fig)


def heatmaps(geral, fig_dir, archs):
    pre_ids, pos_ids = preprocessing.PRE_IDS, postprocessing.POS_IDS
    lab_pre = ["nenhum"] + pre_ids[1:]
    lab_pos = ["nenhum"] + pos_ids[1:]
    fig, axes = plt.subplots(1, len(archs), figsize=(4.3 * len(archs), 4.2))
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, archs):
        g = geral[geral["Arquitetura"] == ARCH_NAMES[a]]
        D = np.full((len(pre_ids), len(pos_ids)), np.nan)
        for _, row in g.iterrows():
            D[lab_pre.index(row["Pre"]), lab_pos.index(row["Pos"])] = row["Dice"]
        base = D[0, 0]
        lim = np.nanmax(np.abs(D - base)) or 1e-3
        ax.imshow(D - base, cmap=DIVERGE, vmin=-lim, vmax=lim)
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if np.isfinite(D[i, j]):
                    ax.text(j, i, f"{D[i, j]:.3f}"[1:], ha="center", va="center", fontsize=5, color=INK)
        ax.set_xticks(range(len(pos_ids)), lab_pos, rotation=90)
        ax.set_yticks(range(len(pre_ids)), lab_pre)
        ax.set_xlabel("Pós-processamento"); ax.set_ylabel("Pré-processamento")
        ax.set_title(f"{ARCH_NAMES[a]} (referência: Dice {base:.3f})", color=INK)
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, ec=INK, lw=1.2))
    fig.savefig(fig_dir / "heatmap_dice.png")
    plt.close(fig)


def curves(out_dir, fig_dir, archs):
    pres = preprocessing.PRE_IDS
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.2), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), pres):
        for a in archs:
            f = out_dir / "runs" / f"{a}_{p}.json"
            if not f.exists():
                continue
            h = json.loads(f.read_text())["historico"]
            e = [x["epoca"] for x in h]
            ax.plot(e, [x["perda_treino"] for x in h], color=ARCH_COLORS[a], lw=1.5, label=f"{ARCH_NAMES[a]} treino")
            ax.plot(e, [x["perda_valid"] for x in h], color=ARCH_COLORS[a], lw=1.5, ls="--",
                    label=f"{ARCH_NAMES[a]} validação")
        ax.set_title("nenhum" if p == "P0" else p, color=INK)
        ax.grid(color=GRID, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes.ravel()[-1].axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[-1].legend(h, l, loc="center", frameon=False, fontsize=7)
    fig.supxlabel("Época"); fig.supylabel("Perda (BCE + Dice)")
    fig.savefig(fig_dir / "curvas_treino.png")
    plt.close(fig)


def per_regime(reg, params, fig_dir, archs):
    fig, axes = plt.subplots(1, len(archs), figsize=(4.2 * len(archs), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    conds = ["Sem pré e sem pós", "Somente pré", "Somente pós", "Pré e pós"]
    shades = {"Sem pré e sem pós": 0.35, "Somente pré": 0.55, "Somente pós": 0.75, "Pré e pós": 1.0}
    x = np.arange(len(REG_PT))
    for ax, a in zip(axes, archs):
        ch = best_choices(params, a)
        for k, c in enumerate(conds):
            p, q = ch[c]
            g = reg[(reg["Arquitetura"] == ARCH_NAMES[a]) & (reg["Pre"] == (p if p != "P0" else "nenhum"))
                    & (reg["Pos"] == (q if q != "Q0" else "nenhum"))].set_index("Regime")
            vals = [g.loc[r, "Dice"] if r in g.index else np.nan for r in REG_PT]
            err = [g.loc[r, "Dice_IC95"] if r in g.index else 0 for r in REG_PT]
            ax.bar(x + (k - 1.5) * 0.2, vals, 0.18, yerr=err, color=ARCH_COLORS[a], alpha=shades[c],
                   label=f"{c} ({p if p != 'P0' else '-'}/{q if q != 'Q0' else '-'})", ec="white", lw=0.5,
                   error_kw={"lw": 0.6, "ecolor": MUTED})
        ax.set_xticks(x, list(REG_PT.values()), rotation=30)
        ax.set_title(ARCH_NAMES[a], color=INK)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=5.5, frameon=False, loc="lower left")
    axes[0].set_ylabel("Dice (teste, IC 95%)")
    fig.savefig(fig_dir / "dice_por_regime.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piloto", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    out_dir = ROOT / ("results/piloto" if args.piloto else "results/exp")
    fig_dir = ROOT / "docs/artigo/figs" / ("piloto" if args.piloto else "exp")
    fig_dir.mkdir(parents=True, exist_ok=True)
    params = json.loads((out_dir / "parametros_pos.json").read_text())
    archs = [a for a in ARCH_NAMES if any(t.startswith(a + "_") for t in params)]
    geral = pd.read_csv(out_dir / "metricas_completas.csv")
    reg = pd.read_csv(out_dir / "metricas_por_regime.csv")
    heatmaps(geral, fig_dir, archs)
    curves(out_dir, fig_dir, archs)
    per_regime(reg, params, fig_dir, archs)
    visual_tables(cfg, out_dir, fig_dir, params, archs)
    (out_dir / "escolhas_validacao.json").write_text(
        json.dumps({a: best_choices(params, a) for a in archs}, indent=1, ensure_ascii=False))
    print("figuras em", fig_dir)


if __name__ == "__main__":
    main()
