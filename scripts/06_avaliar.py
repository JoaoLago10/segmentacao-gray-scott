"""Aplica Q0--Q10 às probabilidades salvas e calcula as métricas das quatro condições.

Para cada modelo (arquitetura x pré):
1. Escolhe os parâmetros de cada Qi na validação (máximo do Dice médio por trajetória).
2. Aplica ao teste e calcula, por trajetória: TP/FP/FN/TN, IoU, Dice, Precision, Recall e
   as métricas temporais; mede a latência do pós por frame.

Saídas em results/{exp,piloto}/:
    metricas_por_trajetoria.csv   uma linha por (arquitetura, pré, pós, trajetória de teste)
    metricas_padrao.csv           tabela padronizada da turma (Seção 0.9.7), média e IC 95%
    metricas_por_regime.csv       idem, por regime
    metricas_temporais.csv        métricas temporais agregadas
    parametros_pos.json           parâmetros escolhidos na validação e Dice de validação
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import postprocessing  # noqa: E402
from src.data.experiment import ROOT, load_config, load_split  # noqa: E402
from src.evaluation.metrics import bootstrap_ci, confusion, scores, temporal  # noqa: E402
from src.ground_truth.generation import gt_mask, valid_frames  # noqa: E402
from src.models import ARCH_NAMES  # noqa: E402
from src.preprocessing.methods import PRE_IDS  # noqa: E402

FLUIDO = "gray_scott_reaction_diffusion"
METRICS = ["IoU", "Dice", "Precision", "Recall"]
TEMPORAIS = ["ErroArea", "ErroCentroide", "ErroConsist"]


def condicao(pre: str, pos: str) -> str:
    return {(True, True): "Sem pré e sem pós", (False, True): "Somente pré",
            (True, False): "Somente pós", (False, False): "Pré e pós"}[(pre == "P0", pos == "Q0")]


def load_gt(cfg, split, piloto_ref=None):
    d = load_split(cfg, split)
    if piloto_ref is not None:  # alinha com o subconjunto usado no treino piloto
        idx = [np.nonzero((d["regime"] == r) & (d["traj"] == t))[0][0]
               for r, t in zip(piloto_ref["regime"], piloto_ref["traj"])]
        tsel = np.searchsorted(d["times"], piloto_ref["times"])
        d = {"A": d["A"][idx][:, tsel], "B": d["B"][idx][:, tsel], "regime": d["regime"][idx],
             "traj": d["traj"][idx], "times": d["times"][tsel]}
    d["M"] = gt_mask(d["B"])
    d["ok"] = valid_frames(d["B"], d["times"])
    return d


def eval_traj(qid, params, P, A, M):
    t0 = time.perf_counter()
    pred = postprocessing.apply(qid, P, A, **params)
    dt = time.perf_counter() - t0
    c = confusion(pred, M).sum(0)
    return c, pred, dt


REF: dict = {}  # GT compartilhado com os processos filhos (fork, sem cópia)


def evaluate_model(job):
    tag, out_dir = job
    arch, pre = tag.split("_")
    gts = REF
    probs = {s: np.load(out_dir / "probs" / f"{tag}_{s}.npz")["P"].astype(np.float32) for s in gts}
    params_out, rows = {}, []
    for qid in postprocessing.POS_IDS:
        # 1) escolha dos parâmetros na validação
        best, best_score = {}, -1.0
        g = gts["valid"]
        for params in postprocessing.grid(qid) or [{}]:
            dices = []
            for i in range(len(g["A"])):
                ok = g["ok"][i]
                if not ok.any():
                    continue
                c, _, _ = eval_traj(qid, params, probs["valid"][i][ok], g["A"][i][ok], g["M"][i][ok])
                dices.append(scores(c)["Dice"])
            s = float(np.mean(dices))
            if s > best_score:
                best, best_score = params, s
        params_out[qid] = {"params": best, "dice_valid": best_score}
        # 2) teste
        g = gts["test"]
        for i in range(len(g["A"])):
            ok = g["ok"][i]
            if not ok.any():
                continue
            c, pred, dt = eval_traj(qid, best, probs["test"][i][ok], g["A"][i][ok], g["M"][i][ok])
            s = scores(c)
            rows.append({"arch": arch, "Pre": pre, "Pos": qid, "Regime": str(g["regime"][i]),
                         "Traj": int(g["traj"][i]), "n_frames": int(ok.sum()),
                         "TP": int(c[0]), "FP": int(c[1]), "FN": int(c[2]), "TN": int(c[3]),
                         **{k: float(v) for k, v in s.items()}, **temporal(pred, g["M"][i][ok]),
                         "lat_pos_ms": 1000 * dt / ok.sum()})
    print(f"  {tag} ok", flush=True)
    return tag, params_out, rows


def aggregate(df: pd.DataFrame, runs: dict, by: list[str]) -> pd.DataFrame:
    out = []
    for keys, g in df.groupby(["arch", "Pre", "Pos"] + by, sort=False):
        arch, pre, pos = keys[:3]
        run = runs[f"{arch}_{pre}"]
        lat_pre = run["latencia_pre_ms_frame"] if pre != "P0" else 0.0
        row = {"Fluido": FLUIDO, "Condicao": condicao(pre, pos), "Pre": pre if pre != "P0" else "nenhum",
               "Arquitetura": ARCH_NAMES[arch], "Pos": pos if pos != "Q0" else "nenhum"}
        row.update({k: v for k, v in zip(by, keys[3:])})
        for m in METRICS:
            mean, half = bootstrap_ci(g[m].to_numpy())
            row[m], row[f"{m}_IC95"] = mean, half
        row["Latencia_ms"] = lat_pre + run["latencia_rede_ms_frame"] + g["lat_pos_ms"].mean()
        row["Memoria_MB"] = run["memoria_pico_MB"]
        row["Latencia_pre_ms"] = lat_pre
        row["Latencia_rede_ms"] = run["latencia_rede_ms_frame"]
        row["Latencia_pos_ms"] = g["lat_pos_ms"].mean()
        for m in TEMPORAIS:
            row[m] = float(np.nanmean(g[m]))
        row["n_trajetorias"] = len(g)
        out.append(row)
    return pd.DataFrame(out)


def order(df: pd.DataFrame) -> pd.DataFrame:
    """Ordem sequencial padronizada: condição -> arquitetura -> pré -> pós."""
    cond = ["Sem pré e sem pós", "Somente pré", "Somente pós", "Pré e pós"]
    arch = list(ARCH_NAMES.values())
    pre = ["nenhum"] + PRE_IDS[1:]
    pos = ["nenhum"] + postprocessing.POS_IDS[1:]
    df = df.copy()
    df["_c"] = df["Condicao"].map(cond.index)
    df["_a"] = df["Arquitetura"].map(arch.index)
    df["_p"] = df["Pre"].map(pre.index)
    df["_q"] = df["Pos"].map(pos.index)
    extra = [c for c in ("Regime",) if c in df]
    return df.sort_values(["_c", "_a", "_p", "_q"] + extra).drop(columns=["_c", "_a", "_p", "_q"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piloto", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    out_dir = ROOT / ("results/piloto" if args.piloto else "results/exp")
    runs = {p.stem: json.loads(p.read_text()) for p in sorted((out_dir / "runs").glob("*.json"))}
    if not runs:
        sys.exit(f"nenhum treino em {out_dir / 'runs'}")
    ref = {}
    for s in ("valid", "test"):
        r = np.load(out_dir / "probs" / f"{next(iter(runs))}_{s}.npz")
        ref[s] = load_gt(cfg, s, {k: r[k] for k in ("regime", "traj", "times")} if args.piloto else None)
        ref[s] = {k: ref[s][k] for k in ("A", "M", "ok", "regime", "traj")}
    REF.update(ref)
    jobs = [(tag, out_dir) for tag in runs]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("fork")) as ex:
        results = list(ex.map(evaluate_model, jobs))
    params = {tag: p for tag, p, _ in results}
    df = pd.DataFrame([r for _, _, rows in results for r in rows])
    df.to_csv(out_dir / "metricas_por_trajetoria.csv", index=False)
    (out_dir / "parametros_pos.json").write_text(json.dumps(params, indent=1))

    std_cols = ["Fluido", "Condicao", "Pre", "Arquitetura", "Pos"] + \
        [c for m in METRICS for c in (m, f"{m}_IC95")] + ["Latencia_ms", "Memoria_MB"]
    geral = order(aggregate(df, runs, []))
    geral[std_cols].to_csv(out_dir / "metricas_padrao.csv", index=False, float_format="%.4f")
    geral.to_csv(out_dir / "metricas_completas.csv", index=False, float_format="%.4f")
    order(aggregate(df, runs, ["Regime"])).to_csv(out_dir / "metricas_por_regime.csv", index=False,
                                                  float_format="%.4f")
    geral[["Condicao", "Pre", "Arquitetura", "Pos"] + TEMPORAIS].to_csv(
        out_dir / "metricas_temporais.csv", index=False, float_format="%.4f")
    print(geral[geral["Condicao"] != "Pré e pós"][std_cols[1:11]].to_string(index=False))


if __name__ == "__main__":
    main()
