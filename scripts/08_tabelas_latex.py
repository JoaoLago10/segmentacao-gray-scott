"""Gera as tabelas LaTeX do artigo a partir dos CSVs (formato padronizado da turma).

Saídas em docs/artigo/tabelas/{exp,piloto}/:
    t_condicoes_123.tex   condições 1--3 (63 linhas), ordem sequencial padronizada
    t_condicao_4.tex      condição 4 (300 linhas, longtable, apêndice)
    t_resumo.tex          referência e melhores combinações (escolhidas na validação) por arquitetura
    t_temporais.tex       métricas temporais das mesmas combinações
    t_regimes.tex         Dice por regime nas mesmas combinações
    t_custo.tex           parâmetros, tempo de treino, latência e memória
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.experiment import ROOT  # noqa: E402
from src.models import ARCH_NAMES  # noqa: E402

M = ["IoU", "Dice", "Precision", "Recall"]
COND = ["Sem pré e sem pós", "Somente pré", "Somente pós", "Pré e pós"]
REG = ["gliders", "bubbles", "maze", "worms", "spirals", "spots"]


def f3(x):
    return f"{x:.3f}".replace(".", ",")


def pm(r, m):
    return f"{f3(r[m])} $\\pm$ {f3(r[m + '_IC95'])}"


def lat(x):
    return f"{x:.1f}".replace(".", ",") if pd.notna(x) else "--"


def row_std(r):
    return " & ".join([r["Condicao"], r["Pre"], r["Arquitetura"], r["Pos"]] + [pm(r, m) for m in M]
                      + [lat(r["Latencia_ms"]), lat(r["Memoria_MB"])]) + r" \\"


HEAD = (r"Condição & Pré & Arquitetura & Pós & IoU & Dice & Precision & Recall & Lat. (ms) & Mem. (MB) \\")


def longtable(rows, caption, label):
    return "\n".join([
        r"{\scriptsize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{longtable}{llllcccccc}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule", HEAD, r"\midrule", r"\endfirsthead",
        r"\toprule", HEAD, r"\midrule", r"\endhead",
        r"\bottomrule", r"\endfoot",
        *rows, r"\end{longtable}}", ""])


def key(p, q):
    return (p if p != "P0" else "nenhum", q if q != "Q0" else "nenhum")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piloto", action="store_true")
    args = ap.parse_args()
    sub = "piloto" if args.piloto else "exp"
    res = ROOT / "results" / sub
    out = ROOT / "docs/artigo/tabelas" / sub
    out.mkdir(parents=True, exist_ok=True)
    g = pd.read_csv(res / "metricas_completas.csv")
    reg = pd.read_csv(res / "metricas_por_regime.csv")
    esc = json.loads((res / "escolhas_validacao.json").read_text())
    runs = {p.stem: json.loads(p.read_text()) for p in (res / "runs").glob("*.json")}
    nota = " Resultados do \\textbf{piloto} reduzido (CPU), apenas para validar o pipeline." if args.piloto else ""

    c123 = g[g["Condicao"] != "Pré e pós"]
    (out / "t_condicoes_123.tex").write_text(longtable(
        [row_std(r) for _, r in c123.iterrows()],
        "Resultados no teste nas condições sem pré e sem pós, somente pré e somente pós "
        "(média por trajetória $\\pm$ semi-amplitude do IC de 95\\% por bootstrap)." + nota,
        "tab:condicoes123"))
    c4 = g[g["Condicao"] == "Pré e pós"]
    (out / "t_condicao_4.tex").write_text(longtable(
        [row_std(r) for _, r in c4.iterrows()],
        "Resultados no teste na condição pré e pós (todas as combinações)." + nota, "tab:condicao4"))

    sel = []
    for a, ch in esc.items():
        for c in COND:
            p, q = ch[c]
            pp, qq = key(p, q)
            r = g[(g["Arquitetura"] == ARCH_NAMES[a]) & (g["Pre"] == pp) & (g["Pos"] == qq)]
            if len(r):
                sel.append((a, c, pp, qq, r.iloc[0]))

    lines = [r"\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{2.5pt}",
             r"\caption{Referência e melhor configuração de cada condição por arquitetura "
             r"(escolhida pelo Dice de validação; métricas no teste)." + nota + r"}\label{tab:resumo}",
             r"\begin{tabular}{llllcccccc}", r"\toprule", HEAD, r"\midrule"]
    last = None
    for a, c, pp, qq, r in sel:
        if last and last != a:
            lines.append(r"\midrule")
        lines.append(row_std(r))
        last = a
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out / "t_resumo.tex").write_text("\n".join(lines))

    lines = [r"\begin{table}[t]\centering\scriptsize",
             r"\caption{Métricas temporais no teste (erro em relação ao GT; menor é melhor): variação de área "
             r"(fração do domínio por passo de 200~s), distância do centroide periódico (px) e consistência "
             r"temporal (diferença de IoU entre frames consecutivos)." + nota + r"}\label{tab:temporais}",
             r"\begin{tabular}{llllccc}", r"\toprule",
             r"Arquitetura & Condição & Pré & Pós & $E_{\text{área}}$ & $E_{\text{centr.}}$ (px) & $E_{\text{cons.}}$ \\",
             r"\midrule"]
    last = None
    for a, c, pp, qq, r in sel:
        if last and last != a:
            lines.append(r"\midrule")
        lines.append(f"{ARCH_NAMES[a]} & {c} & {pp} & {qq} & {f3(r['ErroArea'])} & "
                     f"{lat(r['ErroCentroide'])} & {f3(r['ErroConsist'])} \\\\")
        last = a
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out / "t_temporais.tex").write_text("\n".join(lines))

    lines = [r"\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3pt}",
             r"\caption{Dice no teste por regime (média por trajetória)." + nota + r"}\label{tab:regimes}",
             r"\begin{tabular}{lll" + "c" * len(REG) + "}", r"\toprule",
             "Arquitetura & Condição & Pré/Pós & " + " & ".join(x.capitalize() for x in REG) + r" \\",
             r"\midrule"]
    last = None
    for a, c, pp, qq, _ in sel:
        if last and last != a:
            lines.append(r"\midrule")
        rr = reg[(reg["Arquitetura"] == ARCH_NAMES[a]) & (reg["Pre"] == pp) & (reg["Pos"] == qq)].set_index("Regime")
        vals = [f3(rr.loc[x, "Dice"]) if x in rr.index else "--" for x in REG]
        lines.append(f"{ARCH_NAMES[a]} & {c} & {pp}/{qq} & " + " & ".join(vals) + r" \\")
        last = a
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out / "t_regimes.tex").write_text("\n".join(lines))

    lines = [r"\begin{table}[t]\centering\scriptsize",
             r"\caption{Custo computacional por arquitetura (média sobre os pré-processamentos). Latência por "
             r"frame $128\times128$ só da rede, com todos os blocos do frame num lote." + nota + r"}\label{tab:custo}",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Arquitetura & Parâmetros (M) & Épocas & Treino (min) & Latência (ms) & Memória (MB) \\", r"\midrule"]
    for a in esc:
        rs = [v for k, v in runs.items() if k.startswith(a + "_")]
        n = len(rs)
        lines.append(f"{ARCH_NAMES[a]} & {rs[0]['n_parametros'] / 1e6:.2f}".replace(".", ",")
                     + f" & {rs[0]['epocas']} & {lat(sum(r['tempo_treino_s'] for r in rs) / n / 60)} & "
                     f"{lat(sum(r['latencia_rede_ms_frame'] for r in rs) / n)} & "
                     f"{lat(sum(r['memoria_pico_MB'] for r in rs) / n)} \\\\")
    dev = next(iter(runs.values()))
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\par\smallskip Dispositivo: {dev['gpu'] or dev['dispositivo']} (PyTorch {dev['versoes']['torch']}).",
              r"\end{table}", ""]
    (out / "t_custo.tex").write_text("\n".join(lines).replace("_", r"\_"))
    print("tabelas em", out)


if __name__ == "__main__":
    main()
