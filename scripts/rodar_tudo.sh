#!/usr/bin/env bash
# Experimento completo (GPU recomendada: ~5-6 h numa T4; 33 treinos de 20 épocas).
# Uso: bash scripts/rodar_tudo.sh            # experimento completo -> results/exp
#      bash scripts/rodar_tudo.sh --piloto   # versão reduzida (CPU)  -> results/piloto
# Pode ser interrompido e retomado: treinos já concluídos (runs/*.json) são pulados.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export MPLCONFIGDIR=${MPLCONFIGDIR:-.cache/mpl}
EXTRA=("$@")
$PY scripts/04_preparar_dados.py
$PY scripts/05_treinar.py --arch all --pre all "${EXTRA[@]}"
$PY scripts/06_avaliar.py "${EXTRA[@]}"
$PY scripts/07_figuras.py "${EXTRA[@]}"
$PY scripts/08_tabelas_latex.py "${EXTRA[@]}"
echo "Pronto. CSV em results/, figuras em docs/artigo/figs/, tabelas em docs/artigo/tabelas/."
