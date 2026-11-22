# Segmentação 2D de padrões no Gray-Scott (The Well)

Trabalho P2 de Processamento Digital de Imagens (UFMA): segmentação dos padrões de Turing do dataset
[`gray_scott_reaction_diffusion`](https://polymathic-ai.org/the_well/datasets/gray_scott_reaction_diffusion/)
do [The Well](https://github.com/PolymathicAI/the_well), seguindo o [protocolo da turma](PROTOCOLO.MD):
**U-Net, SegFormer e Mask2Former** (20 épocas, melhor época pela validação), baseline + 10 pré-processamentos
(P1–P10) e baseline + 5 pós-processamentos (Q1–Q5), em quatro condições: sem pré e sem pós, somente pré,
somente pós, pré e pós.

## Situação

| Etapa | Situação |
|---|---|
| 1. Auditoria do dataset | concluída ([relatório](docs/01_auditoria.md)) |
| 2. Ground truth | **definido**: `B > τ`, τ = Otsu global do treino (0,1507), máscara gerada em cada frame; entrada = só A ([ficha](docs/02_ficha_ground_truth.md), [código](src/ground_truth/generation.py)) |
| 3. Software (P, Q, redes, métricas) | concluído, com testes (`pytest tests`) |
| 4. Piloto (CPU, reduzido) | `bash scripts/rodar_tudo.sh --piloto` → `results/piloto/` |
| 5. Experimento completo (GPU) | **pendente**: [notebooks/executar_gpu.ipynb](notebooks/executar_gpu.ipynb) no Kaggle (T4 x2) |
| 6. Tabelas e figuras | geradas automaticamente pelos scripts 06–08 |
| 7. Artigo (LaTeX, elsarticle) | [docs/artigo/](docs/artigo/); os números finais dependem da etapa 5 |

## Decisões

- **Protocolo da turma** ([PROTOCOLO.MD](PROTOCOLO.MD)): U-Net, SegFormer, Mask2Former; P1–P10 (Z-score, RMS, Robust,
  Min-Max, Percentile clipping, Gaussiano, Mediana, Bilateral, Difusão anisotrópica, Wavelets); Q1–Q5 (limiar
  otimizado, abertura, fechamento, filtro de componentes, consistência temporal); 20 épocas, checkpoint da melhor
  época pelo Dice de validação.
- **Resolução:** 128×128 sem alteração (o maior eixo já é ≤ 128 px; sem upscaling, nem dentro das redes).
- **Dados:** todas as 1200 trajetórias da divisão oficial; 10 frames por trajetória de treino e 50 na validação e
  no teste (~2,8 GB, abaixo do teto de 25 GB).
- **Pesos:** SegFormer (mit-b0, ImageNet) e Mask2Former (Swin-T, ADE20K) pré-treinados; U-Net do zero, com
  padding circular.
- **Baseline P0** = A bruto (já é uma concentração adimensional em [0,09; 1]); **"sem pós"** = limiar 0,5.
- Frames excluídos: t = 0 e frames sem padrão (max B < 1e-6).

## Estrutura

- `configs/experimento.yaml`: toda a configuração do experimento.
- `src/data/`: leitura parcial dos HDF5 do Hugging Face e subconjuntos do experimento.
- `src/ground_truth/`: candidatos (Etapa 2) e o GT definitivo.
- `src/preprocessing/`: P0–P10 (ajuste só no treino; filtros periódicos).
- `src/postprocessing/`: Q0–Q5 (periódicos; parâmetros escolhidos na validação).
- `src/models/`: U-Net, SegFormer e Mask2Former com interface comum.
- `src/evaluation/`: IoU, Dice, Precision, Recall por trajetória e por classe, IC 95% por bootstrap e métricas temporais.
- `scripts/04`–`08`: dados → treino → avaliação → figuras → tabelas LaTeX; `scripts/rodar_tudo.sh` executa tudo.
- `results/{piloto,exp}/`: `resultados_protocolo.csv` (esquema da Seção 12 do protocolo), `metricas_padrao.csv`,
  métricas por regime, temporais e por trajetória.
- `docs/artigo/`: artigo (`main.tex` + `secoes/`), com tabelas e figuras geradas.

## Reprodução

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/05_treinar.py --bench --amp      # estima o tempo total na GPU
PY=.venv/bin/python bash scripts/rodar_tudo.sh            # completo (GPU)
PY=.venv/bin/python bash scripts/rodar_tudo.sh --piloto   # piloto (CPU)
```

Artigo: compilar `docs/artigo/main.tex` no Overleaf (pdfLaTeX + BibTeX). Depois do experimento
completo, troque `\newcommand{\resdir}{piloto}` por `{exp}` em `main.tex`.
