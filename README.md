# Segmentação 2D de padrões no Gray-Scott (The Well) com redes de nuvem de pontos

Trabalho P2 de Processamento Digital de Imagens (UFMA): segmentação dos padrões de Turing do dataset
[`gray_scott_reaction_diffusion`](https://polymathic-ai.org/the_well/datasets/gray_scott_reaction_diffusion/)
do [The Well](https://github.com/PolymathicAI/the_well), com 10 pré-processamentos (P1–P10),
3 arquiteturas de **nuvem de pontos** (PointNet, PointNet++, DGCNN; 20 épocas) e 10 pós-processamentos (Q1–Q10),
em quatro condições: sem pré e sem pós, somente pré, somente pós, pré e pós.

## Situação

| Etapa | Situação |
|---|---|
| 1. Auditoria do dataset | concluída ([relatório](docs/01_auditoria.md)) |
| 2. Ground truth | **definido** em 21/11/2026: `B > τ`, τ = Otsu global do treino (0,1507), máscara gerada em cada frame; entrada = só A ([ficha](docs/02_ficha_ground_truth.md), [código](src/ground_truth/generation.py)) |
| 3. Software (P1–P10, Q1–Q10, redes, métricas) | concluído, com testes (`pytest tests`) |
| 4. Piloto (CPU, reduzido) | `bash scripts/rodar_tudo.sh --piloto` → `results/piloto/` |
| 5. Experimento completo (GPU) | **pendente**: rodar [notebooks/executar_gpu.ipynb](notebooks/executar_gpu.ipynb) no Colab/Kaggle (~5–6 h numa T4) |
| 6. Tabelas e figuras | geradas automaticamente pelos scripts 06–08 |
| 7. Artigo (LaTeX, elsarticle) | [docs/artigo/](docs/artigo/), com fundamentação completa; números finais dependem da etapa 5 |

## Decisões (21/11/2026)

- **Arquiteturas:** PointNet, PointNet++ (SSG) e DGCNN, implementadas em PyTorch puro ([src/models/pointcloud.py](src/models/pointcloud.py)).
  Cada frame vira uma nuvem de pontos `(x, y, a)`, em que `a` é o campo A pré-processado; treino em blocos periódicos 64×64.
- **Épocas:** 20 para todas as arquiteturas.
- **Resolução:** mantida em 128×128. É o menor fluido do The Well, então o fator de redução é 1, e reduzir mais
  destruiria estruturas de 4–8 px.
- **"Sem pré"** = A bruto; **"sem pós"** = limiar 0,5; Q1 = limiar escolhido na validação.
- Frames excluídos: t = 0 e frames sem padrão (max B < 1e-6).

## Estrutura

- `configs/experimento.yaml`: toda a configuração do experimento.
- `src/data/`: leitura parcial dos HDF5 do Hugging Face, subconjuntos e conversão frame → nuvem de pontos.
- `src/ground_truth/`: candidatos (Etapa 2) e o GT definitivo.
- `src/preprocessing/`: P0–P10 (ajuste só no treino; filtros periódicos).
- `src/postprocessing/`: Q0–Q10 (periódicos; parâmetros escolhidos na validação).
- `src/models/`: PointNet, PointNet++ e DGCNN.
- `src/evaluation/`: IoU, Dice, Precision, Recall por trajetória, IC 95% por bootstrap e métricas temporais.
- `scripts/04`–`08`: dados → treino → avaliação → figuras → tabelas LaTeX; `scripts/rodar_tudo.sh` executa tudo.
- `results/{piloto,exp}/`: CSVs padronizados (`metricas_padrao.csv`), por regime, temporais e por trajetória.
- `docs/artigo/`: artigo (`main.tex` + `secoes/`), com tabelas e figuras geradas.

## Reprodução

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
PY=.venv/bin/python bash scripts/rodar_tudo.sh            # completo (GPU)
PY=.venv/bin/python bash scripts/rodar_tudo.sh --piloto   # piloto (CPU)
```

Artigo: compilar `docs/artigo/main.tex` no Overleaf (pdfLaTeX + BibTeX). Depois do experimento
completo, troque `\newcommand{\resdir}{piloto}` por `{exp}` em `main.tex`.
