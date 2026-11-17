# Segmentação 2D de padrões no Gray-Scott (The Well)

Trabalho P2 de Processamento Digital de Imagens (UFMA): segmentação dos padrões de Turing do dataset
[`gray_scott_reaction_diffusion`](https://polymathic-ai.org/the_well/datasets/gray_scott_reaction_diffusion/)
do [The Well](https://github.com/PolymathicAI/the_well), com 10 pré-processamentos, 3 arquiteturas
(U-Net, SegFormer, Mask2Former) e 10 pós-processamentos.

## Situação

| Etapa | Situação |
|---|---|
| 1. Auditoria do dataset | concluída ([relatório](docs/01_auditoria.md)) |
| 2. Ground truth | alternativas analisadas, aguarda definição do professor ([ficha](docs/02_ficha_ground_truth.md)) |
| 3 a 7. Software, piloto, experimentos, tabelas, artigo | pendentes |

Apresentação do andamento: [docs/apresentacao/](docs/apresentacao/) (Beamer + roteiro de fala).

## Estrutura

- `src/data/`: leitura parcial dos HDF5 direto do Hugging Face (sem baixar os 154 GB) e cache de amostras.
- `src/ground_truth/`: candidatos de ground truth (limiar de B, três classes, componentes conexos periódicos).
- `scripts/`: amostragem e análises das Etapas 1 e 2.
- `results/`: resultados numéricos (JSON).
- `docs/`: relatórios, figuras e apresentação.
- `historico_conversa_segmentacao_Gray-scoot-oficial.md`: histórico e instruções do projeto.

## Reprodução

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/01_fetch_audit_samples.py      # ~2,5 GB em data/processed/samples
MPLCONFIGDIR=.cache/mpl .venv/bin/python scripts/02_audit_analysis.py
MPLCONFIGDIR=.cache/mpl .venv/bin/python scripts/03_gt_candidates.py
```
