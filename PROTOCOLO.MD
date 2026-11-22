
# Protocolo padrão do experimento — Segmentação de sistemas físicos 2D (The Well)

## 1. Objetivo

Este documento define um protocolo comum para a turma realizar experimentos de segmentação de campos 2D usando datasets da coleção **The Well**. A intenção é comparar arquiteturas de deep learning sob configurações controladas de dados, pré-processamento, pós-processamento e avaliação.

O protocolo busca:

- comparar modelos e técnicas de processamento de forma reproduzível;
- limitar o custo computacional e de armazenamento;
- preservar a proporção espacial dos campos;
- evitar ampliação (upscaling) de dados;
- manter divisões de treino, validação e teste sem vazamento entre trajetórias;
- documentar escolhas e parâmetros.

> **Importante:** as resoluções e os limites de armazenamento descritos aqui são regras operacionais propostas para o experimento da turma, não garantias de que toda estrutura física relevante será preservada. Faça uma inspeção e um piloto antes de congelar os subconjuntos.

---

## 2. Datasets selecionados

Foram selecionados 11 conjuntos 2D. Datasets 3D da lista original não fazem parte deste experimento.

| Dataset                           | Coordenadas  | Resolução original | Tamanho informado (GB) |
| --------------------------------- | ------------ | -------------------: | ---------------------: |
| `acoustic_scattering`           | Cartesian 2D |           256 × 256 |                   157* |
| `active_matter`                 | Cartesian 2D |           256 × 256 |                   51.3 |
| `euler_multi_quadrants`         | Cartesian 2D |           512 × 512 |                 5,170* |
| `gray_scott_reaction_diffusion` | Cartesian 2D |           128 × 128 |                    154 |
| `helmholtz_staircase`           | Cartesian 2D |          1024 × 256 |                     52 |
| `planetswe`                     | Angular 2D   |           256 × 512 |                    186 |
| `rayleigh_benard`               | Cartesian 2D |           512 × 128 |                    358 |
| `shear_flow`                    | Cartesian 2D |           128 × 256 |                    115 |
| `turbulent_radiative_layer_2D`  | Cartesian 2D |           128 × 384 |                    6.9 |
| `viscoelastic_instability`      | Cartesian 2D |           512 × 512 |                     66 |
| `viscoelastic_instability_v2`   | Cartesian 2D |           512 × 512 |                    260 |

\* O valor de `acoustic_scattering` é aproximado, usando `acoustic_discontinuous` (157 GB) como referência da lista fornecida. O valor de Euler (5,170 GB) é referente ao dataset informado e deve ser confirmado para a variante efetivamente utilizada. Verifique também a versão e os metadados de cada arquivo antes de baixar/processar.

---

## 3. Padrão de resolução espacial

### Regra

- **Teto da maior dimensão: 128 pixels.**
- Reduzir os campos cuja dimensão maior exceda 128 px.
- **Não realizar upscaling:** nenhuma dimensão espacial será aumentada.
- Preservar a proporção do campo.
- Não forçar todos os campos a serem quadrados.
- Usar uma reamostragem apropriada para campos numéricos científicos e validar visualmente/quantitativamente a perda de estruturas.

### Resoluções propostas após redução

| Dataset                           |    Original |       Resolução de trabalho proposta |
| --------------------------------- | ----------: | -------------------------------------: |
| `acoustic_scattering`           |  256 × 256 |                             128 × 128 |
| `active_matter`                 |  256 × 256 |                             128 × 128 |
| `euler_multi_quadrants`         |  512 × 512 |                             128 × 128 |
| `gray_scott_reaction_diffusion` |  128 × 128 |           128 × 128 (sem alteração) |
| `helmholtz_staircase`           | 1024 × 256 |                              128 × 32 |
| `planetswe`                     |  256 × 512 |                              64 × 128 |
| `rayleigh_benard`               |  512 × 128 |                              128 × 32 |
| `shear_flow`                    |  128 × 256 |                              64 × 128 |
| `turbulent_radiative_layer_2D`  |  128 × 384 | 43 × 128 (aproximação proporcional) |
| `viscoelastic_instability`      |  512 × 512 |                             128 × 128 |
| `viscoelastic_instability_v2`   |  512 × 512 |                             128 × 128 |

A dimensão calculada de `turbulent_radiative_layer_2D` (aproximadamente 42.7 × 128) pode ser arredondada para 43 × 128. Se a implementação exigir dimensões múltiplas de 8 ou 16, faça padding, sem esticar o campo. Registre a dimensão efetiva e a máscara de validade.

### Reamostragem e máscaras

- Campos escalares: use um método de redução com antialiasing adequado.
- Campos vetoriais: reamostre cada componente espacialmente de forma consistente; não altere magnitude física sem justificativa.
- Ground truth/máscaras: use nearest-neighbor para não criar rótulos intermediários.
- Coordenadas angulares (`planetswe`): respeite periodicidade e geometria do domínio; não trate automaticamente como plano cartesiano convencional.
- Padding, quando necessário: preencher para formato de tensor exigido pelo modelo; não confundir padding com upscaling.
- Pixels de padding devem ser ignorados na loss e nas métricas sempre que possível.

**Recomendação de piloto:** compare pelo menos dois níveis de resolução (por exemplo, 128 e 256 px de dimensão máxima) em dois datasets, antes de consolidar a resolução final. A configuração deste protocolo continua sendo 128 px; o piloto serve para verificar se ela preserva estruturas relevantes.

---

## 4. Padronização do tamanho dos datasets (armazenamento)

O tamanho dos arquivos em GB é diferente da resolução dos frames. Para controlar armazenamento e volume de processamento, adotamos uma **meta de até 25 GB por dataset**.

### Regra de armazenamento

- Meta/teto do subconjunto: **25 GB por dataset**.
- Dataset original menor que 25 GB: manter completo, se viável.
- Dataset maior que 25 GB: construir subconjunto representativo, sem modificar os arquivos originais.
- O teto não é uma obrigação de usar exatamente 25 GB: o tamanho final pode ser menor se a seleção equilibrada de trajetórias e frames resultar nisso.
- Selecionar trajetórias e frames temporalmente distribuídos, em vez de simplesmente cortar o início do arquivo ou sortear frames indiscriminadamente.
- Preservar, tanto quanto possível, a diversidade de condições iniciais e a evolução temporal.
- Registrar índices/IDs das trajetórias e frames selecionados, método de amostragem, seed e tamanho final do subconjunto.

### Tabela de orçamento de dados

| Dataset                           | Original (GB) | Limite de trabalho (GB) | Estratégia                                       |
| --------------------------------- | ------------: | ----------------------: | ------------------------------------------------- |
| `acoustic_scattering`           |           157 |                      25 | Subamostragem representativa                      |
| `active_matter`                 |          51.3 |                      25 | Subamostragem representativa                      |
| `euler_multi_quadrants`         |        5,170* |                      25 | Piloto de suficiência antes de fixar a seleção |
| `gray_scott_reaction_diffusion` |           154 |                      25 | Subamostragem representativa                      |
| `helmholtz_staircase`           |            52 |                      25 | Subamostragem representativa                      |
| `planetswe`                     |           186 |                      25 | Subamostragem representativa                      |
| `rayleigh_benard`               |           358 |                      25 | Subamostragem representativa                      |
| `shear_flow`                    |           115 |                      25 | Subamostragem representativa                      |
| `turbulent_radiative_layer_2D`  |           6.9 |          6.9 (completo) | Manter integralmente                              |
| `viscoelastic_instability`      |            66 |                      25 | Subamostragem representativa                      |
| `viscoelastic_instability_v2`   |           260 |                      25 | Subamostragem representativa                      |

**Estimativa do total:** aproximadamente **256.9 GB** (incluindo o dataset de 6.9 GB mantido completo), se todos os outros subconjuntos forem limitados a 25 GB. Este valor é um orçamento aproximado, não a promessa do tamanho exato dos arquivos após seleção/exportação.

### Amostragem recomendada

1. Respeite primeiro os splits oficiais por trajetória, se fornecidos.
2. Nunca distribua frames da mesma trajetória entre treino, validação e teste quando isso causar vazamento temporal.
3. Se for necessário reduzir, faça a seleção dentro de cada split, preservando sua finalidade.
4. Distribua a seleção ao longo da trajetória (início, meio e fim), em vez de selecionar apenas frames consecutivos.
5. Mantenha os conjuntos de validação e teste fixos e documentados.
6. Use a mesma seleção de dados para todas as arquiteturas e condições de processamento.

Para datasets muito grandes (especialmente Euler), não corte automaticamente para 25 GB sem avaliar a diversidade de trajetórias e estados físicos. Um piloto de 10, 25 e 50 GB pode indicar se o desempenho de validação estabiliza.

---

## 5. Ground truth (máscaras de referência)

Cada dataset deve ter uma definição de classe/segmentação **fisicamente justificável e documentada**. O ground truth pode ser diferente entre fenômenos se a tarefa física for diferente, mas a definição deve ser congelada antes da comparação dos modelos.

Procedimento:

1. Identificar as variáveis físicas disponíveis e seus metadados.
2. Definir o fenômeno ou regiões que serão segmentados.
3. Especificar regra de geração de máscara (por exemplo, critério físico, intervalo ou limiar), com justificativa.
4. Inspecionar amostras e validar as máscaras.
5. Congelar as máscaras e usar exatamente a mesma referência em todos os modelos e métodos.
6. Registrar eventuais limitações: máscaras derivadas por limiar avaliam reprodução do critério, não necessariamente descoberta independente do fenômeno.

Não presumir que a mesma variável ou limiar seja adequado a todos os datasets. A tarefa precisa ser definida conforme os campos disponíveis e a física de cada simulação.

---

## 6. Técnicas de pré-processamento padrão (10)

O baseline deve ser incluído: normalização padrão escolhida, sem filtro espacial adicional. Cada técnica abaixo deve ser comparada de maneira controlada. **Não aplicar todas em sequência.** Testar cada opção como uma condição separada (ou combinações previamente planejadas).

| ID  | Técnica                    | Grupo                  | Objetivo/observação                                                                             |
| --- | --------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------- |
| P1  | Z-score normalization       | Normalização         | Centralizar e escalar por média/desvio; estatísticas calculadas no treino                       |
| P2  | RMS normalization           | Normalização         | Escala pela magnitude RMS; útil para amplitudes                                                  |
| P3  | Robust scaling (median/IQR) | Normalização robusta | Menos sensível a valores extremos                                                                |
| P4  | Min-Max normalization       | Normalização         | Reescala para intervalo definido; atenção a extremos                                            |
| P5  | Percentile clipping         | Controle de extremos   | Limitar valores extremos com percentis definidos no treino; avaliar perda de informação física |
| P6  | Gaussian smoothing          | Filtro espacial        | Suavização; pode remover detalhes pequenos                                                      |
| P7  | Median filtering            | Filtro espacial        | Reduz ruído impulsivo; cuidado com estruturas finas                                              |
| P8  | Bilateral filtering         | Filtro espacial        | Suaviza preservando bordas aproximadas; parâmetros fixados na validação                        |
| P9  | Anisotropic diffusion       | Filtro espacial        | Suavização adaptativa que pode preservar fronteiras                                             |
| P10 | Wavelet denoising           | Multiescala            | Atenuação de ruído em escalas; registrar wavelet e limiar                                      |

### Regras para o pré-processamento

- Calcular estatísticas de normalização apenas no treino e reutilizá-las em validação/teste.
- Normalizar canais/campos físicos individualmente, salvo justificativa metodológica.
- Aplicar clipping somente após análise dos extremos e registrar percentis.
- Registrar parâmetros (sigma, kernel, iterações, wavelet, thresholds etc.).
- Não usar filtros que removam estruturas relevantes sem avaliar seu efeito.
- Manter baseline para separar o efeito de normalização do efeito de filtros espaciais.

---

## 7. Arquiteturas de segmentação

As três arquiteturas padrão:

1. **U-Net** — baseline de segmentação convolucional.
2. **SegFormer** — arquitetura baseada em Transformer com encoder hierárquico e decoder leve.
3. **Mask2Former** — arquitetura de segmentação baseada em máscaras e atenção.

Comparações devem usar:

- mesma divisão dos dados;
- mesmo ground truth;
- canais de entrada compatíveis;
- protocolo de treinamento comparável;
- seleção de hiperparâmetros usando validação;
- checkpoints e seeds registrados.

Quando a configuração de canais ou dimensão do tensor exigir adaptações, documente-as para cada arquitetura.

### Padronização do treinamento (número de épocas)

- **Número fixo de épocas:** **20 épocas** por execução de treinamento.
- **Critério de igualdade:** Todas as arquiteturas (U-Net, SegFormer e Mask2Former) e todas as condições de pré-processamento (P1–P10 e baseline) devem ser treinadas exatamente pelo mesmo limite de 20 épocas, garantindo comparação justa sob o mesmo orçamento computacional (especialmente em ambientes como Google Colab T4).
- **Seleção do melhor modelo (Checkpoint):** Não utilizar necessariamente os pesos da última época (época 20) caso ocorra *overfitting*. Ao final de cada época, avaliar a métrica principal (Dice ou IoU) no conjunto de **validação** e salvar o checkpoint correspondente à época de melhor desempenho (`best_model.pt`), registrando em qual época o pico ocorreu.
- **Early Stopping:** Caso seja adotado *early stopping* para evitar desperdício de GPU em casos de estagnação severa, o limite máximo permanece estritamente em **20 épocas**, e a paciência (*patience*) e a época de parada devem ser documentadas nos resultados.

---

## 8. Técnicas de pós-processamento padrão (5)

Aplicar às predições armazenadas, sempre mantendo também a saída original sem pós-processamento (baseline). Os parâmetros devem ser selecionados na validação e congelados antes de usar o teste.

| ID | Técnica                       | Objetivo                                                              | Parâmetros                      |
| -- | ------------------------------ | --------------------------------------------------------------------- | -------------------------------- |
| Q1 | Threshold optimization         | Converter probabilidades em máscara binária                         | Limiar                           |
| Q2 | Morphological opening          | Remover pequenas estruturas/ruído conforme a tarefa                  | Elemento estruturante            |
| Q3 | Morphological closing          | Corrigir pequenas lacunas/descontinuidades                            | Elemento estruturante            |
| Q4 | Connected-component filtering  | Filtrar componentes por tamanho ou critério definido                 | Área mínima/regra de seleção |
| Q5 | Temporal consistency filtering | Reduzir oscilações entre frames, quando temporalmente justificável | Janela/regra temporal            |

**Cuidados:** opening e closing podem eliminar ou unir estruturas reais. Component filtering só é válido quando a hipótese sobre tamanho/conectividade das regiões fizer sentido. Filtragem temporal não pode apagar mudanças físicas reais. Se frames forem avaliados isoladamente, Q5 não se aplica; marcar como não aplicável em vez de inventar um resultado.

---

## 9. Matriz experimental e número de execuções

Para os 11 datasets, 10 pré-processamentos, 3 arquiteturas e 5 pós-processamentos:

**11 × 10 × 3 × 5 = 1.650 combinações de avaliação.**

As predições de cada modelo podem ser armazenadas e reutilizadas para testar os pós-processamentos. Assim, não é necessário treinar um modelo novamente para cada método de pós-processamento.

O número de treinamentos principais (antes de repetições e baselines) é:

**11 × 10 × 3 = 330 treinamentos.**

Recomendação: incluir um baseline por dataset e arquitetura, com entrada de referência sem filtro espacial, além de registrar repetições com seeds diferentes quando o orçamento computacional permitir. Evite contar o pós-processamento como treinamento.

---

## 10. Métricas de avaliação

### Métricas principais

- **IoU (Intersection over Union):** sobreposição entre predição e referência.
- **Dice score:** similaridade entre predição e referência.

### Métricas complementares

- Precision
- Recall
- Latência de inferência
- Memória máxima utilizada
- Estabilidade temporal, quando aplicável
- Intervalos de confiança e variação entre seeds

Relatar métricas por dataset e por classe. Não interpretar diretamente scores de datasets com definições de ground truth diferentes como tarefas equivalentes; contextualizar a física e a regra da máscara.

---

## 11. Controle de qualidade e reprodutibilidade

Checklist mínimo:

- [ ] Versão/variante do dataset registrada.
- [ ] Split oficial ou split por trajetória registrado.
- [ ] IDs das trajetórias e frames selecionados salvos.
- [ ] Resolução original e resolução após redução registradas.
- [ ] Método de reamostragem registrado.
- [ ] Ground truth e critério de máscara congelados.
- [ ] Estatísticas de normalização calculadas apenas no treino.
- [ ] Parâmetros de pré e pós-processamento registrados.
- [ ] Arquitetura, hiperparâmetros e seed registrados.
- [ ] Checkpoints e predições salvos.
- [ ] Métricas calculadas ignorando padding.
- [ ] Teste mantido isolado até a avaliação final.

---

## 12. Organização sugerida dos resultados

Cada linha do arquivo de resultados deve representar uma configuração, com colunas como:

`dataset`, `dataset_variant`, `trajectory_split`, `resolution_original`, `resolution_used`, `sample_budget_gb`, `preprocessing_id`, `preprocessing_params`, `architecture`, `epochs`, `best_epoch`, `seed`, `postprocessing_id`, `postprocessing_params`, `IoU`, `Dice`, `Precision`, `Recall`, `latency_ms`, `peak_memory`, `notes`.

Armazenar também as máscaras, probabilidades previstas e parâmetros de cada configuração para permitir auditoria e repetição.

---

## 13. Pipeline resumido

```text
The Well (11 datasets 2D)
        |
        v
Inspecionar HDF5, campos e metadados
        |
        v
Split por trajetória (oficial, quando disponível)
        |
        v
Subconjunto por dataset: teto ~25 GB (menores mantidos completos)
        |
        v
Redução espacial proporcional: dimensão maior <= 128 px
        |        (sem upscaling; máscaras com nearest-neighbor)
        v
Definição e validação do ground truth
        |
        v
10 métodos de pré-processamento + baseline
        |
        v
U-Net / SegFormer / Mask2Former
        |
        v
Salvar probabilidades e predições
        |
        v
5 métodos de pós-processamento + baseline
        |
        v
IoU / Dice / Precision / Recall / temporal / custo
        |
        v
Análise por dataset e comparação entre sistemas físicos
```

---

## 14. Resumo das decisões padrão

| Item                     | Padrão da turma                                           |
| ------------------------ | ---------------------------------------------------------- |
| Datasets                 | 11 sistemas 2D listados neste documento                    |
| Resolução espacial     | Dimensão maior limitada a 128 px                          |
| Upcaling                 | Não permitido                                             |
| Proporção espacial     | Preservar                                                  |
| Padding                  | Permitido para requisitos do modelo; excluir das métricas |
| Armazenamento            | Teto-alvo de 25 GB/dataset                                 |
| Datasets menores         | Manter completos, se viável                               |
| Subamostragem            | Trajetórias e frames temporalmente distribuídos          |
| Ground truth             | Critério físico específico, documentado e congelado     |
| Pré-processamento       | 10 técnicas (P1–P10), testadas separadamente             |
| Arquiteturas             | U-Net, SegFormer, Mask2Former                              |
| Pós-processamento       | 5 técnicas (Q1–Q5), testadas na validação              |
| Épocas de treinamento   | 20 épocas fixas por execução                            |
| Métricas principais     | Dice e IoU                                                 |
| Métricas complementares | Precision, Recall, latência, memória, estabilidade       |
| Total de combinações   | 1.650 (11 × 10 × 3 × 5)                                 |
| Treinamentos principais  | 330 (antes de baselines/repetições)                      |

---

## 15. Nota final para a turma

Este protocolo é um ponto de partida comum para permitir comparações justas. Não substitui a análise física dos datasets: antes de executar, cada grupo deve confirmar variáveis disponíveis, geometria, periodicidade, qualidade do ground truth e adequação da redução de resolução. Se uma técnica não fizer sentido para um dataset, documentar a justificativa e marcá-la como não aplicável — não forçar uma transformação só para preencher a matriz.

A prioridade é produzir resultados **reprodutíveis, fisicamente interpretáveis e comparáveis**, e não apenas maximizar uma métrica de segmentação.
