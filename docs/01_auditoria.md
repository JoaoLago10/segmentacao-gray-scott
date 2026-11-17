# Etapa 1: auditoria do dataset `gray_scott_reaction_diffusion`

Data: 10/11/2026. Fonte: repositório Hugging Face `polymathic-ai/gray_scott_reaction_diffusion`
(o mesmo usado por `the_well`), lido remotamente por requisições de intervalo de bytes.

Reprodução:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/01_fetch_audit_samples.py   # ~2,5 GB de amostras em data/processed/samples
MPLCONFIGDIR=.cache/mpl .venv/bin/python scripts/02_audit_analysis.py
```

Código: [src/data/remote.py](../src/data/remote.py) (metadados e leitura parcial),
[src/data/sampling.py](../src/data/sampling.py), [scripts/](../scripts/).
Resultados numéricos: [results/audit/auditoria.json](../results/audit/auditoria.json).

## 1. Estrutura dos arquivos

- 18 arquivos HDF5: um por regime em cada partição (`data/{train,valid,test}/gray_scott_reaction_diffusion_{regime}_F_{f}_k_{k}.hdf5`).
- Tamanho: 21,0 GB por regime no treino e 2,65 GB na validação e no teste (total de 157,9 GB, coerente com os 153,8 GB da documentação, que provavelmente usa GiB/1024).
- Campos: `t0_fields/A` e `t0_fields/B`, forma `(n_traj, 1001, 128, 128)`, `float32`, **sem compressão e contíguos**. Não há campos vetoriais nem tensoriais (`t1_fields` e `t2_fields` vazios).
- Escalares constantes: `scalars/F` e `scalars/k` (também nos atributos do arquivo).
- Dimensões: `time` de 0 a 10.000 s com passo de 10 s (1001 instantes); `x` e `y` com 128 pontos de -1 a 1 **incluindo as duas extremidades**.
- Contorno: `boundary_conditions/x_periodic` e `y_periodic`, ambos `PERIODIC`.
- Estatísticas oficiais de normalização (`stats.yaml`): média A = 0,7292, B = 0,0966; desvio A = 0,2399, B = 0,1237.
  Elas são calculadas sobre o conjunto inteiro; para o nosso pipeline, as estatísticas de normalização serão recalculadas só no treino (Seção 0.5, item 6).

## 2. Divisão oficial e distribuição dos regimes

Todos os seis regimes estão nas três partições, com o mesmo número de trajetórias:

- Treino: 160 trajetórias por regime (960 no total).
- Validação: 20 por regime (120).
- Teste: 20 por regime (120).

Total: 1200 trajetórias, como na documentação. A divisão é por trajetória, então não há frames da mesma simulação em partições diferentes.

## 3. Parâmetros f e k

Os atributos `F` e `k` de cada arquivo coincidem com a tabela da Seção 0.2 (Gliders 0,014/0,054; Bubbles 0,098/0,057; Maze 0,029/0,057; Worms 0,058/0,065; Spirals 0,018/0,051; Spots 0,030/0,062). Em `scalars/` estão em `float32` (por exemplo, 0,01400000043).

## 4. Trajetórias estacionárias

- Critério usado: desvio padrão espacial de A no último frame menor que 10⁻⁴.
- A lista detectada coincide **exatamente** com a lista oficial do README:
  - Gliders: 70 no treino, 11 na validação (7, 8, 10, 11, 12, 14 a 19) e 8 no teste (12 a 19).
  - Spirals: 4 no treino (97, 134, 147, 153), 1 na validação (14) e 1 no teste (11).
  - Nenhum outro regime tem trajetórias estacionárias.
- Nessas trajetórias, o estado final é A = 1 e B = 0 em todo o domínio, ou seja, **o padrão desaparece**.
- **O "time" do README é índice de frame**, não tempo físico. Ele coincide (diferença de 0 a 4 frames) com o primeiro índice em que A fica idêntico em float32 entre frames consecutivos (índices 121 a 155, ou seja, t = 1210 a 1550 s).
- Para a segmentação, o instante relevante é anterior: o máximo de B cai abaixo de 10⁻⁶ entre os índices 33 e 68 (t ≈ 330 a 680 s). A partir daí, qualquer máscara baseada em B fica vazia. Ver [estacionarias_std_A.png](figs/audit/estacionarias_std_A.png).
- Consequência: em Gliders, 70 das 160 trajetórias de treino (44%) e 11 das 20 de validação (55%) ficam sem padrão em ~95% do tempo. A decisão G5 (excluir, subamostrar ou tratar à parte) e a convenção G4 (máscaras vazias) são necessárias antes do treino.

## 5. Faixas de valores

Amostras: frames 0, 20, ..., 1000 (51 por trajetória) de todas as 20 trajetórias da validação e de 16 trajetórias do treino (0, 10, ..., 150) por regime.

- Não há NaN nem infinito.
- **t = 0 é atípico.** As condições iniciais do tipo série de Fourier têm A até 1,52 e B até -0,55. Para t > 0, os valores ficam em A ∈ [0,09; 1,0006] e B ∈ [-0,0018; 1,0], com B negativo apenas residual (mínimo de -0,0018, provável oscilação do método espectral).
- Consequências:
  - P5 (log) precisa de deslocamento explícito.
  - Min-Max (P2) é muito sensível ao frame t = 0.
  - Recomenda-se excluir t = 0 do treino e da avaliação (decisão a confirmar).
- Histogramas de B (treino, t > 0; ver [histogramas_A_B_treino.png](figs/audit/histogramas_A_B_treino.png)): são bimodais na maioria dos regimes, com um modo perto de 0 (fundo) e outro entre 0,25 e 0,45 (padrão). O formato e a posição do segundo modo variam entre os regimes; Maze tem um platô largo entre 0,05 e 0,3.
- Médias de B por regime (validação, todos os frames): Gliders 0,019; Bubbles 0,113; Maze 0,158; Worms 0,092; Spirals 0,088; Spots 0,098.

## 6. Condições iniciais

- O HDF5 não tem campo que indique o tipo de condição inicial.
- Pela inspeção visual ([condicoes_iniciais_maze_validacao.png](figs/audit/condicoes_iniciais_maze_validacao.png)) e pelo critério "A > 1 ou B < 0 em t = 0", as trajetórias com série de Fourier ficam na segunda metade de cada arquivo (por exemplo, validação 10 a 19 e treino a partir de ~85). Em Bubbles o critério não separa os dois tipos.
- As trajetórias estacionárias de Gliders estão concentradas nessa segunda metade. Isso indica que o colapso depende do tipo de condição inicial, mas o critério é heurístico e não está confirmado.

## 7. Dinâmica temporal

Ver [curvas_temporais_B_validacao.png](figs/audit/curvas_temporais_B_validacao.png), que mostra média e desvio espacial de B e variação RMS entre frames distantes 200 s (mediana e faixa de 10% a 90%, sem as trajetórias estacionárias).

- **Maze e Spots:** transiente curto e padrão praticamente estático depois de ~1.500 s (Maze) e ~3.500 s (Spots). A variação entre frames cai duas ordens de grandeza.
- **Bubbles e Worms:** evolução lenta até o final, com grande variabilidade entre trajetórias.
- **Spirals e Gliders:** padrão dinâmico permanente, com variação constante após ~200 s.
- Primeiro instante em que a variação mediana fica abaixo de 10% do pico: Gliders 200 s, Spirals 200 s, Maze 1.400 s, Bubbles 2.000 s, Worms 2.600 s, Spots 3.400 s.
- Consequências:
  - Os 1001 frames são muito redundantes nos regimes estáticos, o que favorece subamostragem temporal (D2).
  - Métricas temporais e Q10 são pouco informativas em Maze e Spots após o transiente, e muito informativas em Spirals e Gliders.

## 8. Rascunho da tabela visual por instante (Seção 0.9.5)

[tabela_visual_rascunho.png](figs/audit/tabela_visual_rascunho.png): trajetória 0 da validação de cada regime (nenhuma é estacionária). As linhas mostram A e B, e as colunas mostram t = 0, 200, 400, 1.000, 5.000 e 10.000 s.

Justificativa dos instantes, a partir das curvas da Seção 7:

- t = 0: condição inicial.
- t = 200 e 400 s: formação do padrão, onde a variação é máxima.
- t = 1.000 s: fim do transiente em Gliders, Spirals e Maze.
- t = 5.000 s: meio da simulação, que seria um candidato a "frame médio" se essa for a interpretação do professor.
- t = 10.000 s: estado final.

Faltam as colunas de GT e predição, que dependem das Etapas 2 e 5.

## 9. Divergências e problemas encontrados

1. A equação de B no README e na página oficial está escrita como `∂B/∂t = δ_B ΔB − AB² − (f+k)B`. O termo de reação correto do modelo de Gray-Scott é **+AB²**, como na Seção 0.2 do histórico. Trata-se de um erro tipográfico da documentação; o artigo deve usar +AB².
2. O "time" da lista de estacionárias é índice de frame (t físico = 10 × índice).
3. As estatísticas `stats.yaml` misturam todas as partições, então não devem ser usadas para normalizar.
4. As coordenadas `x` e `y` incluem -1 e 1, embora o domínio seja periódico. Isso só afeta a interpretação do espaçamento (2/127), não as imagens.
5. O pacote `the_well` não foi necessário. Este ambiente usa Python 3.14 e o leitor próprio ([src/data/remote.py](../src/data/remote.py)) lê os HDF5 diretamente, também com `h5py` em arquivos locais.

## 10. Custo de dados para as etapas seguintes

- Um frame tem 64 KiB por campo. Com subamostragem a cada 10 frames (101 frames por trajetória), o treino completo ocupa 960 × 101 × 2 × 64 KiB ≈ 12,7 GB, e a validação e o teste ≈ 1,6 GB cada.
- Pela rede (10 MB/s), isso leva de 25 a 30 minutos. Os 154 GB completos não são necessários.
