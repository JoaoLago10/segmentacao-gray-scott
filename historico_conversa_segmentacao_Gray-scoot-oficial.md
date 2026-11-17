# Histórico completo da conversa --- Segmentação de fluidos 2D no The Well

> Documento preparado para fornecer contexto ao agente do Cursor e dar
> continuidade ao trabalho.
>
> **Tema:** seleção de arquiteturas de Deep Learning, construção de
> ground truth, pré-processamento, pós-processamento e avaliação de
> segmentação em datasets 2D do The Well.
>
> **ATUALIZAÇÃO DA EQUIPE:** o projeto passou a ser baseado
> **exclusivamente no dataset `gray_scott_reaction_diffusion`** do The
> Well. A **Seção 0** abaixo prevalece sobre qualquer trecho posterior
> (em especial as Seções 3, 4 e 5) que mencione outros datasets.
>
> A **Seção 0.9** reúne as instruções do comunicado do professor
> (trabalho P2 de PDI), que também prevalecem.
>
> **Observação:** este documento reúne as perguntas e respostas desta
> conversa em formato Markdown. As referências citadas nas respostas
> originais foram preservadas como identificadores de citação; o agente
> deverá conferir as informações diretamente na documentação oficial do
> The Well e nos artigos originais antes de utilizá-las em um trabalho
> acadêmico.

------------------------------------------------------------------------

# 0. Decisão da equipe: projeto baseado no dataset Gray-Scott

> **Esta seção prevalece sobre o restante do documento.** O histórico
> das Seções 1 a 6 é mantido como registro da discussão e da metodologia
> geral (arquiteturas, pré/pós-processamento, métricas), mas toda a
> implementação, os experimentos e os resultados devem usar apenas o
> Gray-Scott.

## 0.1 Decisão

A equipe utilizará o dataset **`gray_scott_reaction_diffusion`** do The
Well como base única de todo o projeto:

-   Página oficial do dataset:
    https://polymathic-ai.org/the_well/datasets/gray_scott_reaction_diffusion/
-   Repositório do The Well: https://github.com/PolymathicAI/the_well
-   Artigo do The Well: https://arxiv.org/abs/2412.00568
-   Código gerador dos dados (MATLAB/Chebfun):
    https://github.com/danfortunato/spectral-gray-scott

Os demais datasets (`acoustic_scattering`, `active_matter`,
`euler_multi_quadrants`, `helmholtz_staircase`, `planetswe`,
`rayleigh_benard`, `shear_flow`, `turbulent_radiative_layer_2D`,
`viscoelastic_instability`) **ficam fora do escopo**.

## 0.2 Ficha do dataset (conforme a documentação oficial)

  -----------------------------------------------------------------------
  Item                    Informação
  ----------------------- -----------------------------------------------
  Fenômeno                Formação de padrões de Turing em equações de
                          reação-difusão de Gray-Scott

  Campos disponíveis      Concentrações das espécies químicas **A** e
                          **B** (dois campos escalares)

  Dimensão                1001 passos de tempo de imagens 128 × 128

  Trajetórias             6 conjuntos de parâmetros × 200 condições
                          iniciais = **1200 trajetórias**

  Tamanho estimado        153,8 GB (conjunto completo)

  Grade                   Uniforme, coordenadas cartesianas

  Condições de contorno   **Periódicas**

  Condições iniciais      Séries de Fourier aleatórias e aglomerados
                          aleatórios de gaussianas

  Passo de simulação      1 s; dados salvos a cada Δt = 10 s

  Intervalo de tempo      t = 0 a t = 10.000

  Domínio espacial        [−1, 1] × [−1, 1]

  Coeficientes de         δ_A = 2·10⁻⁵ e δ_B = 1·10⁻⁵ em todas as
  difusão                 simulações
  -----------------------------------------------------------------------

Equações:

\[ \frac{\partial A}{\partial t} = \delta_A \Delta A - AB^2 + f(1-A)
\]

\[ \frac{\partial B}{\partial t} = \delta_B \Delta B + AB^2 - (f+k)B \]

Os seis regimes (parâmetros de alimentação *f* e remoção *k*):

  -----------------------------------------------------------------------
  Regime                  f                       k
  ----------------------- ----------------------- -----------------------
  Gliders                 0,014                   0,054

  Bubbles                 0,098                   0,057

  Maze                    0,029                   0,057

  Worms                   0,058                   0,065

  Spirals                 0,018                   0,051

  Spots                   0,030                   0,062
  -----------------------------------------------------------------------

Os valores acima foram transcritos da página oficial; o agente deve
confirmá-los nos metadados dos arquivos HDF5 antes de usá-los.

## 0.3 O que muda no projeto

  -----------------------------------------------------------------------
  Aspecto                 Antes (10 datasets)     Agora (Gray-Scott)
  ----------------------- ----------------------- -----------------------
  Domínio da comparação   Dez sistemas físicos    Seis regimes de padrões
                                                  (f, k) de um mesmo
                                                  sistema

  Canais de entrada       Variam por dataset      2 canais: A e B

  Resolução               Variável                128 × 128 fixa

  Contorno                Variável                Periódico

  Combinações de          3.000                   **300** (10 × 3 × 10)
  avaliação                                       

  Treinamentos            300, mais baselines     **30**, mais 3
  principais                                      baselines (por
                                                  configuração e semente)

  Eixo de generalização   Entre datasets          **Entre regimes (f, k)**

  Dataset do piloto       `euler_multi_quadrants` `gray_scott_reaction_
                                                  diffusion`
  -----------------------------------------------------------------------

Consequências diretas:

-   **Não há campo de velocidade.** Vorticidade, Q-criterion e demais
    critérios baseados em velocidade (Seção 1.5 e 2.1) **não se aplicam**.
    O ground truth deve ser derivado das concentrações A e B.
-   **Não há análise "por dataset".** As tabelas e matrizes de calor
    passam a ser apresentadas **por regime**, além do agregado.
-   A recomendação sobre `viscoelastic_instability_v2` e a divisão em
    cinco datasets prioritários e cinco complementares **deixam de ser
    relevantes**.

## 0.4 Ground truth para o Gray-Scott (decisão pendente)

O Gray-Scott é um sistema de reação-difusão, não um escoamento
propriamente dito. Os objetos naturais de segmentação são as **regiões
ocupadas por padrões** (pontos, listras, labirintos, espirais, planadores).

Alternativas a serem avaliadas pelo agente e decididas pela equipe:

1.  **Binária por limiar de B:** padrão (B acima de τ) versus fundo.

    \[ M(x,y,t)=
    \begin{cases}
    1, & B(x,y,t)>\tau\\
    0, & B(x,y,t)\leq\tau
    \end{cases}
    \]

    O limiar τ deve ser justificado (por exemplo, método de Otsu ajustado
    somente nos dados de treino, ou critério físico) e pode precisar
    variar por regime, já que as amplitudes de B diferem entre padrões.
2.  **Três classes (fundo, padrão e interface):** a interface obtida por
    uma faixa em torno do módulo do gradiente de B. Útil para avaliar
    precisão de bordas.
3.  **Segmentação por estruturas conexas:** componentes conexos de B
    acima do limiar, por exemplo para contar e acompanhar pontos
    ("spots") e planadores ("gliders"). Considerar a periodicidade ao
    rotular componentes.

**Atenção ao vazamento de rótulo:** se a máscara for obtida por
limiarização de B e B também for entrada da rede, o modelo pode apenas
reaprender a regra de limiarização (alerta já feito na Seção 1.5). Opções
para tratar isso, a serem decididas:

-   Descrever o experimento como reprodução de uma segmentação definida
    por critério físico (válido, porém com conclusões limitadas).
-   Usar apenas **A** como entrada e segmentar com base em **B**, o que
    transforma o problema em inferir a estrutura de B a partir de A.
-   Definir máscaras por critério mais elaborado (gradiente, interface,
    estrutura conexa), que não seja reprodutível por simples limiar.

O agente **não deve escolher τ nem a definição final de classes sem
apresentar a justificativa e as alternativas à equipe.**

## 0.5 Cuidados específicos do Gray-Scott

1.  **Trajetórias estacionárias.** A documentação oficial avisa que, para
    certos valores de *f* e *k*, o sistema atinge um equilíbrio e o campo
    A se torna constante. Isso é descrito principalmente no regime
    *f* = 0,014, *k* = 0,054 (Gliders) e em algumas trajetórias de
    *f* = 0,018, *k* = 0,051 (Spirals), com instantes de estabilização
    em torno de 107 a 159. A página lista as trajetórias afetadas nos
    conjuntos de treino, validação e teste.
    -   Verificar se o valor "time" da página corresponde a índice de
        passo ou a tempo físico.
    -   Frames após a estabilização são praticamente idênticos: podem
        inflar métricas e enviesar o treino. Decidir entre excluí-los,
        subamostrá-los ou tratá-los como categoria própria.
    -   Máscaras vazias (sem padrão) tornam IoU e Dice indefinidos;
        definir convenção explícita.
    -   Q10 (filtro temporal) é trivial nesses trechos.
2.  **Condições de contorno periódicas.**
    -   Usar preenchimento circular (*wrap*) em filtros espaciais
        (P6–P10) e operações morfológicas (Q2–Q8).
    -   Rotular componentes conexos considerando que uma região que toca
        uma borda continua na borda oposta.
    -   Nas redes, considerar convoluções com padding circular.
3.  **Aumento de dados.** Como os campos são escalares e o domínio é
    periódico e isotrópico, translações circulares e rotações/reflexões
    de 90° preservam o significado físico. Aplicá-las igualmente à
    entrada e à máscara.
4.  **Resolução baixa (128 × 128).** SegFormer e Mask2Former reduzem a
    resolução internamente (em geral por fator 4 ou mais), gerando mapas
    pequenos. Avaliar reamostragem da entrada ou adaptação do
    decodificador, registrando a decisão.
5.  **Número de canais.** As redes pré-treinadas esperam 3 canais; aqui
    serão 2 (A e B) ou 1. Adaptar a primeira camada e documentar.
6.  **Pré-processamento (P1–P10).** As concentrações são não negativas,
    mas podem conter zeros; para P5 (logarítmica), tratar zeros com
    deslocamento explícito (por exemplo, log1p ou ε documentado).
    Ajustar estatísticas de normalização apenas no treino e **por
    canal** (A e B separadamente).
7.  **Divisão dos dados.** Usar a divisão oficial de treino, validação e
    teste, garantindo que nenhuma trajetória apareça em mais de uma
    partição. Confirmar nos arquivos como os 6 regimes estão
    distribuídos entre as partições.
8.  **Volume de dados.** São 153,8 GB no total. Planejar armazenamento,
    leitura em lote e possível subconjunto para o piloto.
9.  **Baseline do The Well.** A documentação reporta VRMSE no teste para
    a tarefa de previsão (FNO 0,1365; TFNO 0,3633; U-Net 0,2252;
    CNextU-net 0,1761). Esse benchmark é de **previsão de campos**, não
    de segmentação, e não deve ser comparado diretamente com IoU/Dice.

## 0.6 Experimento atualizado

Mantendo 10 pré-processamentos, 3 arquiteturas (U-Net, SegFormer,
Mask2Former) e 10 pós-processamentos, aplicados somente ao Gray-Scott:

\[ N=10\times3\times10=\boxed{300} \]

-   Treinamentos principais: 10 × 3 = 30, mais 3 baselines (33 por
    configuração de hiperparâmetros e por semente).
-   Os 10 pós-processamentos são aplicados às probabilidades salvas, sem
    retreinar.
-   Resultados reportados **por regime** (Gliders, Bubbles, Maze, Worms,
    Spirals, Spots) e agregados, por trajetória e por classe.
-   Métricas principais: IoU e Dice. Complementares: Precision, Recall,
    estabilidade temporal, latência e memória.

## 0.7 Fases do projeto (substituem as fases da Seção 3.6)

**Fase 1: experimento principal.** Treinar e avaliar com a divisão
oficial, contemplando os seis regimes. Identificar as melhores
combinações de pré-processamento, arquitetura e pós-processamento e como
variam entre regimes.

**Fase 2: generalização entre regimes.** Treinar em um subconjunto de
regimes e testar em regimes não vistos. Esse é o tipo de avaliação que a
própria documentação do dataset considera relevante: desempenho em
valores de (f, k) que produzem padrões qualitativamente diferentes.

A contribuição científica passa a ser: **como pré e pós-processamento
afetam a segmentação de diferentes padrões de reação-difusão e se os
resultados se generalizam entre regimes.**

## 0.8 Impacto nas orientações ao agente (Seção 5)

-   **5.1 (auditoria):** auditar apenas o Gray-Scott. Os itens 2 e 7
    (lista de dez datasets e `viscoelastic_instability_v2`) ficam
    dispensados. Acrescentar: confirmar nomes dos campos A e B, formato
    HDF5, divisões oficiais, distribuição dos seis regimes e lista de
    trajetórias estacionárias.
-   **5.2 (ground truth):** produzir **uma** ficha, com as alternativas
    da Seção 0.4, e aguardar decisão da equipe.
-   **5.3 (arquitetura de software):** remover a parametrização por
    dataset em `configs/datasets/`; usar configurações por regime e por
    experimento.
-   **5.4 (piloto):** usar o Gray-Scott, inicialmente com um ou dois
    regimes, e depois ampliar para os seis.
-   **5.5 (escalabilidade):** substituir a expansão para outros
    datasets por expansão para todos os regimes, repetições com
    sementes diferentes e a Fase 2 (generalização entre regimes).

## 0.9 Comunicado do professor: trabalho P2 de PDI (instruções oficiais)

> **Estas instruções vêm do professor e prevalecem sobre qualquer
> escolha anterior deste documento.** O link do ChatGPT citado no
> comunicado (https://chatgpt.com/share/6ad901a5-776c-83e9-aa93-f7e4e6db41be)
> corresponde ao histórico de conversa registrado neste arquivo.

### 0.9.1 Regra geral e motivação

-   **Motivação central:** onde estão as estruturas de interesse nos
    fluidos e como elas evoluem ao longo do tempo?
-   **O experimento é o mesmo para todos os alunos.** O que muda é o
    fluido e, consequentemente, o ground truth.
-   Cada aluno escolhe um fluido e **nenhum fluido pode se repetir**
    entre os alunos. O fluido da nossa equipe é o
    `gray_scott_reaction_diffusion`.
-   Basta seguir o pipeline do prompt (as Seções 1 a 3 deste arquivo).

### 0.9.2 Escopo: o que ficou de fora

-   **Desconsiderar o uso do SAM / SAM 2** (Seção 1.6 e referência na
    Seção 6). Permanecem as três arquiteturas: U-Net, SegFormer e
    Mask2Former.

### 0.9.3 Métricas

-   Aplicar o tópico **"2. Métricas: o que eu utilizaria no
    artigo"** do chat original, que corresponde à Seção 2.8 deste
    arquivo: IoU, Dice, Precision, Recall, métricas temporais, custo
    computacional (latência e memória) e intervalo de confiança de 95%.
-   Reportar também por classe e por trajetória, conforme a Seção 5.6.

### 0.9.4 Ground truth: frame médio que caracterize o fluido

O professor explicou em sala que o ground truth deve ser **uma imagem em
um frame médio que caracterize o fluido**.

> **Pontos a confirmar com o professor antes de implementar** (o agente
> não deve decidir sozinho):
>
> 1.  O que é "frame médio": um instante intermediário da simulação, ou
>     uma imagem média de vários instantes?
> 2.  No Gray-Scott há **seis regimes** (Gliders, Bubbles, Maze, Worms,
>     Spirals e Spots) com padrões muito diferentes. O ground truth será
>     um por regime, ou um único para o fluido?
> 3.  Como o GT único será usado no treino e na avaliação: máscara de
>     referência aplicada a todos os frames, ou apenas avaliação em
>     frames selecionados?
> 4.  Compatibilidade com a motivação "como as estruturas evoluem ao
>     longo do tempo" e com as métricas temporais da Seção 0.9.3.
>
> Enquanto não houver resposta, manter as alternativas de ground truth
> da Seção 0.4 como propostas.

### 0.9.5 Tabela visual do fluido por instante

Preparar uma **tabela visual com o fluido em diferentes instantes**, por
exemplo: t = 0, t = 10, t = ... .

Proposta para o Gray-Scott:

-   Os dados são salvos a cada Δt = 10 s, de t = 0 a t = 10.000 (1001
    instantes). Escolher instantes que mostrem a evolução do padrão
    (por exemplo, início, transiente e regime final) e **registrar quais
    foram escolhidos e por quê**.
-   Colunas sugeridas: instante, campo A, campo B e, quando houver,
    ground truth e predição.
-   Se possível, uma linha de exemplo por regime.
-   Cuidado com trajetórias estacionárias (Seção 0.5): escolher
    trajetórias que evoluem, ou indicar explicitamente quando o campo
    estabilizou.

### 0.9.6 Tabela de resultados: quatro condições

Preparar uma tabela de resultados, com as métricas, cruzando os
resultados do fluido nas quatro condições:

  -----------------------------------------------------------------------
  Condição                Descrição               Observação
  ----------------------- ----------------------- -----------------------
  Sem pré e sem pós       Dados originais e       Baseline
                          predição da rede com
                          binarização padrão

  Somente pré             P1 a P10, sem           Efeito isolado do
                          pós-processamento       pré-processamento

  Somente pós             Dados originais, com Q1 Efeito isolado do
                          a Q10                   pós-processamento

  Pré e pós               P1 a P10 combinados com Combinação completa
                          Q1 a Q10                (Seção 0.6)
  -----------------------------------------------------------------------

Pontos a definir:

-   O que significa "sem pré": dados originais ou apenas a normalização
    padrão da condição de referência (Seção 2.6).
-   O que significa "sem pós": como a rede produz probabilidades, é
    preciso ao menos uma binarização. Sugestão: limiar fixo documentado
    (por exemplo, 0,5), e Q1 (limiarização com limiar escolhido na
    validação) tratado como pós-processamento.
-   Todas as condições devem usar as mesmas arquiteturas, o mesmo teste
    e o mesmo ground truth.

### 0.9.7 Padronização das tabelas entre os alunos

Quando o artigo for publicado, as tabelas de resultados de todos os
alunos serão **juntadas em uma só**. Por isso, o formato deve ser
idêntico entre os alunos. Proposta de colunas (a confirmar com a turma):

  ----------------------------------------------------------------------
  Coluna                         Conteúdo
  ------------------------------ ---------------------------------------
  Fluido                         Nome do dataset

  Condição                       Sem pré e pós, só pré, só pós ou pré e
                                 pós

  Pré-processamento              ID (P1 a P10) ou "nenhum"

  Arquitetura                    U-Net, SegFormer ou Mask2Former

  Pós-processamento              ID (Q1 a Q10) ou "nenhum"

  IoU, Dice, Precision, Recall   Média e desvio ou IC de 95%

  Latência e memória             Unidades fixas (ms e MB)
  ----------------------------------------------------------------------

Regras: mesma ordem de colunas, mesmos identificadores (P1 a P10, Q1 a
Q10), mesmo número de casas decimais e mesma unidade. Salvar os
resultados também em CSV com esse esquema, para facilitar a junção.

### 0.9.8 Artigo

-   Usar o template **Elsevier Results in Engineering** (SciSpace):
    https://scispace.com/templates/elsevier-results-in-engineering-template-i4k529prb67ik4l
-   **Não economizar na fundamentação teórica** de:
    -   **o fluido:** as equações de Gray-Scott, o papel de *f* e *k*, os
        padrões de Turing e os seis regimes;
    -   **as arquiteturas:** U-Net, SegFormer e Mask2Former;
    -   **as métricas de avaliação:** IoU, Dice, Precision, Recall,
        métricas temporais e computacionais;
    -   como complemento, os pré e pós-processamentos usados.

### 0.9.9 Entregas adicionais para o agente

Além da auditoria e do piloto (Seção 5):

1.  Tabela visual por instante (Seção 0.9.5).
2.  Tabela de resultados nas quatro condições, no formato padronizado
    (Seções 0.9.6 e 0.9.7).
3.  Texto de fundamentação teórica para o artigo (Seção 0.9.8).
4.  Lista dos pontos em aberto da Seção 0.9.4 para o professor.

------------------------------------------------------------------------

# 1. Pergunta inicial: classificação ou segmentação?

**Usuário**

Para https://polymathic-ai.org/the_well/ quero utilizar 3 arquiteturas
de deep learning, seria melhor classificar os fluidos 2D ou segmentar?
Se sim, quais arquiteturas (de preferência mais atualizadas)?

**Assistente**

## Deep Learning para fluidos 2D no The Well: classificação ou segmentação?

Para desenvolver um experimento científico utilizando o dataset The
Well, a recomendação inicial foi priorizar a **segmentação dos fluidos
2D**, desde que o objetivo seja identificar regiões de interesse,
estruturas e padrões espaciais presentes nas simulações.

A diferença entre os problemas:

-   **Classificação:** determina a qual categoria pertence uma imagem ou
    quadro da simulação. Exemplo: classificar diferentes regimes de
    escoamento ou identificar padrões de comportamento de um fluido.
-   **Segmentação:** classifica cada pixel de uma imagem, permitindo
    delimitar regiões como fluido, obstáculos e estruturas de interesse.
    Dependendo das variáveis disponíveis, também pode identificar
    regiões com diferentes propriedades físicas.

O The Well é interessante porque contém simulações científicas de
diferentes sistemas físicos, incluindo dinâmicas de fluidos. Os dados
podem conter campos físicos que vão além de imagens RGB convencionais,
como velocidade, pressão e densidade, dependendo do conjunto de dados
escolhido.

Isso permite formular um experimento que vá além da classificação de
imagens: utilizar campos físicos bidimensionais como entrada de redes
neurais e investigar sua capacidade de reconhecer estruturas espaciais.

### 1.1 Classificação ou segmentação: qual é melhor?

  -----------------------------------------------------------------------
  Critério                Classificação 2D        Segmentação 2D
  ----------------------- ----------------------- -----------------------
  Objetivo                Identificar a categoria Identificar a classe de
                          de uma imagem ou quadro cada pixel

  Saída                   Uma classe por imagem   Um mapa de classes por
                                                  pixel

  Exemplo                 Classificar um regime   Delimitar regiões de
                          de escoamento           interesse no escoamento

  Ground truth            Rótulos por imagem ou   Máscaras de segmentação
                          sequência               

  Métricas principais     Acurácia, precisão,     IoU, Dice, precisão e
                          recall e F1-score       recall por classe

  Análise espacial        Limitada                Detalhada

  Análise de estruturas   Indireta                Direta

  Aplicação à             Possível, mas indireta  Mais direta quando as
  reconstrução 3D                                 máscaras representam
                                                  estruturas relevantes

  Complexidade de         Geralmente menor        Geralmente maior
  anotação                                        
  -----------------------------------------------------------------------

A recomendação inicial foi escolher segmentação se a pergunta científica
for: **onde estão as estruturas de interesse nos fluidos e como elas
evoluem ao longo do tempo?**

Se a pergunta for qual o tipo ou regime do fluido representado em cada
simulação, classificação pode ser mais apropriada.

Também foi apresentada uma terceira possibilidade: se o objetivo
principal for prever a evolução futura dos campos físicos, como
velocidade, pressão e densidade, o problema mais adequado poderá ser
previsão de campos físicos, usando arquiteturas como Fourier Neural
Operator (FNO), em vez de classificação ou segmentação.

### 1.2 Três arquiteturas sugeridas

A seleção inicial foi:

1.  U-Net --- referência convolucional.
2.  SegFormer --- arquitetura baseada em Transformers.
3.  Mask2Former --- arquitetura baseada em Transformers para diferentes
    modalidades de segmentação.

#### U-Net

Referência: https://arxiv.org/abs/1505.04597

A U-Net é uma arquitetura convolucional com codificador e decodificador,
conectados por *skip connections* que preservam informações espaciais.

Os campos físicos poderiam ser utilizados como canais de entrada. Por
exemplo, caso a simulação disponibilize densidade, pressão e componentes
da velocidade:

\[ X_t(x,y) =
```{=tex}
\begin{bmatrix}
\rho(x,y,t)\\
p(x,y,t)\\
u_x(x,y,t)\\
u_y(x,y,t)
\end{bmatrix}
```
\]

A rede produziria um mapa de segmentação:

\[ f\_{`\theta`{=tex}}(X_t) `\rightarrow `{=tex}M_t \]

**Vantagens:** - Boa preservação de detalhes espaciais. - Implementação
relativamente simples em PyTorch. - Adequada como baseline
experimental. - Permite testar diferentes combinações de campos físicos.

**Limitação:** a U-Net convencional não modela explicitamente
dependências temporais entre quadros. Para estudar a evolução dos
fluidos, seria necessário acrescentar informação temporal à entrada ou
utilizar uma variante temporal.

#### SegFormer

Referências:

-   Artigo:
    https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html
-   Implementação: https://github.com/NVlabs/SegFormer

O SegFormer combina um codificador hierárquico baseado em Transformers
com um decodificador leve para segmentação semântica.

Sua principal característica é combinar informações espaciais em
diferentes escalas com mecanismos de atenção. Isso é interessante para
simulações de fluidos, nas quais podem existir estruturas pequenas,
regiões extensas, interfaces e padrões espaciais que dependem do
contexto global.

A entrada pode ser representada por:

\[ X_t `\in `{=tex}`\mathbb{R}`{=tex}\^{H `\times `{=tex}W
`\times `{=tex}C} \]

A saída para segmentação com K classes:

\[ `\hat{M}`{=tex}\_t `\in `{=tex}`\mathbb{R}`{=tex}\^{H
`\times `{=tex}W `\times `{=tex}K} \]

**Vantagens:** - Modelagem de relações espaciais de longo alcance. -
Extração de características em diferentes escalas. - Variantes de
diferentes tamanhos. - Boa opção para estudar eficiência computacional e
qualidade da segmentação.

**Limitação:** modelos pré-treinados comuns foram treinados com imagens
naturais. Para campos físicos científicos, pode ser necessário adaptar a
primeira camada, a quantidade de canais e o treinamento.

#### Mask2Former

Referências:

-   Artigo:
    https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html
-   Implementação: https://github.com/facebookresearch/Mask2Former

O Mask2Former foi desenvolvido para segmentação semântica, de instâncias
e panóptica. Seu mecanismo de atenção utiliza máscaras previstas para
restringir a atenção a regiões relevantes.

**Vantagens:** - Suporte a diferentes tarefas de segmentação. -
Capacidade de representar regiões por máscaras. - Possibilidade de
investigar estruturas com formas complexas. - Comparação interessante
com U-Net e SegFormer.

**Limitações:** - Implementação e treinamento mais complexos. - Maior
custo computacional em determinadas configurações. - Necessidade de
adaptar o modelo ao formato dos dados científicos. - Não resolve, por si
só, a criação de máscaras de referência.

Foi observado que o repositório oficial havia sido arquivado em janeiro
de 2025, recomendando-se verificar a compatibilidade com versões atuais
de PyTorch e bibliotecas de visão computacional.

### 1.3 Comparação das arquiteturas

  -----------------------------------------------------------------------
  Característica    U-Net             SegFormer         Mask2Former
  ----------------- ----------------- ----------------- -----------------
  Família           CNN               Transformer       Transformer com
                                      hierárquico       máscaras

  Tipo de           Semântica         Semântica         Semântica,
  segmentação                                           instâncias e
                                                        panóptica

  Preservação de    Boa               Boa, dependendo   Depende do
  detalhes                            da configuração   backbone e da
                                                        configuração

  Contexto espacial Limitado na       Forte capacidade  Forte capacidade
  global            versão            de modelagem      de modelagem
                    convencional                        

  Adaptação a       Relativamente     Requer adaptação  Requer adaptação
  campos físicos    simples           dos canais e      dos canais e
                                      treinamento       treinamento

  Complexidade      Baixa a moderada  Moderada          Alta

  Papel             Baseline          Transformer       Segmentação
                                      eficiente         baseada em
                                                        máscaras
  -----------------------------------------------------------------------

A qualidade real depende do dataset, das classes, dos rótulos, da
resolução e da configuração de treinamento. Não se deve afirmar
antecipadamente que Mask2Former terá necessariamente o melhor
desempenho.

### 1.4 Datasets inicialmente considerados

Foram citados como possibilidades:

  ---------------------------------------------------------------------------------
  Dataset                      Resolução espacial Característica   Possível
                                                                   aplicação
  ------------------------- --------------------- ---------------- ----------------
  `active_matter`                       256 × 256 Sistema de       Identificação de
                                                  matéria ativa    estruturas e
                                                                   regiões de
                                                                   interesse

  `euler_multi_quadrants`               512 × 512 Dinâmica de      Segmentação de
                                                  fluidos          regiões
                                                                   definidas por
                                                                   critérios
                                                                   físicos

  `rayleigh_benard`                     512 × 128 Convecção        Identificação de
                                                                   estruturas
                                                                   convectivas
  ---------------------------------------------------------------------------------

A escolha inicial sugerida foi `euler_multi_quadrants`, caso o objetivo
fosse desenvolver uma metodologia de segmentação de campos de fluidos e
houvesse uma definição física adequada para as classes.

Foi ressaltado que não se deve presumir que o dataset já venha
acompanhado de máscaras de segmentação prontas.

### 1.5 Criação do ground truth: possibilidades iniciais

Foram apresentadas três estratégias:

1.  **Segmentação baseada em critérios físicos:** definir classes por
    limiares ou combinações de variáveis físicas.
2.  **Segmentação de estruturas físicas:** identificar regiões de
    interesse, como estruturas de vorticidade, regiões de alta/baixa
    densidade ou estruturas convectivas.
3.  **Anotação especializada:** criar máscaras manualmente ou com
    ferramentas de anotação, usando critérios documentados e revisão
    especializada.

Exemplo de máscara por limiar de densidade:

\[ M(x,y,t) =
```{=tex}
\begin{cases}
1, & \rho(x,y,t) > \tau\\
0, & \rho(x,y,t) \leq \tau
\end{cases}
```
\]

Foi alertado que, se a mesma densidade for usada como entrada e para
criar a máscara, a rede poderá apenas aprender a reproduzir a regra de
limiarização. Isso pode ser válido como experimento metodológico, mas
não demonstra necessariamente descoberta de novas estruturas.

Também foi apresentada a possibilidade de calcular vorticidade em
escoamentos 2D:

\[ `\omega`{=tex}(x,y,t)= `\frac{\partial v}{\partial x}`{=tex} -
`\frac{\partial u}{\partial y}`{=tex} \]

E definir uma máscara de regiões de elevada magnitude de vorticidade:

\[ M(x,y,t)=
```{=tex}
\begin{cases}
1, & |\omega(x,y,t)|>\tau_\omega\\
0, & |\omega(x,y,t)|\leq\tau_\omega
\end{cases}
```
\]

Uma região de alta vorticidade não corresponde automaticamente a um
vórtice completo; critérios adicionais podem ser necessários, como
Q-criterion ou discriminante do gradiente de velocidade.

### 1.6 Possibilidade mais recente: SAM 2

> **Fora do escopo:** por orientação do professor (Seção 0.9.2), o SAM
> deve ser desconsiderado no trabalho. Mantido apenas como registro.

Também foi sugerido considerar o Segment Anything Model 2 (SAM 2) como
alternativa mais recente:

-   Artigo: https://arxiv.org/abs/2408.00714

O SAM 2 foi desenvolvido para segmentação de objetos em imagens e
vídeos. Seu uso em simulações científicas exige adaptação e avaliação
cuidadosa, pois não foi concebido especificamente para segmentar campos
físicos do The Well.

A recomendação original foi manter U-Net, SegFormer e Mask2Former para
uma comparação de segmentação semântica supervisionada mais direta,
considerando SAM 2 como possível linha complementar para acompanhar
estruturas em sequências.

------------------------------------------------------------------------

# 2. Pergunta: como desenvolver o ground truth, usar dez pré-processamentos e dez pós-processamentos?

**Usuário**

Perfeito, após isso optei pela tarefa SEGMENTAÇÃO, sendo assim:

a)  como desenvolver o ground truth por fluido 2D\
b)  Quero aplicar 10 métodos de pré-processamento e 10 de
    pós-processamento, como seria uma tabela de resultado cruzando as
    possibilidades junto às arquiteturas de segmentação já definidas? E
    um pipeline.

**Assistente**

# Metodologia experimental: segmentação de fluidos 2D no The Well

A estrutura proposta foi dividida em três etapas:

1.  Construção do ground truth: definir e gerar máscaras de referência
    para cada simulação.
2.  Experimentos de pré-processamento e segmentação: avaliar dez métodos
    de pré-processamento, três arquiteturas e dez métodos de
    pós-processamento.
3.  Avaliação quantitativa: comparar combinações usando métricas de
    segmentação, custo computacional e estabilidade temporal.

Foi ressaltado que o ground truth não deve ser definido apenas para
facilitar o treinamento: ele precisa representar o fenômeno físico que
se deseja segmentar.

## 2.1 Construção do ground truth

### Definir o que será segmentado

Para um primeiro experimento, foi recomendada segmentação binária:

  ------------------------------------------------------------------------
  Classe                                      Código Significado
  --------------------- ---------------------------- ---------------------
  Fundo ou região                                  0 Região que não
  externa                                            pertence à estrutura
                                                     de interesse

  Região de interesse                              1 Região que atende ao
                                                     critério físico
                                                     definido
  ------------------------------------------------------------------------

Ressalva: em uma simulação de fluido que ocupa todo o domínio
computacional, todos os pixels podem representar estados válidos do
fluido. Nesse caso, não existe necessariamente um fundo externo. É
necessário definir um objeto de segmentação coerente com o dataset,
como: - Regiões de alta densidade. - Regiões de elevada vorticidade. -
Estruturas coerentes de um escoamento. - Regiões de uma interface entre
fluidos. - Regiões que atendem a um critério físico específico.

### Estratégia 1 --- Ground truth derivado dos campos físicos

Exemplo usando densidade:

\[ M(x,y,t)=
```{=tex}
\begin{cases}
1, & \rho(x,y,t)>\tau\\
0, & \rho(x,y,t)\leq\tau
\end{cases}
```
\]

O limiar deve ser justificado pela distribuição dos valores e pela
interpretação física da simulação.

Se a mesma densidade for usada na entrada e para criar a máscara, a rede
poderá aprender somente a regra de limiarização. Isso não invalida o
experimento, mas limita as conclusões.

### Estratégia 2 --- Ground truth baseado em propriedades físicas

Exemplo usando vorticidade:

\[ `\omega`{=tex}(x,y,t)= `\frac{\partial v}{\partial x}`{=tex} -
`\frac{\partial u}{\partial y}`{=tex} \]

Máscara:

\[ M(x,y,t)=
```{=tex}
\begin{cases}
1, & |\omega(x,y,t)|>\tau_\omega\\
0, & |\omega(x,y,t)|\leq\tau_\omega
\end{cases}
```
\]

A máscara pode identificar regiões de intensa rotação local. Para
identificar vórtices completos, podem ser necessários critérios
adicionais.

### Estratégia 3 --- Anotação especializada

Procedimento: 1. Selecionar quadros representativos. 2. Gerar
visualizações padronizadas dos campos físicos. 3. Anotar as regiões de
interesse. 4. Revisar as máscaras com critérios definidos. 5. Medir
concordância entre anotadores, quando houver mais de um. 6. Separar os
dados de treinamento, validação e teste.

A anotação de uma imagem colorida não deve ser confundida
automaticamente com a identificação da estrutura física original. A
visualização precisa preservar uma relação bem definida com os dados
numéricos.

## 2.2 Organização dos dados

O The Well organiza os dados em arquivos HDF5, com campos físicos em
grades espaciais e temporais. Os campos podem ser escalares, vetoriais e
tensoriais.

Uma amostra pode ser representada por:

\[ X_t `\in `{=tex}`\mathbb{R}`{=tex}\^{H `\times `{=tex}W
`\times `{=tex}C} \]

A máscara de referência:

\[ M_t `\in `{=tex}{0,1}\^{H `\times `{=tex}W} \]

para segmentação binária, ou:

\[ M_t `\in `{=tex}{0,1,`\ldots`{=tex},K-1}\^{H `\times `{=tex}W} \]

para segmentação multiclasse.

  -----------------------------------------------------------------------
  Elemento                Conteúdo                Finalidade
  ----------------------- ----------------------- -----------------------
  Entrada original        Campos físicos da       Preservar dados
                          simulação               científicos

  Entrada pré-processada  Campos após             Avaliar o efeito do
                          transformação           pré-processamento

  Ground truth            Máscara de referência   Treinar e avaliar

  Predição                Probabilidades da rede  Representar classes
                                                  previstas

  Máscara pós-processada  Resultado após método   Avaliar efeito do
                                                  pós-processamento
  -----------------------------------------------------------------------

Transformações geométricas devem ser aplicadas consistentemente à
entrada e à máscara. Transformações de coordenadas também podem exigir a
transformação dos componentes dos campos vetoriais, não apenas a rotação
visual da imagem.

## 2.3 Dez métodos de pré-processamento

A proposta foi separar transformações numéricas dos campos físicos e
filtros espaciais.

  -----------------------------------------------------------------------------------
  ID             Método          Categoria              Objetivo       Observação
  -------------- --------------- ---------------------- -------------- --------------
  P1             Normalização    Numérico               Centralizar e  Ajustar
                 Z-score                                padronizar     parâmetros
                                                        valores        apenas no
                                                                       treinamento

  P2             Normalização    Numérico               Reescalar      Sensível a
                 Min-Max                                valores        extremos

  P3             Normalização    Numérico               Reescalar por  Útil com
                 robusta                                mediana e      valores
                                                        intervalo      extremos
                                                        interquartil   

  P4             Percentile      Numérico               Reduzir        Registrar
                 clipping                               influência de  percentis
                                                        extremos       

  P5             Transformação   Numérico               Reduzir        Tratar zeros e
                 logarítmica                            assimetria     negativos

  P6             Filtro          Espacial               Atenuar ruído  Pode suavizar
                 Gaussiano                              de alta        interfaces
                                                        frequência     

  P7             Filtro de       Espacial               Reduzir ruído  Pode remover
                 mediana                                impulsivo      estruturas
                                                                       pequenas

  P8             Filtro          Espacial               Suavizar       Pode ser mais
                 bilateral                              preservando    caro
                                                        bordas         

  P9             Difusão         Espacial               Reduzir        Depende dos
                 anisotrópica                           variações      parâmetros
                                                        locais         
                                                        preservando    
                                                        estruturas     

  P10            Denoising por   Espacial/multiescala   Atenuar        Exige escolha
                 wavelets                               componentes    de wavelet e
                                                        considerados   limiar
                                                        ruído          
  -----------------------------------------------------------------------------------

P1--P5 transformam valores numéricos; P6--P10 alteram a estrutura
espacial.

Regras recomendadas: 1. Usar dados originais como baseline, sem filtro
espacial. 2. Aplicar uma transformação por experimento. 3. Treinar as
três arquiteturas sob condições comparáveis. 4. Manter o ground truth
fixo. 5. Registrar parâmetros. 6. Avaliar se a transformação melhora a
segmentação sem eliminar estruturas físicas.

As normalizações devem ser ajustadas apenas com dados de treinamento e
aplicadas sem recalibração à validação e ao teste.

A transformação logarítmica não deve ser aplicada indiscriminadamente a
variáveis que possam assumir valores negativos. A transformação deve
respeitar o domínio físico da variável.

O CLAHE foi citado como possível experimento complementar para mapas
visuais escalares, mas não como um dos dez métodos principais para
processar diretamente campos físicos multicanais.

## 2.4 Dez métodos de pós-processamento

Considere que o modelo produza:

\[ P(x,y,k) \]

Para segmentação binária, pode-se utilizar a probabilidade da classe
positiva (P_1(x,y)).

A máscara final é:

\[ `\hat{M}`{=tex}*{final}=g*{`\phi`{=tex}}(P) \]

Tabela proposta:

  ------------------------------------------------------------------------
  ID                Método            Objetivo           Principal cuidado
  ----------------- ----------------- ------------------ -----------------
  Q1                Limiarização de   Converter          Limiar
                    probabilidades    probabilidades em  selecionado na
                                      máscara binária    validação

  Q2                Abertura          Remover pequenas   Pode eliminar
                    morfológica       estruturas e       estruturas reais
                                      saliências         

  Q3                Fechamento        Fechar lacunas e   Pode unir regiões
                    morfológico       descontinuidades   distintas

  Q4                Abertura seguida  Reduzir ruído e    Depende do
                    de fechamento     corrigir lacunas   elemento
                                                         estruturante

  Q5                Remoção de        Eliminar regiões   Área mínima
                    componentes       abaixo de área     justificada
                    pequenos          mínima             

  Q6                Preenchimento de  Preencher regiões  Pode apagar
                    buracos           internas não       cavidades reais
                                      segmentadas        

  Q7                Seleção de        Selecionar         Maior componente
                    componentes       componentes por    nem sempre é
                    conexos           critério           correto

  Q8                Watershed         Separar regiões    Pode fragmentar
                                      que se encontram   excessivamente
                                      ou se sobrepõem    

  Q9                Conditional       Refinar            Exige
                    Random Fields     segmentação        implementação e
                    (CRF)             considerando       parâmetros
                                      relações entre     definidos
                                      pixels             

  Q10               Filtragem         Reduzir oscilações Pode apagar
                    temporal das      entre quadros      mudanças físicas
                    máscaras                             rápidas
  ------------------------------------------------------------------------

Os métodos Q2--Q8 são adequados a máscaras binárias ou multiclasse
convertidas em regiões, com estratégia explícita por classe. CRF requer
implementação compatível com a representação. Q10 exige uma sequência de
quadros.

Não se deve aplicar automaticamente todos os métodos a todas as
predições. O objetivo é descobrir quais são úteis.

## 2.5 Cruzamento das arquiteturas com pré e pós-processamento

Foram definidos: - 10 métodos de pré-processamento. - 3 arquiteturas. -
10 métodos de pós-processamento.

Número de combinações:

\[ N=10`\times3`{=tex}`\times10`{=tex}=`\boxed{300}`{=tex} \]

### Tabela principal

  ---------------------------------------------------------------------------------------------------------------------
  Experimento   Pré-processamento   Arquitetura   Pós-processamento        IoU     Dice   Precision   Recall   Latência
  ------------- ------------------- ------------- ------------------- -------- -------- ----------- -------- ----------
  E001          P1 --- Z-score      U-Net         Q1 --- Limiarização      ---      ---         ---      ---        ---

  E002          P1 --- Z-score      U-Net         Q2 --- Abertura          ---      ---         ---      ---        ---

  E003          P1 --- Z-score      U-Net         Q3 --- Fechamento        ---      ---         ---      ---        ---

  E004          P1 --- Z-score      U-Net         Q4 --- Abertura +        ---      ---         ---      ---        ---
                                                  fechamento                                                 

  E005          P1 --- Z-score      U-Net         Q5 --- Componentes       ---      ---         ---      ---        ---
                                                  pequenos                                                   

  E006          P1 --- Z-score      U-Net         Q6 ---                   ---      ---         ---      ---        ---
                                                  Preenchimento de                                           
                                                  buracos                                                    

  E007          P1 --- Z-score      U-Net         Q7 --- Componentes       ---      ---         ---      ---        ---
                                                  conexos                                                    

  E008          P1 --- Z-score      U-Net         Q8 --- Watershed         ---      ---         ---      ---        ---

  E009          P1 --- Z-score      U-Net         Q9 --- CRF               ---      ---         ---      ---        ---

  E010          P1 --- Z-score      U-Net         Q10 --- Filtro           ---      ---         ---      ---        ---
                                                  temporal                                                   

  E011--E100    P2--P10             U-Net         Q1--Q10                  ---      ---         ---      ---        ---

  E101--E200    P1--P10             SegFormer     Q1--Q10                  ---      ---         ---      ---        ---

  E201--E300    P1--P10             Mask2Former   Q1--Q10                  ---      ---         ---      ---        ---
  ---------------------------------------------------------------------------------------------------------------------

Os traços representam resultados ainda não medidos.

Para uma publicação, acrescentar: - Dataset e trajetória. - Semente
aleatória. - Parâmetros de pré-processamento. - Parâmetros de
pós-processamento. - Número de parâmetros do modelo. - Tempo de
treinamento. - Memória máxima. - Intervalo de confiança.

Separar tempo de inferência da rede do tempo total com
pós-processamento.

### Matrizes de calor

Além da tabela longa, foi sugerida uma matriz para cada arquitetura.
Linhas representam P1--P10 e colunas Q1--Q10. Cada célula apresenta, por
exemplo, o Dice médio.

Produzir uma matriz para U-Net, outra para SegFormer e outra para
Mask2Former.

Isso permite identificar se determinado pré-processamento beneficia
consistentemente as três arquiteturas ou se seu efeito depende do
modelo.

## 2.6 Redução do custo computacional

As 300 combinações não exigem 300 treinamentos independentes.

### Etapa 1 --- Treinar redes

Treinar cada arquitetura para cada pré-processamento:

\[ N\_{`\text{treinamentos}`{=tex}}=10`\times3`{=tex}=30 \]

Acrescentar uma condição de referência para cada arquitetura,
totalizando inicialmente 33 treinamentos por configuração de
hiperparâmetros e por semente.

A referência deve ser definida cuidadosamente: dados com normalização
padrão, mas sem filtro espacial, por exemplo.

### Etapa 2 --- Avaliar pós-processamento

Salvar probabilidades previstas no conjunto de validação e, depois, no
teste. Aplicar os dez pós-processamentos às predições salvas.

Assim, para os 30 modelos principais:

\[ 30`\times10`{=tex}=300 \]

combinações de avaliação, sem retreinar para cada pós-processamento.

Os pós-processamentos devem receber representação compatível com seu
funcionamento. Filtro temporal precisa de sequência; método morfológico
precisa de máscara ou estratégia por classe.

## 2.7 Pipeline experimental

1.  Carregar dados.
2.  Selecionar dataset.
3.  Ler campos físicos e metadados.
4.  Identificar trajetórias, instantes e dimensões.
5.  Verificar valores ausentes, extremos e condições de contorno.
6.  Dividir dados por trajetória.
7.  Construir e validar ground truth.
8.  Aplicar P1--P10 e condição de referência.
9.  Treinar U-Net, SegFormer e Mask2Former.
10. Salvar probabilidades.
11. Aplicar Q1--Q10.
12. Avaliar.
13. Analisar matrizes e resultados.

Foi recomendado utilizar a divisão oficial de treinamento, validação e
teste do dataset, quando disponível, evitando que quadros temporalmente
próximos da mesma trajetória apareçam simultaneamente em treino e teste.

## 2.8 Métricas

### Intersection over Union

\[ IoU=`\frac{TP}{TP+FP+FN}`{=tex} \]

### Dice Score

\[ Dice=`\frac{2TP}{2TP+FP+FN}`{=tex} \]

Em que: - TP: pixels positivos corretamente identificados. - FP: pixels
incorretamente classificados como positivos. - FN: pixels positivos não
identificados.

Tabela complementar:

  -----------------------------------------------------------------------
  Grupo                   Métrica                 Objetivo
  ----------------------- ----------------------- -----------------------
  Segmentação             IoU                     Sobreposição espacial

  Segmentação             Dice                    Similaridade das
                                                  máscaras

  Segmentação             Precision               Quantificar falsos
                                                  positivos

  Segmentação             Recall                  Quantificar falsos
                                                  negativos

  Temporal                Variação de área        Identificar oscilações
                                                  da extensão segmentada

  Temporal                Deslocamento do         Medir mudanças na
                          centroide               posição

  Temporal                Consistência temporal   Avaliar coerência entre
                                                  quadros

  Computacional           Latência                Tempo de inferência

  Computacional           Memória máxima          Consumo de recursos

  Estatística             IC de 95%               Quantificar incerteza
  -----------------------------------------------------------------------

A consistência temporal deve ser interpretada com cuidado: mudanças
entre quadros podem refletir evolução física real, e não erros de
segmentação.

## 2.9 Cuidados metodológicos

1.  Ground truth independente do modelo.
2.  Sem vazamento de dados entre trajetórias.
3.  Parâmetros escolhidos na validação.
4.  Repetições com diferentes sementes.
5.  Comparação justa, com mesmo teste e mesmas máscaras.

Se as máscaras forem geradas por limiarização da densidade, o
experimento deve ser descrito como reprodução de uma segmentação
definida por critério físico, e não necessariamente descoberta de
estruturas desconhecidas.

## 2.10 Piloto inicial recomendado

-   Dataset: `euler_multi_quadrants`, após verificar campos e critérios
    de segmentação.
-   Ground truth: máscaras com critério físico justificado.
-   Pré-processamento: condição de referência e dez métodos.
-   Arquiteturas: U-Net, SegFormer e Mask2Former.
-   Pós-processamento: dez métodos sobre as predições salvas.
-   Métricas principais: Dice e IoU.
-   Complementares: estabilidade temporal, latência e memória.

------------------------------------------------------------------------

# 3. Pergunta: aplicar o experimento aos dez datasets 2D listados

> **Superada pela Seção 0:** o projeto usará somente o
> `gray_scott_reaction_diffusion`. Esta seção é mantida como registro.

**Usuário**

Para finalizar, os fluidos:

-   `acoustic_scattering` --- Cartesian 2D (256 × 256)
-   `active_matter` --- Cartesian 2D (256 × 256)
-   `euler_multi_quadrants` --- Cartesian 2D (512 × 512)
-   `gray_scott_reaction_diffusion` --- Cartesian 2D (128 × 128)
-   `helmholtz_staircase` --- Cartesian 2D (1024 × 256)
-   `planetswe` --- Angular 2D (256 × 512)
-   `rayleigh_benard` --- Cartesian 2D (512 × 128)
-   `shear_flow` --- Cartesian 2D (128 × 256)
-   `turbulent_radiative_layer_2D` --- Cartesian 2D (128 × 384)
-   `viscoelastic_instability` --- Cartesian 2D (512 × 512)

Se aplicar a este experimento?

**Assistente**

# Aplicação do experimento de segmentação aos 10 datasets do The Well

A resposta foi que é possível aplicar a metodologia aos dez conjuntos,
mas não se deve tratar todos como se representassem o mesmo fenômeno
físico.

A estratégia recomendada foi utilizar um protocolo comum de avaliação
--- U-Net, SegFormer e Mask2Former, dez métodos de pré-processamento e
dez de pós-processamento --- adaptando a definição das máscaras de
referência a cada fenômeno.

Isso permite investigar:

1.  Quais combinações de pré-processamento, arquitetura e
    pós-processamento apresentam melhor desempenho em cada dataset?
2.  Quais combinações são robustas quando aplicadas a diferentes tipos
    de sistemas físicos?

Foi ressaltado que alguns dos dez datasets são de dinâmica de fluidos,
enquanto outros representam fenômenos relacionados, como difusão e
propagação de ondas. Portanto, a formulação mais precisa é um conjunto
de **sistemas físicos 2D**, e não dez tipos equivalentes de fluidos.

## 3.1 Classificação dos dez datasets

  ----------------------------------------------------------------------------------------------------
  Dataset                                   Resolução Fenômeno         Aplicação        Prioridade
                                                      principal        possível de      
                                                                       segmentação      
  --------------------------------- ----------------- ---------------- ---------------- --------------
  `acoustic_scattering`                     256 × 256 Propagação e     Regiões de       Complementar
                                                      espalhamento     amplitude,       
                                                      acústico         energia ou       
                                                                       propriedades do  
                                                                       campo acústico   

  `active_matter`                           256 × 256 Matéria ativa    Regiões com      Alta
                                                                       diferentes       
                                                                       padrões de       
                                                                       atividade e      
                                                                       organização      
                                                                       espacial         

  `euler_multi_quadrants`                   512 × 512 Dinâmica de      Regiões de       Muito alta
                                                      fluidos          densidade,       
                                                      compressíveis    pressão ou       
                                                                       estruturas       
                                                                       definidas por    
                                                                       critérios        
                                                                       físicos          

  `gray_scott_reaction_diffusion`           128 × 128 Reação-difusão   Regiões de       Alta
                                                                       concentração e   
                                                                       padrões          
                                                                       espaciais        

  `helmholtz_staircase`                    1024 × 256 Propagação de    Regiões de       Complementar
                                                      ondas em meio    amplitude e      
                                                      estruturado      estruturas       
                                                                       associadas à     
                                                                       propagação       

  `planetswe`                               256 × 512 Dinâmica de      Regiões de       Alta
                                                      fluidos em       vorticidade ou   
                                                      coordenadas      propriedades de  
                                                      angulares        escoamento       

  `rayleigh_benard`                         512 × 128 Convecção        Estruturas       Muito alta
                                                      térmica          convectivas e    
                                                                       regiões com      
                                                                       diferentes       
                                                                       propriedades     
                                                                       físicas          

  `shear_flow`                              128 × 256 Escoamento por   Regiões de       Muito alta
                                                      cisalhamento     elevada          
                                                                       vorticidade ou   
                                                                       estruturas do    
                                                                       escoamento       

  `turbulent_radiative_layer_2D`            128 × 384 Escoamento       Regiões com      Muito alta
                                                      turbulento com   propriedades     
                                                      processos        termodinâmicas   
                                                      radiativos       distintas        

  `viscoelastic_instability`                512 × 512 Instabilidade    Regiões de       Alta
                                                      viscoelástica    deformação ou    
                                                                       propriedades do  
                                                                       escoamento       
  ----------------------------------------------------------------------------------------------------

As aplicações são propostas de pesquisa, não garantias de que cada
dataset contenha uma variável diretamente adequada à segmentação
sugerida. É necessário conferir metadados, variáveis e características
da simulação.

Foi destacado que `viscoelastic_instability` tem uma versão original
descontinuada por problemas de processamento e que a documentação
recomenda `viscoelastic_instability_v2` em novos experimentos. Isso deve
ser confirmado na versão atual do The Well.

## 3.2 Seleção prioritária

Para um trabalho centrado em segmentação de fluidos 2D, a recomendação
inicial foi começar por cinco:

1.  `euler_multi_quadrants`
2.  `rayleigh_benard`
3.  `shear_flow`
4.  `turbulent_radiative_layer_2D`
5.  `viscoelastic_instability_v2`

Esses conjuntos permitem investigar diferentes regimes e propriedades
físicas, sem restringir a pesquisa a uma única simulação.

Os outros cinco podem ser utilizados em uma segunda etapa para
investigar a transferência da metodologia a sistemas físicos
relacionados.

## 3.3 Ground truth por dataset

A metodologia de geração e validação das máscaras deve ser comum, mas a
definição de classe não precisa ser a mesma em todos os conjuntos.

  ---------------------------------------------------------------------------------
  Dataset                           Possível variável ou    Exemplo de critério
                                    propriedade             para ground truth
  --------------------------------- ----------------------- -----------------------
  `acoustic_scattering`             Amplitude do campo      Regiões que ultrapassam
                                    acústico                limiar definido

  `active_matter`                   Campos que descrevem o  Regiões com padrões
                                    sistema                 físicos definidos

  `euler_multi_quadrants`           Densidade, pressão ou   Regiões delimitadas por
                                    velocidade              critério físico

  `gray_scott_reaction_diffusion`   Concentrações das       Regiões de concentração
                                    espécies químicas       ou padrões de reação

  `helmholtz_staircase`             Campo de onda           Regiões com
                                                            propriedades de
                                                            propagação distintas

  `planetswe`                       Variáveis               Regiões de vorticidade
                                    hidrodinâmicas          ou propriedades do
                                                            escoamento

  `rayleigh_benard`                 Temperatura e           Regiões convectivas
                                    velocidade              segundo critérios
                                                            físicos

  `shear_flow`                      Componentes da          Regiões de elevada
                                    velocidade              vorticidade ou
                                                            cisalhamento

  `turbulent_radiative_layer_2D`    Variáveis               Regiões definidas por
                                    termodinâmicas e        propriedades do
                                    hidrodinâmicas          escoamento

  `viscoelastic_instability_v2`     Campos de velocidade e  Regiões com
                                    variáveis               propriedades de
                                    viscoelásticas          deformação ou
                                                            escoamento distintas
  ---------------------------------------------------------------------------------

Essa tabela é um ponto de partida. A disponibilidade exata das variáveis
deve ser verificada antes de implementar os critérios.

Exemplos: - Em `shear_flow`, caso os campos permitam calcular
vorticidade, uma máscara poderia identificar regiões que ultrapassam um
limiar de magnitude de vorticidade. - Em `rayleigh_benard`, pode-se
investigar a segmentação de regiões convectivas usando propriedades
térmicas e hidrodinâmicas. - Em `euler_multi_quadrants`, pode-se estudar
regiões de densidade ou estruturas do escoamento, desde que o critério
seja justificado.

Se o objetivo for comparar arquiteturas entre datasets, não se deve
comparar diretamente o Dice de tarefas com máscaras semanticamente
diferentes como se medissem a mesma capacidade. Os resultados devem ser
apresentados por dataset e, quando possível, deve-se usar uma tarefa
comparável entre os conjuntos.

## 3.4 Experimento com dez datasets

Mantendo: - 10 datasets. - 10 métodos de pré-processamento. - 3
arquiteturas. - 10 métodos de pós-processamento.

O total de combinações de avaliação será:

\[ N=10`\times10`{=tex}`\times3`{=tex}`\times10`{=tex} \]

\[ `\boxed{N=3000}`{=tex} \]

São 3.000 combinações de avaliação, sem contar condições de referência,
sementes adicionais e testes de hiperparâmetros.

Não são necessários 3.000 treinamentos independentes. Como os
pós-processamentos podem ser aplicados às predições salvas,
inicialmente:

\[ 10`\times10`{=tex}`\times3`{=tex}=300 \]

treinamentos principais, mais as condições de referência. Depois, os dez
pós-processamentos são aplicados às predições de cada modelo.

### Tabela ilustrativa de resultados

  -------------------------------------------------------------------------------------------------------------------
  Dataset                   Arquitetura   Pré-processamento   Pós-processamento           IoU        Dice    Latência
  ------------------------- ------------- ------------------- ------------------- ----------- ----------- -----------
  `euler_multi_quadrants`   U-Net         Z-score             Limiarização                ---         ---         ---

  `euler_multi_quadrants`   SegFormer     Z-score             Limiarização                ---         ---         ---

  `euler_multi_quadrants`   Mask2Former   Z-score             Limiarização                ---         ---         ---

  `rayleigh_benard`         U-Net         Z-score             Limiarização                ---         ---         ---

  `rayleigh_benard`         SegFormer     Z-score             Limiarização                ---         ---         ---

  `rayleigh_benard`         Mask2Former   Z-score             Limiarização                ---         ---         ---
  -------------------------------------------------------------------------------------------------------------------

A tabela completa deve registrar todas as combinações, com
identificadores únicos e parâmetros explícitos.

## 3.5 Pipeline final

1.  Selecionar os dez datasets.
2.  Inspecionar metadados HDF5.
3.  Identificar campos físicos.
4.  Definir fenômeno de interesse para cada dataset.
5.  Construir ground truth.
6.  Dividir dados por trajetória oficial.
7.  Aplicar dez métodos de pré-processamento e baseline.
8.  Treinar U-Net, SegFormer e Mask2Former.
9.  Salvar probabilidades.
10. Aplicar dez métodos de pós-processamento.
11. Avaliar IoU, Dice, Precision e Recall.
12. Avaliar estabilidade temporal, latência e memória.
13. Comparar resultados entre datasets.

Foi recomendado preservar as divisões oficiais e verificar a
documentação específica de cada conjunto.

## 3.6 Duas fases para fortalecer o trabalho

### Fase 1 --- Experimento principal

Utilizar os cinco datasets prioritários, com as três arquiteturas, dez
métodos de pré-processamento e dez métodos de pós-processamento.

Objetivo: identificar as melhores combinações e analisar o comportamento
das arquiteturas em diferentes sistemas físicos.

### Fase 2 --- Avaliação de generalização

Aplicar a metodologia aos cinco datasets restantes.

Objetivo: verificar se os resultados observados na primeira fase se
mantêm em sistemas físicos diferentes.

Pode-se acrescentar experimentos de transferência entre domínios:
treinar em um conjunto e avaliar em outro, desde que definições das
classes, variáveis de entrada e máscaras sejam compatíveis.

Não se deve misturar diretamente todos os datasets em um único
treinamento sem resolver diferenças de resolução, representação física,
coordenadas e significado das classes.

### Conclusão

A recomendação foi aplicar o experimento aos dez conjuntos, mas começar
pelos cinco prioritários. Isso permite validar a construção do ground
truth, testar a metodologia e dimensionar o custo computacional antes de
expandir a pesquisa.

A contribuição científica não deve ser apenas descobrir qual arquitetura
apresenta maior Dice ou IoU. Deve investigar como o pré-processamento e
o pós-processamento afetam a segmentação de diferentes sistemas físicos
e até que ponto os resultados se generalizam entre datasets.

Para sustentar essa conclusão, é fundamental definir máscaras de
referência fisicamente justificáveis, preservar a separação por
trajetória e avaliar cada sistema de acordo com suas características.

------------------------------------------------------------------------

# 4. Síntese das decisões tomadas na conversa

> **Atualização:** os itens que citam dez datasets, 3.000 combinações,
> cinco datasets piloto, `euler_multi_quadrants` e
> `viscoelastic_instability_v2` foram substituídos pela Seção 0
> (um único dataset, 300 combinações, piloto no Gray-Scott).

  -----------------------------------------------------------------------
  Item                                Decisão / recomendação atual
  ----------------------------------- -----------------------------------
  Tarefa principal                    Segmentação 2D

  Arquiteturas                        U-Net, SegFormer e Mask2Former

  Pré-processamento                   Dez métodos P1--P10, mais baseline

  Pós-processamento                   Dez métodos Q1--Q10

  Ground truth                        Máscaras baseadas em critérios
                                      físicos justificados ou anotação
                                      especializada

  Métricas principais                 IoU e Dice

  Métricas complementares             Precision, Recall, estabilidade
                                      temporal, latência e memória

  Unidade de separação dos dados      Trajetórias, evitando vazamento
                                      temporal

  Número de combinações por dataset   300

  Número de combinações para dez      3.000
  datasets                            

  Treinamentos principais para dez    300, mais baselines, por
  datasets                            configuração e semente

  Estratégia de execução              Piloto com cinco datasets; expansão
                                      para os outros cinco

  Dataset inicial sugerido            `euler_multi_quadrants`

  Versão a verificar                  `viscoelastic_instability_v2` em
                                      vez da versão original
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 5. Orientações para o agente do Cursor

> **Atualização:** ler junto com a Seção 0.8, que adapta estas tarefas
> ao uso exclusivo do Gray-Scott.

O agente deve tratar este documento como contexto e plano preliminar de
pesquisa, não como autorização para implementar tudo imediatamente.

## 5.1 Primeira tarefa: auditoria do The Well

Antes de escrever o pipeline, o agente deve:

1.  Consultar a documentação oficial e a estrutura real dos arquivos
    baixados.
2.  Verificar os nomes exatos dos dez datasets e se estão disponíveis na
    versão utilizada.
3.  Inspecionar os metadados HDF5 de cada dataset.
4.  Listar campos escalares, vetoriais e tensoriais.
5.  Verificar dimensões espaciais, temporais, coordenadas e condições de
    contorno.
6.  Verificar número de trajetórias e divisões oficiais de treino,
    validação e teste.
7.  Confirmar a disponibilidade e o formato de
    `viscoelastic_instability_v2`.
8.  Registrar incompatibilidades e diferenças entre datasets.

Não presumir que os nomes de variáveis ou as máscaras descritas neste
documento existam exatamente como apresentados.

## 5.2 Segunda tarefa: especificação do ground truth

Para cada dataset, o agente deve produzir uma ficha contendo:

-   Fenômeno físico.
-   Variáveis disponíveis.
-   Objeto de segmentação proposto.
-   Definição matemática das classes.
-   Critério e parâmetros para gerar as máscaras.
-   Justificativa física e referências.
-   Distribuição das classes.
-   Exemplos visuais de entrada e máscara.
-   Limitações e possíveis vieses.
-   Critérios de validação do ground truth.

O agente não deve escolher limiares arbitrários nem gerar máscaras
definitivas sem justificar a interpretação física. Se houver ambiguidade
científica, deve apresentar alternativas e solicitar decisão antes de
prosseguir.

## 5.3 Terceira tarefa: arquitetura de software

Propor uma estrutura modular, por exemplo:

``` text
project/
├── configs/
│   ├── datasets/
│   ├── preprocessing/
│   ├── models/
│   └── postprocessing/
├── data/
│   ├── raw/
│   ├── processed/
│   └── masks/
├── src/
│   ├── data/
│   │   ├── loaders.py
│   │   ├── metadata.py
│   │   └── splits.py
│   ├── ground_truth/
│   │   ├── generation.py
│   │   └── validation.py
│   ├── preprocessing/
│   │   ├── normalization.py
│   │   ├── filters.py
│   │   └── registry.py
│   ├── models/
│   │   ├── unet.py
│   │   ├── segformer.py
│   │   └── mask2former.py
│   ├── postprocessing/
│   │   ├── morphology.py
│   │   ├── components.py
│   │   ├── watershed.py
│   │   ├── crf.py
│   │   ├── temporal.py
│   │   └── registry.py
│   ├── training/
│   ├── evaluation/
│   └── visualization/
├── experiments/
├── results/
├── notebooks/
├── tests/
└── README.md
```

Essa estrutura é uma proposta inicial e deve ser adaptada ao repositório
existente, caso já exista um projeto.

## 5.4 Quarta tarefa: experimento piloto

Antes de executar 3.000 combinações, implementar um piloto reproduzível
com:

-   Um dataset, inicialmente `euler_multi_quadrants`.
-   Uma definição de ground truth aprovada.
-   Uma condição baseline.
-   Os dez pré-processamentos.
-   As três arquiteturas.
-   Os dez pós-processamentos.
-   Métricas e visualizações.
-   Registro de parâmetros, sementes e resultados.

Validar formatos de tensores, dimensões, normalização, máscaras,
treinamento, inferência e avaliação.

## 5.5 Quinta tarefa: escalabilidade

Somente após o piloto: - Expandir para os outros quatro datasets
prioritários. - Executar repetições com sementes diferentes. - Salvar
probabilidades para evitar retreinamento por pós-processamento. -
Automatizar a geração das tabelas e matrizes de calor. - Expandir para
os cinco datasets complementares. - Considerar experimentos de
generalização entre domínios.

## 5.6 Requisitos de qualidade

-   Usar PyTorch e versões compatíveis e documentadas das dependências.
-   Fixar sementes e registrar versões de bibliotecas.
-   Não misturar trajetórias entre treino, validação e teste.
-   Não ajustar parâmetros com base no teste.
-   Preservar os dados originais.
-   Salvar configurações completas de cada experimento.
-   Testar transformações em campos escalares e vetoriais separadamente.
-   Não aplicar transformações visuais que destruam o significado físico
    dos dados.
-   Calcular métricas por classe e por trajetória, além de médias
    agregadas.
-   Documentar limitações e casos em que um método não é aplicável.

## 5.7 Entrega esperada da primeira etapa

O primeiro resultado do agente deve ser um relatório de auditoria
contendo:

1.  Estrutura dos dez datasets.
2.  Variáveis e dimensões encontradas.
3.  Divisões oficiais disponíveis.
4.  Propostas de ground truth por dataset.
5.  Problemas de compatibilidade identificados.
6.  Recomendação de dataset e tarefa para o piloto.
7.  Plano de implementação modular.

**Não iniciar treinamentos em larga escala antes de concluir e validar
essa auditoria.**

------------------------------------------------------------------------

# 6. Referências e links mencionados

-   The Well: https://polymathic-ai.org/the_well/
-   Template do artigo (Elsevier Results in Engineering):
    https://scispace.com/templates/elsevier-results-in-engineering-template-i4k529prb67ik4l
-   Gray-Scott (dataset do projeto):
    https://polymathic-ai.org/the_well/datasets/gray_scott_reaction_diffusion/
-   Gerador do Gray-Scott: https://github.com/danfortunato/spectral-gray-scott
-   Repositório do The Well: https://github.com/PolymathicAI/the_well
-   Artigo do The Well: https://arxiv.org/abs/2412.00568
-   U-Net: https://arxiv.org/abs/1505.04597
-   SegFormer --- artigo:
    https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html
-   SegFormer --- implementação: https://github.com/NVlabs/SegFormer
-   Mask2Former --- artigo:
    https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html
-   Mask2Former --- implementação:
    https://github.com/facebookresearch/Mask2Former
-   SAM 2 --- artigo: https://arxiv.org/abs/2408.00714 (fora do escopo, ver Seção 0.9.2)

> **Nota final ao agente:** este histórico registra as decisões e
> hipóteses discutidas, mas não substitui a verificação da documentação,
> dos metadados e dos dados reais. A prioridade é validar
> cientificamente o ground truth e a compatibilidade das tarefas antes
> de implementar o experimento completo.
