# 08 — Fase D: estudo controlado de aprendizagem de affordance geométrica de inserção

> Nota de implementação para futura conversão em secção LaTeX.
> Estado: estudo controlado de aprendizagem — resultado positivo-mas-estreito.
> Data: 2026-05-09.

---

## 1. Objetivo

A Fase D testou uma questão distinta das questões testadas pelas
Fases B e C da Baseline 2 (doc 06 — secção 1; doc 07 — secção 1).
Aquelas fases mantiveram o cabeçote de scoring da Baseline 1
(doc 03 — secção 17) congelado e variaram a representação. A
Fase D inverte a higiene experimental: a representação de entrada
é mantida no espaço de features geométricas já produzidas pela
perceção determinística da Baseline 1, e o que varia é a
fronteira de decisão, que passa a ser estimada por um modelo
interpretável treinado sobre um dataset procedimental de
geometrias prismáticas.

A pergunta experimental foi formulada de forma estrita: dado um
dataset procedimental de pares peça–cavidade rotulados por uma
regra física explícita de inserção parcial, e mantendo as
features ao mínimo (footprints 2D, alturas, profundidades,
áreas e razões), pode um modelo interpretável (regressão
logística e árvore de decisão) reproduzir o ranking correcto das
quatro peças MVP face às quatro cavidades MVP, e fá-lo com
margens comparáveis às da Baseline 1?

A Fase D **não** introduz fusão geométrica, **não** introduz
reconstrução 3D, **não** introduz visão aprendida, **não**
introduz redes neurais, **não** introduz aprendizagem por
reforço, **não** introduz controlo robótico nem planeamento de
trajectória. O modelo prevê um score de affordance geométrica de
inserção parcial sob uma hipótese de extrusão vertical e ordena
cavidades candidatas para uma primitiva fixa de inserção
vertical. Phase D consome as features derivadas pela perceção
determinística da Baseline 1; refina a fronteira de decisão, não
a representação.

A Fase D é implementada em quatro scripts:
`scripts/generate_phaseD_3d_affordance_dataset.py`,
`scripts/train_phaseD_3d_affordance_model.py`,
`scripts/evaluate_phaseD_affordance_ranking.py` e
`scripts/evaluate_phaseD_mvp_board_affordance.py`. Os artefactos
ficam em `data/phaseD_3d_affordance/`.

---

## 2. Representação 3D por extrusão

A representação geométrica subjacente é uma hipótese de
extrusão vertical. Cada peça é assumida como um prisma recto cuja
base é a footprint 2D produzida pela Baseline 1 (doc 03 —
secções 4 e 5) e cuja altura é uma medida escalar `piece_height_mm`.
Cada cavidade é assumida como uma abertura prismática cuja base é
a abertura 2D validada (doc 03 — secção 4) e cuja profundidade é
uma medida escalar `cavity_depth_mm`. Esta representação restringe
o domínio de aplicabilidade a formas convexas prismáticas e exclui
peças com geometria não-extrudável; é a hipótese que torna o
problema tratável como inferência sobre features escalares e não
como inferência sobre voxels ou TSDFs.

As features de pareamento são herdadas da Baseline 1: `area_ratio`,
`iou`, `inside_ratio_raw`, `outside_ratio_raw` calculadas sobre a
melhor rotação encontrada pela pesquisa em grelha. A elas
acrescenta-se a comparação 1D entre profundidade de cavidade e
altura de peça, descrita em §4.

---

## 3. Primeira formulação: full containment

A primeira iteração do gerador de dataset (versões D.1 a D.6 do
script `generate_phaseD_3d_affordance_dataset.py`) implementou
uma regra de rotulagem de **contenção total**:

```
success = lateral_ok AND (cavity_depth >= piece_height − 0.5 mm)
```

isto é, uma configuração só seria positiva se a profundidade da
cavidade fosse pelo menos tão grande como a altura da peça (até
uma tolerância de meio milímetro). Esta regra incorpora
implicitamente a hipótese de que uma inserção válida exige que a
peça desapareça inteiramente dentro da cavidade.

Foi observado, na avaliação D.6 sobre o cenário real MVP, que
esta regra falha estruturalmente. As peças MVP têm
aproximadamente `104.5 mm` de altura e a espessura nominal CAD do
tabuleiro é de `75 mm`, donde nenhuma das cavidades MVP admite
contenção total de qualquer das peças MVP. Constatou-se que a
tarefa MVP é, por definição construtiva, uma **inserção parcial
através de uma abertura** num shape-sorter, e não uma operação de
contenção. Sob a regra D.1–D.6 as probabilidades preditas no
cenário MVP foram praticamente nulas em todas as cavidades. A
regressão logística obteve 2/4 e a árvore 1/4, valores
essencialmente compatíveis com sorteio. Adicionalmente, na
configuração D.6 a `cavity_02` recebeu profundidade por *fallback*
CAD enquanto as restantes cavidades recebiam profundidade
sensorial, o que contaminava qualquer comparação com Baseline 1.
A iteração foi arquivada como evidência negativa metodológica.

---

## 4. Correção da definição física

A iteração D.7 substitui a regra de contenção total por uma regra
de **inserção parcial guiada**, mais fiel à física do
shape-sorter:

```
insertion_required_mm = max(MIN_REQUIRED_INSERTION_MM = 5.0,
                             INSERTION_FRACTION = 0.25 * piece_height_mm)
depth_ok = (cavity_depth_mm >= insertion_required_mm − DEPTH_TOLERANCE_MM = 0.5)
            AND (cavity_depth_mm >= MIN_INSERTION_GUIDANCE_MM = 5.0)
label = lateral_ok AND depth_ok
```

isto é, uma inserção é válida se a cavidade admite que a peça
desça pelo menos `5 mm` ou um quarto da sua altura (o que for
maior), com `0.5 mm` de tolerância numérica. Os limiares
laterais são herdados da Baseline 1 sem alteração:
`outside_ratio_raw ≤ 0.05` e `inside_ratio_raw ≥ 0.80`. Os
valores `MIN_REQUIRED_INSERTION_MM = 5.0`,
`INSERTION_FRACTION = 0.25`, `DEPTH_TOLERANCE_MM = 0.5` e
`MIN_INSERTION_GUIDANCE_MM = 5.0` são pontos de operação fixos
declarados a priori, e não hiperparâmetros sintonizados a
posteriori sobre os resultados.

Como consequência semântica, a feature anteriormente nomeada
`depth_compatibility_mm` foi renomeada para `depth_offset_mm`. O
valor numérico mantém-se inalterado; a renomeação reflecte a
mudança de hipótese subjacente: deixa de medir compatibilidade
de contenção e passa a medir folga de inserção. Pelo mesmo
motivo, a feature `volume_ratio` foi removida, dado que
implicava contenção total e deixou de ser semanticamente válida
sob a regra de inserção parcial.

---

## 5. Dataset regenerado

A regeneração D.7 do dataset procedimental tem os seguintes
parâmetros e estatísticas:

- Total de configurações: **26 208**.
- Taxa global de positivos: **20.71 %** (era 9.76 % no D.1/D.2).
- **67.85 %** dos positivos satisfazem `cavity_depth < piece_height`,
  isto é, configurações em que o regime de inserção é
  estruturalmente parcial. Este regime estava completamente
  ausente do dataset D.1/D.2.
- Intervalos: `PIECE_HEIGHT_RANGE_MM = (20, 150)` (era 50–150);
  `CAVITY_DEPTH_RANGE_MM = (10, 100)` (era 50–100). Os
  alargamentos foram introduzidos para cobrir o regime parcial
  e para incluir cavidades pouco profundas que admitem apenas
  guiamento.

Distribuição de positivos por família procedimental:

| família                     | n_configs | n_positivos | taxa  |
|-----------------------------|----------:|------------:|------:|
| `rectangle`                 | 5 544     | 918         | 0.166 |
| `ellipse`                   | 5 292     | 1 652       | 0.312 |
| `regular_polygon`           | 5 292     | 1 873       | 0.354 |
| `convex_irregular_polygon`  | 5 040     | 340         | 0.068 |
| `rounded_rectangle`         | 5 040     | 644         | 0.128 |

Nenhuma família tem zero positivos. A família
`convex_irregular_polygon` é a mais difícil, com taxa de
positivos abaixo de 7 %, o que é consistente com a maior
diversidade interna de aspect ratios admitida pela família.

---

## 6. Modelos treinados

Sob a higiene experimental adoptada na Fase B (doc 06 —
secção 2), foram treinados apenas dois modelos interpretáveis:

- **Regressão logística** com regularização L2 (`C = 1.0`),
  `class_weight = 'balanced'`, `StandardScaler` aplicado às
  features, `max_iter = 5000`.
- **Árvore de decisão** com `max_depth = 4`,
  `class_weight = 'balanced'`, `random_state = 0`.

Não foram treinados modelos de redes neurais, random forests
(deliberadamente excluído nesta fase), gradient boosting, nem
qualquer componente de aprendizagem por reforço. A escolha
restringe-se a duas famílias com fronteiras de decisão
interpretáveis e auditáveis.

---

## 7. Resultados de classificação

Sobre o split standard (treino / validação / teste estratificado
por família), os modelos obtiveram:

| modelo  | split | n     | acc   | precisão | recall | F1    | AUC   |
|---------|-------|------:|------:|---------:|-------:|------:|------:|
| logreg  | test  | 3 885 | 0.921 | 0.719    | 0.978  | 0.829 | 0.987 |
| tree    | test  | 3 885 | 0.942 | 0.774    | 0.995  | 0.871 | 0.975 |

Sob avaliação leave-one-family-out (LOFO), comparando D.7
contra a versão anterior D.3:

| família held-out             | logreg F1 D.7 | logreg F1 D.3 | tree F1 D.7 | tree F1 D.3 |
|------------------------------|--------------:|--------------:|------------:|------------:|
| `convex_irregular_polygon`   | 0.683         | 0.758         | 0.667       | 0.701       |
| `ellipse`                    | 0.825         | 0.762         | 0.895       | 0.863       |
| `rectangle`                  | 0.818         | 0.696         | 0.837       | 0.899       |
| `regular_polygon`            | 0.865         | 0.864         | 0.898       | 0.954       |
| `rounded_rectangle`          | 0.833         | 0.531         | 0.543       | 0.546       |
| **média**                    | **0.805**     | 0.722         | **0.768**   | 0.792       |

Verifica-se que o LOFO médio da regressão logística melhorou
substancialmente entre D.3 e D.7 (`0.722 → 0.805`), sustentado
sobretudo por ganhos em `rectangle` e `rounded_rectangle`. A
árvore regrediu ligeiramente em média (`0.792 → 0.768`), com
quebra em `regular_polygon` e em `convex_irregular_polygon`. Não
foi observada degradação catastrófica em nenhum dos modelos.

---

## 8. Ranking procedimental

A avaliação de ranking procedimental,
`scripts/evaluate_phaseD_affordance_ranking.py`, mede top-1
sobre todos os pares peça–cavidade gerados:

- **Regressão logística**: top-1 em `all_procedural = 1.000`,
  em `test_split = 0.978` e em `mvp_procedural = 1.000`. As
  margens médias rank-1 vs rank-2 encolheram face a D.5
  (`0.018` em D.7 contra `0.379` em D.5), o que é esperado e
  desejável: o dataset corrigido contém regiões de operação
  mais difíceis (inserção parcial com folga reduzida) que
  estavam ausentes do dataset anterior.
- **Árvore de decisão**: top-1 em todos os escopos `= 1.000`
  excepto `test_split = 0.946`. As margens situam-se entre
  `0.000` e `0.048`, com empates frequentes no rank-1. Este
  fenómeno é estrutural: as folhas da árvore retornam
  probabilidades discretas, donde múltiplas configurações
  caem na mesma folha e recebem o mesmo score, produzindo
  empates de margem nula.

---

## 9. Avaliação no cenário real MVP board

A avaliação no cenário MVP é feita por
`scripts/evaluate_phaseD_mvp_board_affordance.py`. Para
eliminar a contaminação observada em D.6, em que `cavity_02`
recebia profundidade CAD enquanto as restantes recebiam
profundidade sensorial, a profundidade de cavidade foi
**fixada uniformemente em 75 mm (CAD nominal)** em todas as
quatro cavidades. Não há mistura sensor/CAD por cavidade.

Resultados rank-1 por peça contra a referência da Baseline 1
(doc 03 — secção 17.4):

| peça        | referência B1 | logreg rank-1   | logreg score | tree rank-1     | tree score |
|-------------|---------------|-----------------|-------------:|-----------------|-----------:|
| `rectangle` | `cavity_00`   | **`cavity_00`** ✓ | 0.9961       | **`cavity_00`** ✓ | 0.9665     |
| `square`    | `cavity_02`   | **`cavity_00`** ✗ | 0.9991       | **`cavity_02`** ✓ | 0.9665     |
| `circle`    | `cavity_03`   | **`cavity_00`** ✗ | 0.9997       | **`cavity_03`** ✓ | 0.9665     |
| `triangle`  | `cavity_01`   | **`cavity_00`** ✗ | 0.9998       | **`cavity_01`** ✓ | 0.9665     |

Top-1 accuracy no tabuleiro real:

- **logreg**: 1/4 (25 %). Constatou-se que o modelo colapsa
  sistematicamente para `cavity_00`, a cavidade de maior área.
  A inspecção do vector de coeficientes revela que o
  coeficiente associado a `area_ratio` é `−16.5`, isto é, a
  regressão logística aprendeu a penalizar fortemente
  desfasamentos de área, o que sob profundidade de cavidade
  uniforme reduz a decisão a "menor peça encaixa em maior
  cavidade" e falha em três das quatro peças.
- **tree**: 4/4 (100 %). A árvore reproduz a atribuição
  correcta para todas as quatro peças MVP. Verifica-se, contudo,
  que os scores vencedores são **idênticos a `0.9665`** para
  três das quatro peças (`square`, `circle`, `triangle`), com
  margem de ranking exactamente igual a `0.000` nesses três
  casos. Estes empates são consequência directa da quantização
  de folhas referida em §8.

Comparando contra a Baseline 1 (doc 03 — secção 17.4):

| peça        | B1 (cavidade, margem)   | Phase D logreg (cavidade, margem) | Phase D tree (cavidade, margem) |
|-------------|-------------------------|-----------------------------------|---------------------------------|
| `rectangle` | `cavity_00`, **0.293**  | `cavity_00`, ~                    | `cavity_00`, **0.000***         |
| `square`    | `cavity_02`, **0.168**  | `cavity_00`, ~ (incorrecto)       | `cavity_02`, **0.000***         |
| `circle`    | `cavity_03`, **0.114**  | `cavity_00`, ~ (incorrecto)       | `cavity_03`, **0.000***         |
| `triangle`  | `cavity_01`, **0.227**  | `cavity_00`, ~ (incorrecto)       | `cavity_01`, **0.000***         |

(* a margem `0.000` reflecte quantização de folha, não
confiança real do modelo.)

A Baseline 1 obtém 4/4 corretas com margens decisivas em todos
os pares diagonais (`compatible = True`). Phase D tree obtém
4/4 corretas mas com margens nulas em três pares. Phase D
logreg obtém 1/4 correcta. **O modelo aprendido, em qualquer
das duas variantes, não supera a Baseline 1 em
discriminabilidade no cenário MVP.**

O mecanismo é geométrico e não modelativo. No tabuleiro real
todas as cavidades têm a mesma profundidade nominal CAD
(`75 mm`), pelo que `depth_offset_mm` e `cavity_depth_mm` não
discriminam entre cavidades. A discriminação cai inteiramente
sobre features de ajuste lateral (`area_ratio`, `iou`), que são
exactamente as features que a Baseline 1 já calcula e combina
deterministicamente. O contributo da Fase D no cenário MVP é,
portanto, **metodológico** (regra de rotulagem corrigida,
dataset honesto, modelo que treina sem fugas de avaliação) e
não **científico** (sem ganho mensurável sobre a Baseline 1).

---

## 10. Discussão crítica

A Fase D é registada como **estudo controlado de aprendizagem
com resultado positivo-mas-estreito**. O resultado é positivo
no sentido em que: (i) a regra de rotulagem foi corrigida de
contenção total para inserção parcial, alinhando o dataset com
a física do shape-sorter; (ii) o dataset regenerado contém uma
fracção substancial (~68 %) de positivos no regime de inserção
parcial, regime que estava ausente do dataset original;
(iii) os modelos interpretáveis treinam sem degradação
catastrófica em LOFO; (iv) a árvore de decisão reproduz a
atribuição correcta para todas as quatro peças MVP. O
resultado é estreito no sentido em que essa reprodução
acontece com margens nulas em três dos quatro pares, e a
regressão logística reduz-se efectivamente a "menor peça
encaixa em maior cavidade" sob profundidade uniforme,
falhando em três dos quatro pares.

A leitura honesta é a seguinte. A Fase D demonstra que é
possível treinar um classificador interpretável de affordance
geométrica de inserção parcial sobre o espaço de features da
Baseline 1, com generalização razoável entre famílias
procedimentais. Não demonstra que esse classificador supere a
Baseline 1 no cenário MVP. Sob a hipótese operativa do
shape-sorter actual — quatro peças prismáticas, quatro
cavidades de profundidade CAD uniforme — a discriminação útil
está concentrada em features laterais que a Baseline 1 já
explora directamente, e o ganho marginal da fronteira aprendida
sobre a fronteira analítica é mensuravelmente nulo. A Baseline 1
mantém-se como o método operacional de referência mais forte
sobre o conjunto MVP convexo.

---

## 11. Limitações

1. **Suposição de extrusão vertical.** Aplicabilidade restrita a
   formas convexas prismáticas; peças não-extrudáveis ficam
   fora do domínio.
2. **Dataset procedimental.** Não há capturas reais alargadas no
   conjunto de treino; o dataset é gerado parametricamente.
3. **Sem dinâmica de contacto.** Não modela fricção, força nem
   contacto físico.
4. **Sem controlo robótico.** Não planeia trajectória, não
   executa agarre, não valida pose de inserção.
5. **Sem perceção visual aprendida.** As features são as da
   perceção determinística da Baseline 1.
6. **Sem reconstrução 3D arbitrária.** A representação
   subjacente é a hipótese de extrusão vertical, não uma
   reconstrução volumétrica.
7. **Sem evidência de superação face à Baseline 1 no MVP.** A
   árvore reproduz a atribuição correcta sem margens robustas;
   a regressão logística falha em três das quatro peças.
8. **Margens nulas da árvore.** Reflectem a quantização de
   folhas em árvores de profundidade limitada, não confiança
   real do modelo.
9. **Profundidade de cavidade fixada à nominal CAD (75 mm).**
   A captura sensorial está limitada a aproximadamente
   `14–17 mm` por limitações da gama útil do annotator de
   profundidade na configuração actual, valor que cai fora do
   intervalo `[10, 100] mm` usado no treino e que constitui,
   portanto, uma região OOD para o classificador. A
   profundidade nominal CAD foi escolhida como representação
   uniforme não-contaminada, sem mistura sensor/CAD por
   cavidade.

---

## 12. Decisão final

Fica decidido manter a Fase D como **estudo controlado de
aprendizagem**, com a seguinte leitura:

- A Fase D alinha a tese com o seu objectivo declarado de
  aprendizagem perceção-acção, ao introduzir uma componente
  treinada sobre features geométricas auditáveis;
- A Fase D documenta uma correcção metodológica não-trivial
  (regra de full containment substituída por regra de
  inserção parcial guiada), com impacto mensurável na
  composição do dataset e na cobertura do regime de operação
  realista;
- A Fase D **não** substitui a Baseline 1 (doc 03 — secção 17)
  como referência operacional sobre o conjunto MVP convexo;
- Não foi feita afinação adicional sobre os pontos de operação
  fixos; não foi treinado random forest; não foi treinado
  modelo de aprendizagem profunda nesta fase.

Trabalho futuro, fora do escopo da tese actual, poderá
explorar: sensor de profundidade com gama alargada que permita
captar a profundidade real das cavidades sem recurso a CAD;
cavidades com profundidades variadas (fora do conjunto MVP
actual); features descritoras adicionais (por exemplo
descritores de forma não-convexos do tipo alpha-shape) caso
peças côncavas venham a ser introduzidas no conjunto.

---

## 13. Notas de redação para evitar overclaim

Para futura conversão em texto LaTeX, ficam fixadas as
seguintes formulações como aceitáveis:

- "Foi treinado um modelo interpretável para estimar uma
  *affordance* geométrica de inserção parcial sob uma hipótese
  de extrusão vertical."
- "Sob a regra corrigida e com profundidade CAD uniforme, a
  árvore de decisão reproduz a atribuição correcta para todas
  as quatro peças MVP, mas com margens nulas em três dos
  quatro casos."
- "Phase D consome as features derivadas pela perceção
  determinística da Baseline 1; refina a fronteira de decisão,
  não a representação."

E ficam fixadas, como exemplos a evitar pela razão indicada
entre parêntesis:

- "O modelo aprendeu a controlar o robô." (a Fase D não envolve
  controlo robótico nem execução).
- "O modelo resolve inserção robótica geral." (o domínio é
  restrito a extrusão vertical sobre conjunto convexo MVP).
- "Phase D supera Baseline 1 no cenário MVP." (não é apoiado
  pelos números: árvore com margens nulas, logreg com 1/4).
- "A affordance geométrica é uma garantia física de inserção
  bem-sucedida." (a affordance é uma estimativa geométrica sob
  hipótese de extrusão; não modela contacto, fricção, força
  nem dinâmica).
