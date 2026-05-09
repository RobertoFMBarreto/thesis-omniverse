# 07 — Baseline 2 Fase C: união canónica XY no referencial do mundo, resultados

> Nota de implementação para futura conversão em secção LaTeX.
> Estado: resultado experimental — registo de evidência negativa exploratória.
> Data: 2026-05-09.

---

## 1. Objetivo

A Fase C testou uma questão de representação, distinta da
questão de agregação testada na Fase B (doc 06 — secção 1).
A pergunta foi: dado o mesmo conjunto de três vistas por
entidade usado pela Fase A (doc 04 — secção 1.4), é possível
transformar as observações deterministas multi-vista numa
única representação canónica XY no referencial do mundo, antes
de qualquer pontuação, e obter melhoria face à Baseline 1
(doc 03 — secção 17.4) mantendo o cabeçote de scoring
inalterado?

A Fase C **não** introduziu fusão geométrica volumétrica,
TSDF, SLAM, ICP, estimação de pose, componente aprendido nem
execução robótica. O escopo é estritamente: união no
referencial do mundo das nuvens XY observadas em cada vista,
centragem única após união, e reutilização do rasterizador e
da score head da Baseline 1, sem modificação.

---

## 2. Método

A Fase C é implementada em
`scripts/baseline2_phaseC_canonical_multiview.py`. O script é
novo; **não** modifica o pipeline da Baseline 1 nem o pipeline
da Fase B. Reutiliza, por importação, o rasterizador
`rasterise_xy_to_mask` e a função `score_pair` da Baseline 1,
ambos sem alteração.

Para cada entidade canónica (cada uma das quatro peças MVP e
cada uma das quatro cavidades) o procedimento é:

1. Para cada vista `top_down`, `front_oblique` e `side_oblique`
   (doc 04 — secção 1.4), retroprojectar o mapa de profundidade
   para o referencial do mundo.
2. Segmentar a entidade. Para peças: pontos cuja altura excede
   a superfície de apoio em pelo menos
   `PIECE_HEIGHT_MIN_ABOVE_SURFACE_M = 0.002 m`. Para cavidades
   na vista `top_down`: o pointcloud validado pela Baseline 1
   (`cavity_opening_pointcloud.npy`). Para cavidades nas vistas
   oblíquas: filtro XY de ROI
   (`CAVITY_VIEW_ROI_HALF_SIZE_M = 0.055 m`) com banda Z fina
   (`CAVITY_DEPTH_MIN_BELOW_SURFACE_M = 0.001 m`,
   `CAVITY_DEPTH_MAX_BELOW_SURFACE_M = 0.005 m`), conforme
   sub-iteração B.3 da Fase B (doc 06 — secção 3.2).
3. Filtrar vistas com menos de `MIN_VIEW_POINTS = 50` pontos.
4. Unir os pontos XY de todas as vistas válidas por
   `np.vstack`, no referencial do mundo. Esta união é o ponto
   crítico da Fase C: substitui a agregação ao nível do score
   da Fase B por agregação ao nível da nuvem de pontos XY.
5. Centrar o conjunto unido **uma única vez** no centróide do
   conjunto unido, sem reaplicar centragens parciais por vista.
6. Rasterizar por `rasterise_xy_to_mask` (canvas de 320×320 px
   @ 0.25 mm/px), idêntico ao da Baseline 1.
7. Pontuar cada par peça-cavidade canónico por `score_pair`
   (180 rotações no plano com pesos `W_IOU = 0.55`,
   `W_INSIDE = 0.35`, `W_OUTSIDE = 0.10`, herdados sem alteração).

Não há fallback automático para a Baseline 1 quando o conjunto
unido é esparso. A política da Fase C é marcar
`canonical_sparse = True` e prosseguir; no presente run nenhuma
entidade ficou esparsa. O critério `TIE_MARGIN = 0.01` é
igualmente herdado.

---

## 3. Resultados

### 3.1 Diagnósticos por entidade canónica

| Entidade           | Vistas | Pontos unidos | Hull fallback | Sparse | Inválida |
|--------------------|:------:|--------------:|:-------------:|:------:|:--------:|
| `rectangle` (peça) | 3/3    | 13 872        | não           | não    | não      |
| `square` (peça)    | 3/3    | 10 004        | não           | não    | não      |
| `circle` (peça)    | 3/3    | 8 540         | não           | não    | não      |
| `triangle` (peça)  | 3/3    | 7 371         | não           | não    | não      |
| `cavity_00`        | 3/3    | 2 172         | não           | não    | não      |
| `cavity_01`        | 3/3    | 2 266         | não           | não    | não      |
| `cavity_02`        | 3/3    | 2 264         | não           | não    | não      |
| `cavity_03`        | 3/3    | 2 343         | não           | não    | não      |

O critério terciário (per-view ≥ `MIN_VIEW_POINTS = 50`) foi
satisfeito em todas as oito entidades. A falha do critério
primário reportada em 3.3 **não** se deve a falta de dados.

### 3.2 Matriz 4×4 de scores

|             | `cavity_00` | `cavity_01` | `cavity_02` | `cavity_03` |
|-------------|------------:|------------:|------------:|------------:|
| `rectangle` | **0.663**   | 0.215       | 0.476       | 0.226       |
| `square`    | **0.624**   | 0.222       | 0.541       | 0.188       |
| `circle`    | **0.588**   | 0.213       | 0.575       | 0.169       |
| `triangle`  | **0.524**   | 0.135       | 0.467       | 0.100       |

### 3.3 Ranking rank-1 por peça

| peça        | rank-1        | score | IoU   | rotação | margem    | correcto?                       |
|-------------|---------------|------:|------:|--------:|----------:|---------------------------------|
| `rectangle` | `cavity_00`   | 0.663 | 0.690 | 20°     | 0.187     | sim                             |
| `square`    | `cavity_00`   | 0.624 | 0.570 | 58°     | 0.083     | NÃO (verdadeiro: `cavity_02`)   |
| `circle`    | `cavity_00`   | 0.588 | 0.477 | 44°     | **0.013** | NÃO (verdadeiro: `cavity_03`)   |
| `triangle`  | `cavity_00`   | 0.524 | 0.328 | 336°    | 0.057     | NÃO (verdadeiro: `cavity_01`)   |

O critério primário (4/4 atribuições rank-1 corretas)
**falhou**: apenas 1/4 (`rectangle`). Constatou-se que três das
quatro peças colapsam para o mesmo rank-1, `cavity_00`. A
margem do `circle` (`0.013`) está imediatamente acima de
`TIE_MARGIN = 0.01`, sendo na prática indistinguível de empate.

O critério secundário (margens rank-1 vs rank-2 ≥ 70 % das da
Baseline 1) falha duplamente: a margem do `rectangle` na Fase C
(`0.187`) é inferior a 70 % da margem da Baseline 1
(`0.293 × 0.70 = 0.205`); para as restantes três peças o
critério é vacuoso, dada a atribuição incorrecta.

---

## 4. Comparação com Baseline 1 e Fase B

| peça        | Baseline 1 (cavidade, margem) | Fase B híbrido (cavidade, margem) | **Fase C** (cavidade, margem)   |
|-------------|------------------------------:|----------------------------------:|--------------------------------:|
| `rectangle` | `cavity_00`, 0.293            | `cavity_00`, 0.107                | `cavity_00`, **0.187**          |
| `square`    | `cavity_02`, 0.168            | `cavity_02`, 0.091                | **`cavity_00`, 0.083** ✗        |
| `circle`    | `cavity_03`, 0.114            | `cavity_03`, 0.058                | **`cavity_00`, 0.013** ✗        |
| `triangle`  | `cavity_01`, 0.227            | `cavity_03`, 0.006 ✗              | **`cavity_00`, 0.057** ✗        |

Em síntese: Baseline 1 atinge 4/4 corretas; Fase B híbrido
3/4 corretas com margens reduzidas (doc 06 — secção 4); Fase C
1/4 corretas, com colapso de três peças para a mesma cavidade.

---

## 5. Discussão crítica

A Fase C **não** melhorou face à Baseline 1; também **não**
melhorou face à Fase B. A falha não é por falta de dados — o
critério terciário foi satisfeito em todas as oito entidades,
com 3/3 vistas válidas em cada uma. A falha é representacional.

O mecanismo é o que tinha sido antecipado pelo desenho da Fase C
como modo de falha #4. Quando os mapas de profundidade das
vistas oblíquas de uma peça são retroprojectados para XY no
referencial do mundo e unidos com a vista `top_down`, os pixels
das faces laterais da peça projectam-se em XY como uma região
mais larga do que a verdadeira silhueta top-down. Após a
centragem única no centróide da união e a rasterização
canónica, a máscara resultante de cada peça fica inflada e
borrada em direcção ao envoltório do extensor 3D total da peça.
A consequência directa é que as máscaras de peça encaixam
preferencialmente na maior cavidade — `cavity_00`, abertura
rectangular de aproximadamente 51 × 76 mm — porque a maior
cavidade tolera melhor uma máscara inflada sob o critério
combinado IoU + inside − outside.

Do lado das cavidades o efeito é assimétrico. O conjunto unido
de cada cavidade é dominado pela componente `top_down`, que é
o pointcloud validado pela Baseline 1 (aproximadamente 2 000
dos 2 200 a 2 300 pontos unidos); as componentes oblíquas
contribuem apenas com poucas centenas de pontos por cavidade.
As cavidades canónicas mantêm, portanto, aproximadamente as
suas formas Baseline 1; o problema está exclusivamente do lado
das peças.

Esta assimetria é importante para a leitura do resultado.
**Não** pode ser concluído que a perceção multi-vista, em geral,
é prejudicial. O que pode ser concluído é mais restrito — a
união ingénua em XY no referencial do mundo, sem distinção
entre pontos da face superior e pontos das faces laterais da
peça, infla a silhueta canónica e produz um viés sistemático
para a maior cavidade. A Fase C é evidência negativa
diagnóstica: o resultado é informativo precisamente porque o
cabeçote de scoring foi mantido inalterado e os pesos não foram
afinados, o que isola a representação como única fonte de
variação face à Baseline 1.

---

## 6. Limitações

1. **Conjunto MVP convexo.** Apenas quatro peças e quatro
   cavidades convexas; não pode ser concluído que o mecanismo
   de inflação observado generaliza para peças não-convexas.
2. **Sem reconstrução volumétrica verdadeira.** A Fase C é uma
   união em XY no referencial do mundo; não constrói TSDF,
   voxel grid nem malha.
3. **Sem classificação explícita de superfícies.** Não há
   classificação por orientação da normal nem distinção entre
   pontos da face superior e pontos das faces laterais.
4. **Sem separação top vs lateral.** Em consequência directa do
   ponto anterior, a máscara canónica de peça mistura a
   silhueta top-down com a contribuição XY das faces laterais.
5. **Captura sequencial.** Captura por uma única câmara
   reposicionada programaticamente entre vistas (doc 04 —
   secções 1.4 e 2.6); não é um rig multi-câmara síncrono.
6. **Ambiente sintético Isaac Sim.** Geometria exacta por
   construção; resultados não transportados para captura real
   sem nova validação.
7. **Sem execução robótica.** A Fase C reporta apenas
   `(cavity, rotação no plano, score canónico)`; sem estimação
   de pose 6D, sem pose de inserção, sem pick-and-place.

---

## 7. Decisões tomadas

- **Não afinar thresholds.** Os limiares
  `MIN_VIEW_POINTS = 50`,
  `PIECE_HEIGHT_MIN_ABOVE_SURFACE_M = 0.002`,
  `CAVITY_DEPTH_MIN_BELOW_SURFACE_M = 0.001`,
  `CAVITY_DEPTH_MAX_BELOW_SURFACE_M = 0.005`,
  `CAVITY_VIEW_ROI_HALF_SIZE_M = 0.055` e `TIE_MARGIN = 0.01`
  foram herdados sem alteração da Baseline 1 e da
  sub-iteração B.3 da Fase B; não foram afinados a posteriori.
- **Não adicionar descritores.** Em conformidade com a higiene
  experimental adoptada na Fase B (doc 06 — secção 2), o
  cabeçote de scoring foi mantido congelado.
- **Não continuar a Fase C nesta forma.** O mecanismo de
  inflação é uma consequência geométrica esperada da união
  ingénua em XY, não um defeito a corrigir por afinação dentro
  do mesmo método.
- **Manter a Baseline 1 como baseline determinística mais
  forte.** Sobre o conjunto MVP convexo, a Baseline 1 (doc 03
  — secção 17.4) permanece a referência determinística.
- **Tratar a Fase C como evidência negativa exploratória.** O
  resultado é registado em vez de descartado, porque é
  diagnóstico: documenta o limite da união canónica XY no
  referencial do mundo na ausência de qualquer separação entre
  pontos da face superior e pontos das faces laterais.

---

## 8. Próximos passos

Se trabalho futuro vier a explorar perceção multi-vista numa
forma mais profunda, deve ser uma fase separada, fora do
escopo da Baseline 2 e fora do escopo do baseline
determinístico validado por esta tese. Não é compromisso desta
nota; são direcções possíveis, registadas para referência:

- **Fusão geométrica verdadeira** dos pontos no referencial do
  mundo, no espírito do primeiro ponto de doc 04 — secção 2.7,
  com reconstrução volumétrica que distinga geometria observada
  de geometria não observada.
- **Classificação de superfícies** por orientação da normal,
  permitindo separar pontos da face superior de pontos das
  faces laterais antes de qualquer projecção XY.
- **Representações canónicas que distingam faces top vs
  laterais**, de forma a que a silhueta canónica de peça
  recupere apenas a contribuição da face superior, eliminando
  o mecanismo de inflação observado em 5.

Estas direcções estão fora da baseline validada da tese. Para
o conjunto MVP convexo a tese atual preserva a Baseline 1
(doc 03 — secção 17.4) como referência determinística. A
Fase C é registada como evidência negativa exploratória: a
união canónica XY no referencial do mundo, na forma testada,
foi avaliada e não produziu melhoria face à Baseline 1 nem
face à Fase B sobre o conjunto MVP convexo.
