# 03 — Baseline 1: correspondência geométrica determinística

> Nota de implementação para futura conversão em secção LaTeX.
> Estado: Baseline 1 — perceção e correspondência determinísticas,
> sem componente aprendida.
> Data: 2026-05-02.

---

## 1. Objetivo da fase

Estabelecer uma baseline determinística de correspondência
peça-cavidade baseada em geometria 2D, sobre os artefactos das
Fases 1 (peças) e 2 (cavidades). A baseline serve dois propósitos
explícitos:

- fornecer uma referência de comparação para qualquer método
  aprendido futuro;
- expor problemas de perceção (escala, segmentação, suporte
  amostral) que possam estar mascarados na inspeção visual
  isolada de pegadas individuais.

A baseline **não** classifica formas, **não** usa aprendizagem,
**não** depende de qualquer mapeamento manual peça → cavidade. A
correspondência é descoberta exclusivamente a partir da geometria
real-escala.

---

## 2. Entradas

A baseline lê as nuvens de pontos das Fases 1 e 2:

- `data/pieces_detected/{rectangle, square, circle, star}/piece_pointcloud.npy`
- `data/cavities_detected/cavity_{00,01,02,03}/cavity_pointcloud.npy`

Cada nuvem tem *shape* `(2048, 3)` em metros, com X/Y centrados
no centróide e escala real preservada. Apenas as colunas X e Y
são usadas; Z é descartado nesta baseline (ver secção 9).

Os ficheiros `*_footprint.png` produzidos pelas Fases 1 e 2
**não** são usados pela baseline: são artefactos de visualização.
A baseline rasteriza as suas próprias máscaras a partir dos
pontos, garantindo que ambos os lados (peças e cavidades) usam a
mesma convenção de canvas.

Os nomes das pastas (`rectangle`, `square`, ..., `cavity_00`,
...) são **rótulos de organização** — nunca entram no algoritmo
de correspondência.

---

## 3. Decisões de configuração efectivamente aplicadas

| Constante | Valor | Comentário |
|---|---|---|
| `ROTATION_STEP_DEG` | 2 | 180 ângulos por par peça-cavidade. |
| `WORLD_CANVAS_M` | 0,080 m | Tela de 8 × 8 cm, suficiente para a maior peça. |
| `FOOTPRINT_RESOLUTION_M_PER_PX` | 0,00025 m/px | 0,25 mm/px → tela de 320 × 320 px. |
| `CLEARANCE_DILATION_M` | 0,001 m | 1 mm = 4 px de dilatação da cavidade. Tolerância de perceção/matching, **não** uma folga mecânica validada em CAD. |
| `W_IOU`, `W_INSIDE`, `W_OUTSIDE` | 0,55 / 0,35 / 0,10 | Pesos do score composto. |
| `COMPATIBLE_INSIDE_MIN` | 0,80 | Limiar inferior para inside_ratio. |
| `COMPATIBLE_OUTSIDE_MAX` | 0,20 | Limiar superior para outside_ratio. |
| `COMPATIBLE_IOU_MIN` | 0,55 | Limiar inferior para IoU. |
| `SUSPICIOUS_AREA_RATIO_MAX` | 0,50 | Bandeira de escala suspeita. |
| `LOW_RAW_SUPPORT_AREA_PX` | 200 | Cavidades com menos pixels brutos do que isto recebem `low_raw_support=True`. |
| `TIE_MARGIN` | 0,01 | Margem para empate na seleção da melhor cavidade. |

Estas decisões foram tomadas com base na proposta da revisão
geométrica e nas restrições do projeto. Em particular,
**alpha-shape não foi adotado nesta versão**: optou-se por uma
*pipeline* de rasterização leve (apenas NumPy + OpenCV) com
*hull* convexo como *fallback* explícito.

---

## 4. *Pipeline* de rasterização (peça e cavidade)

A mesma função é aplicada às duas entradas, garantindo
comparabilidade *pixel a pixel*:

1. Eliminação de duplicados/quase-duplicados ao nível de
   meio-pixel (0,125 mm), neutralizando a reamostragem com
   reposição feita nas Fases 1 e 2 quando a máscara bruta tinha
   menos pontos do que `N_POINTS = 2048`.
2. Projeção dos pontos XY na grelha de pixels (320 × 320 px),
   com inversão do eixo Y para coincidir com a convenção das
   imagens de pegada das fases anteriores.
3. *Splat* binário (cada pixel atingido a 255).
4. Operação morfológica *close* com elemento 3 × 3 para juntar
   *pixels* esparsos.
5. Preenchimento dos contornos externos com
   `cv2.findContours(... RETR_EXTERNAL, CHAIN_APPROX_NONE)` e
   `cv2.drawContours(..., thickness=cv2.FILLED)`. Esta escolha
   preserva concavidades do contorno externo (importante para a
   geometria da estrela), enquanto fecha buracos internos
   provocados por *splatting* esparso.
6. *Fallback* de *hull* convexo: se a máscara resultante for
   vazia ou contiver menos de 50 pixels, é construída uma máscara
   alternativa via `cv2.convexHull`. Esta ocorrência é registada
   no metadado do par como `convex_hull_fallback=True`.

A intenção é manter a baseline com dependências mínimas (NumPy +
OpenCV) e evitar tunings de *alpha-shape* nesta primeira
iteração. Se uma forma muito côncava (ex.: estrela) não conseguir
diferenciar-se das demais, *alpha-shape* será considerado numa
iteração futura.

---

## 5. Pesquisa de rotação

A pesquisa é uma **grelha uniforme** de 0° a 360° exclusivos com
passo de 2°: 180 avaliações por par peça-cavidade. A rotação é
aplicada aos **pontos XY** antes da rasterização (não às máscaras
já rasterizadas), evitando artefactos de interpolação.

Custo computacional: 4 peças × 4 cavidades × 180 rotações ≈ 2880
rasterizações; tempo total observado ≈ 5 segundos. Não foi
necessária estratégia *coarse-to-fine*. Não foi usada qualquer
exploração de simetria, porque tal exigiria classificação prévia
da forma — o que está expressamente excluído.

---

## 6. Métricas e *score* composto

Para cada rotação θ, com `P(θ)` = máscara da peça rotacionada e
`C` = máscara da cavidade dilatada por `CLEARANCE_DILATION_M`:

```
inside_ratio  = |P ∩ C| / |P|
outside_ratio = |P ∩ ¬C| / |P|       (= 1 − inside_ratio)
IoU           = |P ∩ C| / |P ∪ C|

score = 0,55 × IoU + 0,35 × inside_ratio − 0,10 × outside_ratio
```

A `area_ratio` é calculada relativamente à máscara da cavidade
**não dilatada**:

```
area_ratio = min(|P|, |C_undilated|) / max(|P|, |C_undilated|)
```

`area_ratio` **não** entra no *score*. É usada apenas como
diagnóstico:

- `suspicious_scale = (area_ratio < 0,50)`.

Cavidades com menos de 200 pixels brutos na metadados de
captura recebem `low_raw_support = True` (atualmente apenas
`cavity_00`).

---

## 7. Critério de compatibilidade na versão atual

Na versão atualmente implementada, uma única flag `compatible` é
posta a `True` se e só se, no ângulo ótimo:

- `inside_ratio ≥ 0,80`,
- `outside_ratio ≤ 0,20`,
- `IoU ≥ 0,55`.

O score, área-rácio e flags `suspicious_scale` /
`low_raw_support` são gravadas em separado. Em caso de empate
entre as duas melhores cavidades para uma peça (margem < 0,01),
a peça é marcada `tie=True` e ambos os candidatos listados.

A análise visual (secção 11) motiva uma proposta de
**reformulação destas flags** para separar "melhor
correspondência geométrica" de "plausibilidade física de escala"
— ver secção 13.

---

## 8. Saídas produzidas

Diretório raiz: `data/baseline1_geometric_matching/`.

**Globais:**

| Ficheiro | Conteúdo |
|---|---|
| `results_all.json` | Serialização completa (cada par peça × cavidade × rotação ótima). |
| `results_matrix.csv` | Matriz 4 × 4 dos *scores* ótimos. |
| `summary.txt` | Resumo legível por humano. |
| `run_metadata.json` | Parâmetros, *timestamps*, caminhos, estado de sucesso. |
| `run_log.txt` | Cópia da consola, sobrescrita a cada execução. |
| `score_matrix_heatmap.png` | Mapa de calor 4 × 4 anotado. |
| `best_match_grid.png` | Grelha 4 × 3: peça | cavidade | sobreposição na rotação ótima. |

**Por peça (`<piece>/`):**

- `best_match.json` — melhor cavidade, melhor rotação, *flags*.
- `ranking.json` — todas as cavidades ordenadas por *score* ótimo.
- `all_cavities_comparison.png` — linha com 4 sobreposições.

**Por par (`<piece>/vs_<cavity>/`):**

- `rotation_scores.csv` — 180 linhas: rotação, inside, outside, IoU, score.
- `pair_summary.json` — resumo do par.
- `overlay_best.png` — sobreposição colorida na rotação ótima.
- `score_curve.png` — curva *score* e IoU vs rotação.

A política de escrita segue a regra das fases anteriores: o
diretório de saída é limpo no início para que ficheiros de
execuções anteriores não possam ser confundidos com o resultado
atual.

---

## 9. Resumo dos resultados de validação

A execução produziu uma **diagonal limpa** na matriz 4 × 4: cada
peça preferiu uma cavidade distinta (sem hipótese imposta a
priori).

| Peça        | Melhor cavidade | Rotação | *Score* | inside | outside | IoU   | area_ratio | suspicious_scale | low_raw_support | compatible |
|-------------|-----------------|---------|---------|--------|---------|-------|------------|------------------|------------------|------------|
| rectangle   | cavity_01        | 90°     | 0,708   | 0,909  | 0,091   | 0,725 | 0,284      | True             | False            | True       |
| square      | cavity_03        | 180°    | 0,855   | 0,968  | 0,032   | 0,945 | 0,303      | True             | False            | True       |
| circle      | cavity_02        | 192°    | 0,837   | 0,969  | 0,031   | 0,911 | 0,315      | True             | False            | True       |
| star        | cavity_00        | 16°     | 0,663   | 0,803  | 0,197   | 0,730 | 0,222      | True             | True             | True       |

Margens entre o melhor e o segundo melhor *score* por peça:

- rectangle: 0,222 (forte);
- square: 0,180 (forte);
- circle: 0,080 (fraca);
- star: 0,101 (fraca).

Todas as quatro peças disparam `suspicious_scale = True`
(area_ratio entre 0,22 e 0,32). Apenas `star ↔ cavity_00`
adiciona `low_raw_support = True`.

---

## 10. Inspeção visual das sobreposições

A inspeção visual das sobreposições produzidas em
`overlay_best.png` confirma:

- **Rectangle vs cavity_01 (90°)** — verde dominante, halo de
  dilatação fino, vermelho confinado às extremidades curtas.
  Eixos longos alinhados. Correspondência fisicamente plausível.
- **Square vs cavity_03** — sobreposição praticamente
  totalmente verde, halo de dilatação mínimo, vermelhos
  residuais nos cantos. Caso mais limpo.
- **Circle vs cavity_02** — verde dominante, halo circular fino,
  pequenos vermelhos no perímetro. Correspondência correta.
- **Star vs cavity_00 (16°)** — o **corpo central** da estrela
  sobrepõe-se ao interior da cavidade (verde no interior, não no
  halo); as **cinco pontas** estão a vermelho fora da cavidade.
  A correspondência **não** está a ser "salva" pela dilatação:
  é geometria real, mas geometria parcial.

A grelha global e a comparação multi-cavidade da estrela
confirmam, adicionalmente, que **as máscaras das peças têm cerca
de 3 × a área das máscaras das cavidades** em todos os pares
ótimos, o que é coerente com `area_ratio ≈ 0,3` e justifica a
flag `suspicious_scale = True` transversal.

---

## 11. Decisão experimental: substituição da estrela por triângulo

Esta secção documenta uma decisão metodológica adotada para a
*MVP* da baseline.

### O que aconteceu com a estrela

A peça `star` foi corretamente detetada na Fase 1, validada
pelos ficheiros de *footprint* e nuvem de pontos, e foi
corretamente associada pela Baseline 1 à única cavidade pequena
disponível, `cavity_00`. No entanto:

- a peça da estrela tem amplitude XY de aproximadamente 20 mm,
  enquanto `cavity_00` tem apenas cerca de 10,7 mm. O
  `area_ratio` é de 0,222 e `suspicious_scale` está ativada;
- `cavity_00` tem apenas 114 *pixels* brutos na máscara de
  segmentação antes da reamostragem com reposição para
  `N_POINTS = 2048`, pelo que `low_raw_support` está ativada;
- visualmente, o corpo central da estrela cabe na cavidade, mas
  as cinco pontas da estrela permanecem fora. A pegada
  efetivamente correspondida é o "miolo convexo" da estrela, e
  não a forma característica completa.

A correspondência ganhou na matriz por dois motivos
combinados:

1. `cavity_00` é a única cavidade pequena, sendo geometricamente
   o único candidato com escala próxima da peça;
2. as restantes cavidades são significativamente maiores, pelo
   que a estrela cabe inteiramente dentro delas com
   `inside_ratio = 1,0`, mas com `IoU` baixo — o termo
   `outside_ratio` neste caso é zero (nada da peça fica fora),
   mas o termo `area_ratio` (não usado no *score*) revela a
   discrepância de tamanho que a IoU penaliza.

### Por que isto é uma limitação do experimento atual

A combinação `suspicious_scale + low_raw_support + margem fraca`
significa que:

- a correspondência é a melhor disponível, mas **não é uma
  correspondência fisicamente plausível** sem uma reconciliação
  de escala entre o lado das peças e o lado das cavidades;
- a estrela introduz **simultaneamente** dois fatores de
  fragilidade (geometria fortemente concava e suporte amostral
  baixo do alvo), tornando difícil isolar o que está a falhar:
  a perceção da peça, a perceção da cavidade, a escala absoluta,
  ou a rasterização da pegada.

Para uma *MVP* da baseline, este acoplamento é prejudicial: o
objetivo da *MVP* é demonstrar o *pipeline* completo
(perceção → representação → correspondência) sob condições
controladas e bem dimensionadas. Conservar a estrela neste
estágio confunde a leitura dos resultados.

### Por que o triângulo é uma escolha melhor para a *MVP*

- Geometria não circular e não retangular, o que mantém a
  pesquisa de rotação relevante (a IoU varia significativamente
  com o ângulo, não havendo simetria rotacional contínua como
  no círculo nem simetria de 90° como no quadrado).
- Geometria **convexa**, evitando os problemas de rasterização
  associados a contornos côncavos.
- Mais simples de validar dimensionalmente em Fusion (três
  vértices, três arestas, ângulos bem definidos) e mais
  previsível na captura *top-down*.
- Permite avaliar a robustez do *score* a peças não retangulares
  sem o ruído das pontas finas da estrela.

### A estrela poderá voltar mais tarde

A estrela é geometricamente um caso de teste interessante para:

- robustez a contornos côncavos (avalia se a rasterização ou um
  futuro *alpha-shape* preserva concavidades);
- robustez a representações de baixa densidade da cavidade
  alvo (`low_raw_support`);
- estudo da dependência do *score* composto em formas onde
  `inside_ratio = 1,0` não implica boa correspondência.

A intenção é reintroduzi-la **após** a baseline estar validada
com geometria controlada e após os problemas de escala terem
sido reconciliados, como **caso de teste de *stress*** do
*pipeline*, possivelmente acompanhada por uma cavidade
alargada/reprojetada para que o `area_ratio` seja plausível.

### Pontos a reter para o relatório

- A baseline **não** usou nenhum mapeamento peça → cavidade. O
  resultado diagonal foi obtido exclusivamente pela geometria
  rasterizada e pela pesquisa de rotação.
- A substituição da estrela pelo triângulo é uma decisão de
  *escopo* da *MVP*, não uma admissão de que a estrela seja
  intratável; é uma decisão de redução de variáveis.
- **Antes de re-executar resultados finais, é necessário
  validar dimensionalmente as peças e as cavidades em Fusion**
  contra as amplitudes XY medidas em
  `data/pieces_detected/validation_summary.csv` e
  `data/cavities_detected/validation_summary.csv`. A
  *suspicious_scale = True* transversal sugere uma das
  seguintes hipóteses, todas a auditar:
  (a) intrínsecos de câmara diferentes entre captura de peças
      e captura de cavidades;
  (b) sub-segmentação dos contornos das cavidades pelo limiar
      de profundidade;
  (c) discrepância dimensional real entre os modelos CAD das
      peças e das cavidades;
  (d) erro sistemático na profundidade da superfície usada
      como referência métrica.

### Dimensões CAD finais para a auditoria de escala

As dimensões CAD do conjunto experimental revisto (estrela
substituída por triângulo) ficam registadas em
`data/expected_cad_dimensions.json` e resumidas abaixo. **Não
são consumidas pelo algoritmo de matching** — servem apenas
para a auditoria de escala referida acima.

Peças (XY nominal × extrusão Z):

| Peça        | XY (mm)                    | Extrusão (mm) |
|-------------|----------------------------|---------------|
| quadrado    | 50 × 50                    | 105           |
| retângulo   | 50 × 75                    | 105           |
| triângulo   | base 50, alt. geom. 50     | 105           |
| círculo     | diâmetro 50                | 105           |

Cavidades (XY nominal × profundidade Z):

| Cavidade    | XY (mm)                    | Profundidade (mm) |
|-------------|----------------------------|-------------------|
| quadrada    | 51 × 51                    | 75                |
| retangular  | 51 × 76                    | 75                |
| triangular  | base 51, alt. geom. 51     | 75                |
| circular    | diâmetro 51                | 75                |

Tabuleiro: espessura 75 mm; dimensões externas ainda por
registar.

Folga (clearance) nominal: **1 mm total, 0,5 mm por lado**.

Comparação direta com o parâmetro da Baseline 1: a constante
`CLEARANCE_DILATION_M = 0,001` (1 mm), originalmente justificada
como tolerância de perceção/matching, **coincide
numericamente** com a folga total CAD. Esta coincidência é
favorável para a *MVP* — a dilatação aplicada na cavidade
(usada para compensar sub-segmentação) tem a mesma ordem de
grandeza da folga mecânica real, pelo que o limiar de
compatibilidade representa um cenário próximo do físico. **Não
deve, contudo, ser tratada como justificação mecânica
validada**: é apenas uma coincidência conveniente; uma futura
folga aplicada mecanicamente diferente exigirá reajuste.

**Decisão experimental confirmada:** a estrela permanece fora
do conjunto principal. Permanece registada em
`expected_cad_dimensions.json` em `optional_stress_test_shapes`
para reintrodução posterior como caso de *stress* concava,
provavelmente acompanhada de uma cavidade dedicada
dimensionalmente compatível (a cavidade `cavity_00` da bancada
anterior tinha 10,7 mm — claramente subdimensionada para a
estrela de 20 mm).

---

## 12. Limitações conhecidas

1. **Z descartado — sem validação de profundidade de
   inserção**. A baseline opera exclusivamente sobre pegadas
   2D no plano XY. As dimensões CAD finais tornam esta
   limitação concreta: a extrusão das peças é de 105 mm e a
   profundidade nominal das cavidades é de 75 mm; uma peça
   inserida até ao fundo sobressai 30 mm acima do topo do
   tabuleiro. A Baseline 1 **não deteta nem pontua** esta
   protrusão. A informação Z já está presente em
   `piece_pointcloud.npy` e `cavity_pointcloud.npy`; uma
   *Baseline 1.5* posterior poderá adicionar uma verificação
   de compatibilidade vertical sem alterar o *pipeline* de
   perceção, comparando a altura observada da peça contra
   `cavity.depth_m` e contra a dinâmica de contacto a modelar.
2. **`suspicious_scale = True` em todos os pares** — sintoma
   forte de problema de perceção a investigar (ver secção 11
   "Pontos a reter").
3. **Margens fracas em duas peças** — `circle` (0,080) e
   `star` (0,101). A pequena margem reflete a proximidade
   geométrica entre quadrado e círculo nesta resolução, e a
   inexistência de uma cavidade verdadeiramente
   estrela-compatível. A primeira é resolvida com mais
   resolução; a segunda com a substituição metodológica
   descrita na secção 11.
4. **Identidade das cavidades é posicional** — `cavity_NN` não
   é semântica. Reordenações da cena alteram os
   identificadores; a baseline é insensível a isto, mas o
   *output* humano pode parecer diferente entre execuções.
5. **`alpha-shape` não usado nesta versão** — concavidades
   internas podem ser perdidas se aparecerem em casos futuros.
6. **`CLEARANCE_DILATION_M = 1 mm` é uma tolerância de
   correspondência**, **não** uma folga mecânica validada
   contra os modelos CAD.
7. **Pesos do *score* compostos foram fixados *a priori* e
   não foram ajustados a este conjunto de 4 peças** — não há
   sobreajuste, mas também não foi feito qualquer estudo de
   sensibilidade a estes pesos.

---

## 13. Proposta de reformulação das *flags* de compatibilidade

Decorrente da inspeção visual (secção 10), propõe-se separar a
*flag* `compatible` em três conceitos não conflitantes:

| *Flag* | Definição | Estado atual nas 4 peças |
|---|---|---|
| `geometric_best_match` | Esta cavidade é a primeira no *ranking* desta peça (diagonal da matriz). | True para todas. |
| `physical_scale_plausible` | `area_ratio ≥ 0,50` e `suspicious_scale = False`. Indica que a escala real entre peça e cavidade é coerente. | False para todas. |
| `margin_weak` | `score_melhor − score_segundo < 0,10`. | True para `circle` e `star`. |

Esta reformulação evita que `compatible = True` seja interpretado
como "pronto para inserção robotizada" quando, na realidade, a
escala absoluta ainda está por reconciliar. A alteração afeta
apenas os metadados de saída e não invalida resultados já
gravados; pode ser aplicada na próxima iteração da baseline.

---

## 14. Relevância para o objetivo da tese

A Baseline 1 não é a abordagem aprendida pretendida pela tese,
mas serve três funções específicas no plano global:

- **Referência de comparação** para qualquer futuro método de
  correspondência aprendido. Espera-se que o método aprendido
  iguale ou supere a Baseline 1 nas peças bem dimensionadas e
  que melhore especificamente os casos onde a Baseline 1
  apresenta margens fracas.
- **Diagnóstico do *pipeline* de perceção**. O facto de
  `suspicious_scale = True` ser disparado em todas as
  correspondências é um sinal que só emerge quando se compara
  diretamente, em escala real, ambos os lados — sinal este que
  não é detetável apenas a partir da inspeção isolada das
  Fases 1 e 2.
- **Definição operacional do que conta como "compatibilidade"**
  de inserção em termos puramente geométricos. O método e os
  limiares aqui propostos serão a base contra a qual qualquer
  noção mais sofisticada (incluindo *affordances* aprendidas
  em fases futuras) será confrontada.

---

## 15. Figuras a incluir mais tarde em LaTeX

Ver `docs/thesis_notes/figures_index.md` para a tabela
consolidada. As figuras mais relevantes desta fase são:

- `data/baseline1_geometric_matching/score_matrix_heatmap.png`
- `data/baseline1_geometric_matching/best_match_grid.png`
- `data/baseline1_geometric_matching/star/all_cavities_comparison.png`
- `data/baseline1_geometric_matching/star/vs_cavity_00/overlay_best.png`
- `data/baseline1_geometric_matching/rectangle/vs_cavity_01/overlay_best.png`
- `data/baseline1_geometric_matching/square/vs_cavity_03/overlay_best.png`
- `data/baseline1_geometric_matching/circle/vs_cavity_02/overlay_best.png`

---

## 16. Estado actual e protocolo de re-execução

Os resultados desta secção (matriz 4×4, *overlays*, *flags* de
compatibilidade) foram produzidos com o conjunto experimental
**anterior** — `rectangle, square, circle, star` — e antes das
correcções introduzidas no *script* de captura de peças
(controlo de câmara via *stage*, estimação por
`auto_depth_layers`, projecção por *pixel* dependente da
profundidade). **Devem ser tratados como diagnóstico
intermédio**, e não como o resultado final da Baseline 1.

### 16.1 Decisões consolidadas

- **Conjunto principal**: `rectangle, square, circle, triangle`.
- **`star`**: removida do conjunto principal por ser
  excessivamente sensível a segmentação e a escala absoluta
  (ver secção 11). Permanece registada em
  `data/expected_cad_dimensions.json` em
  `optional_stress_test_shapes` como caso de *stress* concava
  reservado para trabalho futuro. **Não** entra na re-execução
  da Baseline 1.
- **Sem mapeamento peça→cavidade** em qualquer parte do
  algoritmo (princípio mantido).

### 16.2 Pré-condições para a re-execução

A re-execução deve ocorrer **apenas após** o seguinte
encadeamento (ver doc 01 — secção 18.11 para o protocolo
completo):

1. Corrigir o cálculo da *focal* vertical em *pixels*
   (`fy_px`) em
   `scripts/capture_piece_detection.py` (e em
   `scripts/capture_cavity_detection.py` se partilhar a mesma
   fórmula — ver doc 02 — secção 19).
2. Recapturar as quatro peças do conjunto principal e
   re-validar com `scripts/validate_piece_captures.py`.
3. Recapturar as cavidades, se aplicável, e re-validar.
4. Auditoria de escala contra
   `data/expected_cad_dimensions.json`: amplitudes XY e
   `piece_height_median_m` devem situar-se dentro de uma margem
   de tolerância pré-definida (sugestão: ≤ 2 % de erro relativo,
   uma vez removido o viés sistemático actual de Y).
5. Só então re-executar `scripts/baseline1_geometric_matching.py`
   com o conjunto `rectangle, square, circle, triangle`.

### 16.3 Resultado actual — preservar como diagnóstico

A matriz 4×4, os *overlays* e a discussão das secções 9–11
permanecem documentados como **resultado intermédio**. Servem
três propósitos no relatório:

1. mostrar que a Baseline 1 descobriu correctamente a
   diagonal (rectangle ↔ cavity_01, square ↔ cavity_03, etc.)
   sem qualquer mapeamento manual, prova de conceito do método
   determinístico;
2. expor o sintoma `suspicious_scale = True` transversal,
   diagnóstico que motivou a investigação posterior (estimação
   de superfície, projecção por *pixel*);
3. fundamentar a decisão experimental de retirar a `star` do
   conjunto principal (secção 11).

Estes resultados **não** devem ser citados como medida de
desempenho da abordagem determinística no relatório final.
Quando a re-execução posterior estiver disponível, deve ser
publicada como **resultado de referência** e este resultado
intermédio deve ser claramente etiquetado como "pré-correcções"
ou "diagnóstico intermédio".

### 16.4 Outputs a re-gerar

A próxima execução produzirá um conjunto novo dos mesmos
artefactos em `data/baseline1_geometric_matching/`. Para evitar
confusão entre execuções, recomenda-se:

- antes da re-execução, copiar `results_matrix.csv`,
  `results_all.json`, `summary.txt`, `score_matrix_heatmap.png`,
  `best_match_grid.png` e `run_log.txt` para um directório de
  arquivo (por exemplo
  `data/baseline1_geometric_matching/_archive/2026-05_star_set/`)
  para preservar o registo intermédio;
- só depois deixar que `scripts/baseline1_geometric_matching.py`
  sobrescreva os artefactos canónicos.

A nomenclatura `cavity_NN` continua posicional e
determinística por construção; a substituição da `star` pelo
`triangle` no lado das peças não altera a numeração das
cavidades, mas pode alterar qual cavidade ganha cada *match* se
a escala for substancialmente diferente.

---

## Notas para o autor

Itens a registar manualmente, fora deste documento:

- Dimensões CAD nominais das peças e das cavidades em Fusion,
  com unidades explícitas, para suportar a auditoria de escala
  proposta na secção 11.
- Justificação do conjunto inicial revisto de peças
  (rectangle, square, circle, **triangle**) — porquê estas e
  não outras.
- Especificação USD da câmara da Fase 1 vs Fase 2 (focal,
  abertura, pose), para descartar/confirmar a hipótese de
  intrínsecos divergentes entre as duas capturas.
- Resultado da próxima execução da baseline com triângulo, com
  comparação direta aos números desta execução.
- Decisão final sobre se a estrela volta a entrar como caso de
  *stress* e em que fase do trabalho.
