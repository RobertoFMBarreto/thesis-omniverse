# 02 — Deteção automática do tabuleiro e das cavidades

> Nota de implementação para futura conversão em secção LaTeX.
> Estado: Fase 2 — perceção determinística, sem componente aprendida.
> Data: 2026-05-01.

---

## 1. Objetivo da fase

Esta fase tem como objetivo detetar automaticamente o tabuleiro
sobre a bancada e, dentro do tabuleiro, segmentar cada cavidade
geométrica como uma região de profundidade negativa relativamente
à face superior do tabuleiro. Para cada cavidade detetada são
exportados os artefactos geométricos que mais tarde alimentarão a
baseline determinística de correspondência peça-cavidade
(*footprint matching*).

A fase **não** classifica formas de cavidades, **não** estabelece
qualquer correspondência peça-cavidade, **não** envolve modelos
aprendidos, e **não** controla o robô. Trata-se exclusivamente de
um passo de perceção geométrica baseado em profundidade.

---

## 2. Contexto experimental

A bancada virtual no NVIDIA Isaac Sim 5.1 contém:

- uma mesa/plano de fundo;
- um tabuleiro retangular sobre a mesa, com cavidades passantes;
- a vista da câmara é aproximadamente *top-down* sobre o
  tabuleiro;
- a câmara é a mesma usada na Fase 1, mas aqui reposicionada
  diretamente no *stage* USD para enquadrar o tabuleiro inteiro.

A captura é orquestrada via *Script Editor* do Isaac Sim, com o
mesmo padrão assíncrono e os mesmos anotadores `rgb` e
`distance_to_image_plane` da Fase 1.

---

## 3. Modelo geométrico da cena

A interpretação de profundidade adotada distingue três níveis,
ordenados pela distância à câmara (do mais próximo ao mais
afastado):

1. **Topo do tabuleiro** — distância à câmara mais pequena, porque
   o tabuleiro está elevado em relação à mesa pela sua espessura.
2. **Mesa/fundo** — distância maior, em torno do tabuleiro.
3. **Pixels no interior das cavidades** — distância
   aproximadamente igual à da mesa, porque a câmara observa o
   plano da mesa através do furo da cavidade.

Esta ordenação justifica todas as decisões de segmentação que se
seguem: o tabuleiro é o que está acima da mesa; as cavidades são
buracos no tabuleiro que voltam a expor a profundidade da mesa.

---

## 4. Aquisição RGB-D no Isaac Sim

O módulo de captura é implementado em
`scripts/capture_cavity_detection.py` e segue o padrão validado na
Fase 1:

- Criação de um *render product* sobre a câmara configurada.
- Anexação dos anotadores `rgb` e `distance_to_image_plane`.
- Execução de um passo de simulação assíncrono com
  `await rep.orchestrator.step_async(rt_subframes=RT_SUBFRAMES)`.
- Leitura defensiva via `get_data()`, com normalização do formato
  *ndarray*-ou-dicionário entre versões do Replicator.

Por omissão, a câmara **não é movida** pelo *script* (variável
`SET_CAMERA_POSE = False`): a pose autorizada no *stage* USD é
considerada autoritativa. O *script* limita-se a ler a posição
mundial da câmara via `get_camera_world_pose()` e a usá-la na
projeção inversa para coordenadas mundiais. Esta decisão evita
sobreposição inadvertida da pose da câmara da Fase 1 sobre a
configuração visualmente verificada para a Fase 2.

---

## 5. Estimação automática da profundidade da mesa/fundo

Antes de qualquer segmentação do tabuleiro, é estimada a
profundidade dominante do plano de fundo (mesa) através do
histograma da imagem completa de profundidade no intervalo
configurável `[SURFACE_DEPTH_MIN, SURFACE_DEPTH_MAX]`. O modo
dominante é assumido como a profundidade da mesa, dado que numa
cena *top-down* com tabuleiro pequeno relativamente ao campo de
visão, a mesa ocupa a fração maior dos *pixels* válidos.

A profundidade da mesa serve como **referência negativa**: o
tabuleiro estará sistematicamente mais próximo da câmara do que
este valor.

---

## 6. Deteção automática do tabuleiro

A deteção do tabuleiro é executada em quatro etapas
deterministas:

1. **Máscara de candidatos**: *pixels* cuja profundidade é menor
   do que `table_depth - BOARD_ABOVE_TABLE_MARGIN` (a margem por
   omissão é 5 mm, devendo ser inferior à espessura física do
   tabuleiro).
2. **Componentes ligados** sobre essa máscara, com filtro de área
   `[BOARD_MIN_AREA_PX, BOARD_MAX_AREA_PX]`.
3. **Filtro de retangularidade**: rácio
   `area / area_da_caixa_envolvente >= BOARD_RECTANGULARITY_MIN`.
   Um tabuleiro retangular pleno aproxima-se de 1; o limiar
   defensivo é 0,70, suficiente para tolerar bordas com algum
   ruído sem aceitar componentes manifestamente irregulares.
4. **Seleção do candidato dominante** entre os que passam os
   filtros (maior área).

A máscara resultante (`board_mask`) representa apenas a face
superior do tabuleiro; tem **buracos com a forma das cavidades**.
Para a deteção subsequente das cavidades é construída uma versão
**preenchida** dessa máscara — `board_region_mask` — através de
`BOARD_FILL_MODE`:

- `"contour"` (preferido): o maior contorno externo é desenhado
  preenchido com `cv2.drawContours(..., thickness=cv2.FILLED)`;
- `"bbox"` (alternativa robusta): preenchimento da caixa
  envolvente do tabuleiro.

`board_region_mask` é o **domínio de pesquisa** das cavidades: é
geometricamente impossível existir uma cavidade fora desta região.

---

## 7. Estimação da profundidade do topo do tabuleiro

A profundidade da face superior do tabuleiro é estimada como o modo
do histograma de profundidade restringido aos *pixels* de
`board_mask` (i.e., apenas a face superior visível, excluindo as
cavidades). Esta restrição é determinante: alimentar o histograma
com a imagem inteira fá-lo-ia bloquear no plano da mesa, não no
topo do tabuleiro.

O valor obtido é `board_surface_z`. Quando a fração do *bin*
dominante é inferior a um limiar de aviso, o *script* emite uma
mensagem indicando estimativa potencialmente ruidosa.

---

## 8. Segmentação de cavidades

Uma cavidade é, por construção, uma região cuja profundidade está
**abaixo** da face do tabuleiro mas dentro de uma janela
plausível. A regra é:

```
board_surface_z + CAVITY_DEPTH_MARGIN  <  depth  <  board_surface_z + MAX_CAVITY_DEPTH
```

e é aplicada **apenas no interior de `board_region_mask`** (operação
booleana AND). A janela elimina, simultaneamente, ruído da
superfície do tabuleiro (limite inferior) e furos demasiado
profundos ou ruído proveniente de outras superfícies (limite
superior).

Após o limiar é aplicada uma operação morfológica suave
(*open + close*) para remover pequenos *speckles* sem destruir
contornos finos.

---

## 9. Identificação e ordenação de cavidades

Sobre a máscara binária resultante é executada análise de
componentes ligados, com filtro de área
`[CC_MIN_AREA_PX, CC_MAX_AREA_PX]`.

A ordenação final das cavidades é determinística e documentada,
para que `cavity_00`, `cavity_01`, ... mantenham correspondência
entre execuções desde que a câmara não se mova:

1. *Bin* da coordenada y do centróide em linhas de `ROW_BIN_PX`
   *pixels* (tolerância para cavidades nominalmente alinhadas);
2. Ordenação por `(row_bin, centroide_x)` — linha-a-linha, do
   topo para a base, e dentro de cada linha da esquerda para a
   direita.

Os identificadores `cavity_NN` são, portanto, **identificadores
espaciais determinísticos**, **nunca** rótulos semânticos.

---

## 10. Geração de nuvens de pontos e pegadas por cavidade

Para cada cavidade aceite, é gerada uma representação geométrica
local com a mesma convenção da Fase 1, com a única diferença de
sinal em Z:

- **X e Y**: centrados no centróide mundial da cavidade, em
  metros, com escala real preservada.
- **Z**: **profundidade abaixo do topo do tabuleiro**, calculada
  como `depth[pixel] - board_surface_z`; é, por construção,
  positiva (mais profunda = maior valor).
- **Amostragem fixa**: cada nuvem contém exatamente `N_POINTS =
  2048` pontos. Quando a máscara contém menos *pixels* do que
  `N_POINTS`, é feita amostragem com reposição. Esta replicação
  fica registada nos metadados.
- **Pegada 2D** *top-down*: 256 × 256 *pixels* a 0,5 mm/*pixel*.

A intrínseca de câmara é avaliada na profundidade do topo do
tabuleiro, garantindo coerência métrica para o cálculo de
metros-por-pixel à profundidade efetivamente observada.

---

## 11. Saídas guardadas

Cada execução produz, em `data/cavities_detected/`, um conjunto de
saídas globais e uma subpasta por cavidade detetada.

**Saídas globais:**

| Ficheiro | Conteúdo |
|---|---|
| `rgb.png` | Imagem a cores capturada. |
| `depth_vis.png` | Visualização colorida da imagem de profundidade. |
| `board_mask.png` | Face superior do tabuleiro (com buracos das cavidades). |
| `board_region_mask.png` | Tabuleiro preenchido (cavidades incluídas — domínio de pesquisa). |
| `board_debug.png` | Sobreposição RGB com tabuleiro tingido, contorno preenchido, caixa e centróide. |
| `board_roi_auto_debug.png` | Diagnóstico da deteção do tabuleiro (gravado também em caso de falha). |
| `raw_cavity_mask.png` | Máscara binária após limiar de profundidade restringido a `board_region_mask`. |
| `cavities_debug.png` | Sobreposição RGB com cada cavidade detetada tingida e numerada. |
| `cavities_summary.json` | Metadados globais (tabuleiro, parâmetros, lista de cavidades, componentes rejeitados). |
| `run_log.txt` | Cópia da saída de consola, sobrescrito a cada execução. |

**Saídas por cavidade**, em `cavity_NN/`:

| Ficheiro | Conteúdo |
|---|---|
| `cavity_mask.png` | Máscara binária da cavidade. |
| `cavity_debug.png` | Sobreposição da máscara sobre a imagem RGB. |
| `cavity_footprint.png` | Pegada 2D *top-down*. |
| `cavity_pointcloud.npy` | Nuvem de pontos 3D em metros, *shape* `(2048, 3)`. |
| `cavity_metadata.json` | Metadados da cavidade individual. |

A política de escrita preserva a regra adotada na Fase 1:
ficheiros de execuções anteriores são removidos no início; em caso
de falha, **não** são produzidos *placeholders*; o
`cavities_summary.json` é sempre escrito, com `success=False` e
mensagem de erro quando aplicável.

---

## 12. Procedimento de validação

`scripts/validate_cavity_captures.py` corre fora do Isaac Sim em
Python convencional. Para cada subpasta `cavity_NN/`, verifica:

1. presença dos ficheiros (`cavity_metadata.json`,
   `cavity_pointcloud.npy`, `cavity_footprint.png`,
   `cavity_debug.png`, `cavity_mask.png`);
2. estrutura da nuvem (dimensão 2, segunda dimensão 3, ≥ 100
   pontos);
3. validade numérica (sem NaN, sem infinitos);
4. limites geométricos: amplitudes X e Y positivas, `Z máximo > 0`,
   amplitude em Z não-negativa;
5. *footprint* legível e não vazio;
6. coerência com os campos de metadados (`cavity_id`,
   `centroid_world_m`, `xy_span_m`, `z_depth_range_m`).

Verifica também, no nível global, a presença das saídas globais
listadas acima (incluindo `cavities_summary.json` e `run_log.txt`).

São produzidos:

- `data/cavities_detected/validation_summary.json`
- `data/cavities_detected/validation_summary.csv`
- `data/cavities_detected/footprints_grid.png` — grelha com as
  pegadas de todas as cavidades, rotuladas por `cavity_NN`.

---

## 13. Resumo dos resultados de validação

Foram detetadas e validadas com sucesso **4 cavidades**, todas
passando todos os critérios de validação. Cada nuvem de pontos
contém 2048 pontos; sem NaN nem infinitos; *footprints*
não-vazios. A tabela seguinte resume as métricas geométricas
extraídas de `data/cavities_detected/validation_summary.csv`.

| Cavidade   | Área (px) | Amplitude X (mm) | Amplitude Y (mm) | Amplitude Z (mm) | Pontos |
|------------|-----------|------------------|------------------|------------------|--------|
| cavity_00  | 114       | 10,67            | 10,77            | 20,0             | 2048   |
| cavity_01  | 897       | 19,57            | 31,48            | 16,1             | 2048   |
| cavity_02  | 383       | 18,68            | 17,40            | 17,3             | 2048   |
| cavity_03  | 506       | 19,57            | 17,40            | 14,0             | 2048   |

A inspeção visual manual da grelha de pegadas confirmou que
`cavity_00` corresponde à cavidade da estrela e não a ruído. As
restantes cavidades têm áreas e amplitudes coerentes com peças
nominais retangulares e quadradas/circulares.

---

## 14. Problemas encontrados e correções

Esta secção documenta os problemas técnicos efetivamente
observados durante o desenvolvimento da Fase 2 e as correções
aplicadas.

1. **Sobreposição indevida da pose da câmara**.
   A primeira versão do *script* movia a câmara para a pose usada
   na Fase 1 (peças), apesar de a câmara estar corretamente
   posicionada sobre o tabuleiro no *stage* USD.
   *Correção*: introdução de `SET_CAMERA_POSE = False` por
   omissão e introdução de `get_camera_world_pose()` para que a
   projeção inversa use a pose efetiva do *prim*. Adição de aviso
   quando as constantes de pose ficam compatíveis com a antiga
   pose da Fase 1.

2. **Estimação errada da superfície do tabuleiro**.
   A primeira versão estimava a profundidade dominante a partir
   da imagem completa. Como a mesa ocupa a maior parte do campo
   de visão, o pico do histograma correspondia à mesa e não ao
   topo do tabuleiro, invalidando a janela de profundidade
   utilizada na segmentação das cavidades.
   *Correção*: deteção automática prévia do tabuleiro e
   reescrição de `estimate_board_surface_depth` para aceitar
   `board_mask` como restrição do histograma, garantindo que
   apenas *pixels* do topo do tabuleiro contribuem para a
   estimativa.

3. **Dependência de uma ROI manual configurada**.
   A versão anterior dependia de um ROI ajustado por constantes
   `BOARD_ROI_*`. Esta abordagem é frágil em alterações da
   câmara ou da cena.
   *Correção*: introdução do *pipeline* automático de deteção do
   tabuleiro descrito na secção 6. As constantes `BOARD_ROI_*`
   ficam preservadas como caminho alternativo apenas quando
   `AUTO_DETECT_BOARD = False`.

4. **Cavidades desencontradas por filtro de área**.
   Numa execução, foram identificados 4 componentes ligados mas
   apenas 3 passaram o filtro de área. O componente rejeitado
   tinha área de 114 *pixels* (a cavidade da estrela) e era
   rejeitado pelo limiar `CC_MIN_AREA_PX = 200`.
   *Correção*: redução para `CC_MIN_AREA_PX = 80`, com
   comentário explícito a justificar o valor — suficientemente
   pequeno para preservar cavidades pequenas como a estrela, e
   ainda assim acima do limiar típico de *speckle* de
   profundidade. Foram acrescentados diagnósticos detalhados
   (lista completa de componentes com motivo de rejeição) à
   consola e a `cavities_summary.json` para tornar este tipo de
   ajuste futuro mais simples e auditável.

5. **Necessidade de registo de execução para reprodutibilidade**.
   A reprodução de problemas dependia de copiar manualmente a
   saída da consola do *Script Editor*.
   *Correção*: adição de `setup_run_logging()` com uma classe
   `_TeeStream` que duplica `sys.stdout`/`stderr` para
   `data/cavities_detected/run_log.txt`, sobrescrito a cada
   execução. A solução é idempotente entre execuções consecutivas
   no mesmo processo do *Script Editor* (não empilha *wrappers*).

6. **Ambiguidade nas saídas em caso de falha**.
   Tal como na Fase 1, foram detetadas situações em que
   *placeholders* de zeros eram gravados após falhas, induzindo
   em erro a análise posterior.
   *Correção*: aplicação da mesma política da Fase 1 — só são
   gravados artefactos efetivamente produzidos, *cleanup* prévio
   removendo ficheiros e subpastas `cavity_*` da execução
   anterior, e `cavities_summary.json` sempre escrito com
   `success=False` e mensagem de erro se aplicável.

---

## 15. Limitações da abordagem atual

1. **Cobertura geométrica das cavidades é parcial**.
   A nuvem de pontos é construída a partir de uma única vista
   *top-down*; paredes laterais e fundo das cavidades só são
   parcialmente observáveis. A representação resultante é
   essencialmente uma *2.5D heightmap* da abertura.

2. **Cavidades muito pequenas têm baixa densidade efetiva**.
   `cavity_00` foi reconstruída a partir de 114 *pixels*
   reamostrados com reposição até 2048 pontos. A nuvem é
   utilizável para metodologias baseadas em pegada, IoU ou
   *Chamfer*, mas **não** deve ser usada num critério baseado
   em densidade local de pontos.

3. **Sensibilidade a parâmetros de profundidade**.
   `BOARD_ABOVE_TABLE_MARGIN`, `CAVITY_DEPTH_MARGIN`,
   `MAX_CAVITY_DEPTH`, `CC_MIN/MAX_AREA_PX` e
   `BOARD_RECTANGULARITY_MIN` continuam a depender da geometria
   e da resolução escolhida; cenas substancialmente diferentes
   exigirão ressintonização.

4. **Modelo geométrico assume tabuleiro elevado**.
   A premissa "tabuleiro mais próximo da câmara do que a mesa"
   é estrutural. Cenários com tabuleiro embutido, ao nível da
   mesa, ou observado sob ângulos oblíquos exigirão um esquema
   alternativo de segmentação.

5. **`board_region_mask` por contorno externo**.
   A versão `"contour"` usa o maior contorno externo do
   tabuleiro. Para tabuleiros com geometria não-simplesmente
   conexa (ex. abertura interna grande), pode não corresponder à
   intuição. O modo `"bbox"` é alternativa permissiva mas pode
   incluir ruído ao redor das bordas.

6. **Identificadores `cavity_NN` são posicionais, não
   semânticos**. A ordenação é determinística sob câmara fixa,
   mas reordena-se se a câmara ou o tabuleiro forem deslocados.
   Para qualquer trabalho futuro que dependa de identidade
   estável das cavidades, esta limitação tem de ser
   explicitamente abordada (e.g., correspondência geométrica em
   vez de índice).

---

## 16. Relevância para o objetivo da tese

Tal como na Fase 1, esta fase **não** corresponde à abordagem
aprendida pretendida pela tese. O seu papel é preparar o terreno:

- Fornecer representantes geométricos das cavidades (pegada,
  nuvem de pontos com escala real preservada, máscara), em
  paridade com os representantes das peças produzidos na Fase 1.
- Permitir, na fase seguinte, a construção de uma **baseline
  determinística** de correspondência peça-cavidade — por
  exemplo, comparação de pegadas via IoU ou distância de Chamfer
  sob rotações candidatas.
- Estabelecer uma fonte de dados anotada *de facto*
  geometricamente (e não por etiquetas humanas) para confronto
  posterior com métodos aprendidos.

Pontos importantes a manter explícitos no relatório:

- a fase **não** classifica cavidades;
- os identificadores `cavity_NN` **não** são rótulos semânticos;
- a escala real está preservada, condição necessária para
  qualquer raciocínio posterior sobre inserção;
- a possibilidade de *multi-view* permanece como extensão futura
  para enriquecer a representação 3D das cavidades, em particular
  das paredes laterais.

---

## 17. Figuras a incluir mais tarde em LaTeX

Ver `docs/thesis_notes/figures_index.md` para a tabela
consolidada de figuras candidatas (Fases 1 e 2) com identificadores,
caminhos, nomes propostos para LaTeX e legendas em português de
Portugal. As figuras mais relevantes desta fase são:

- `data/cavities_detected/board_debug.png` — ilustração da deteção
  automática do tabuleiro;
- `data/cavities_detected/board_mask.png` e
  `board_region_mask.png` — pares que ilustram a diferença entre
  superfície detetada e domínio preenchido;
- `data/cavities_detected/raw_cavity_mask.png` — resultado da
  segmentação por profundidade restringida;
- `data/cavities_detected/cavities_debug.png` — vista global das
  cavidades detetadas com identificadores;
- `data/cavities_detected/footprints_grid.png` — pegadas das
  cavidades em grelha rotulada.

---

## Notas para o autor

Itens que devem ser registados manualmente, fora deste documento, e
que não são captados nos ficheiros de saída:

- Espessura física do tabuleiro modelado em Fusion (para
  justificar `BOARD_ABOVE_TABLE_MARGIN`).
- Profundidade física das cavidades (para confirmar a janela
  `[CAVITY_DEPTH_MARGIN, MAX_CAVITY_DEPTH]`).
- Pose física da câmara virtual no USD (translação e orientação)
  no momento da captura validada.
- Versão exata do Isaac Sim e do contentor.
- Justificação do número e da disposição das cavidades no
  tabuleiro.
- Decisão sobre a estratégia de identidade de cavidades a usar em
  fases posteriores (índice posicional vs. correspondência
  geométrica vs. outra).
