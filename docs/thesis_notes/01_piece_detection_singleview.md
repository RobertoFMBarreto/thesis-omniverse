# 01 — Detecção de peça (vista única)

> Nota de implementação para futura conversão em secção LaTeX.
> Estado: Fase 1 — perceção determinística, sem componente aprendida.
> Data: 2026-05-01.

---

## 1. Objetivo da fase

Esta fase tem como objetivo obter uma representação geométrica fiável de uma
peça visível na cena, a partir de uma única captura RGB-D em ambiente
simulado. Pretende-se construir os artefactos de entrada que mais tarde
alimentarão a baseline determinística de correspondência peça-cavidade
(*footprint matching*).

A fase **não** classifica formas, **não** infere afinidades de inserção e
**não** envolve qualquer modelo aprendido. Trata-se exclusivamente de um
passo de perceção geométrica.

---

## 2. Contexto experimental

O sistema completo é inspirado nos brinquedos infantis de classificação de
formas (*shape sorter*). A bancada experimental contém:

- um tabuleiro com cavidades geométricas;
- peças geométricas modeladas em Fusion (rectângulo, quadrado, círculo e
  estrela), aqui usadas como conjunto inicial;
- uma câmara RGB-D virtual no NVIDIA Isaac Sim 5.1, executado em contentor
  e acedido através do cliente WebRTC.

A captura é orquestrada via *Script Editor* do Isaac Sim, usando o padrão
assíncrono compatível e os anotadores `rgb` e `distance_to_image_plane` do
módulo `omni.replicator.core`.

---

## 3. Dados de entrada e pressupostos da cena

Pressupostos atuais para a Fase 1:

- **Vista única**: a câmara está colocada acima do tabuleiro, com
  orientação aproximadamente vertical (*top-down*).
- **Peça única visível por captura**: as restantes peças são manualmente
  ocultadas no Isaac Sim. Esta restrição é uma decisão experimental, não
  uma limitação intrínseca do detetor — o algoritmo deteta múltiplos
  componentes ligados, mas o pipeline atual seleciona apenas um.
- **Superfície de suporte aproximadamente plana** dentro do campo de
  visão.
- **Escala real preservada**: profundidade em metros, sem normalização
  global.

Os nomes das pastas das capturas (`rectangle/`, `square/`, `circle/`,
`star/`) são **rótulos de organização experimental** e não influenciam
nenhuma decisão geométrica do *script*. O detetor desconhece a forma
nominal da peça.

---

## 4. Aquisição RGB-D no Isaac Sim

O módulo de captura é implementado em `scripts/capture_piece_detection.py`
e segue o padrão recomendado para o *Script Editor*:

- Criação de um *render product* sobre a câmara configurada
  (`/World/Camera`).
- Anexação dos anotadores `rgb` (imagem a cores) e
  `distance_to_image_plane` (profundidade em metros, distância ao plano
  da imagem).
- Execução de um passo de simulação assíncrono via
  `rep.orchestrator.step_async(rt_subframes=RT_SUBFRAMES)`, com
  `RT_SUBFRAMES = 8` para estabilizar o *render*.
- Leitura dos dados via `get_data()`, com normalização defensiva entre
  os formatos *ndarray* e dicionário com chave `"data"` que diferentes
  versões do Replicator podem retornar.

A pose da câmara é definida programaticamente através de constantes
`CAM_X`, `CAM_Y`, `CAM_Z`, `CAM_ROT_Z_DEG`. A função `setup_camera`
suporta as operações de transformação USD `xformOp:orient`,
`xformOp:rotateXYZ` e `xformOp:rotateZ`.

**Resolução por omissão**: 640 × 480 *pixels*.
**Intrínsecos**: distância focal de 24 mm e abertura horizontal de
36 mm, definidos como constantes e correspondendo aos atributos
`UsdGeom.Camera` do *prim* da câmara.

---

## 5. Estimação da superfície de suporte

A profundidade da superfície de suporte (mesa/tabuleiro) é estimada
automaticamente a partir do histograma da imagem de profundidade,
restringido ao intervalo configurável `[SURFACE_DEPTH_MIN,
SURFACE_DEPTH_MAX]` (por omissão `[0.10, 0.50]` m).

O valor estimado é o centro do *bin* dominante, com largura de *bin*
de 1 mm. O algoritmo emite um aviso explícito quando a fração de
*pixels* no *bin* dominante é inferior a 5 %, sinalizando histograma
ruidoso ou superfície de suporte mal enquadrada.

Esta abordagem assume que a superfície de suporte ocupa uma fração
não desprezável do campo de visão. Se a peça preencher quase todo o
campo, o pico do histograma pode corresponder à própria peça e não à
mesa.

---

## 6. Segmentação da peça

A segmentação opera sobre a imagem de profundidade, **não** sobre a
imagem RGB. A peça é tratada como geometria positiva acima da
superfície de suporte: um *pixel* pertence à peça se a sua
profundidade for inferior à profundidade da superfície subtraída de
uma tolerância configurável `SURFACE_TOLERANCE` (por omissão 4 mm).

Esta convenção respeita a geometria do mundo: distância à câmara
menor significa mais próximo da câmara, ou seja, mais alto em
relação à mesa numa vista *top-down*.

A escolha da tolerância é um compromisso:

- valores demasiado pequenos deixam passar ruído da própria
  superfície de suporte;
- valores demasiado grandes eliminam peças finas ou prismas baixos.

---

## 7. Seleção do componente ligado

Após a segmentação binária, é executada análise de componentes
ligados (`cv2.connectedComponentsWithStats`) com filtragem por área
(`CC_MIN_AREA_PX`, `CC_MAX_AREA_PX`).

Os componentes válidos são ordenados de forma determinística por
área decrescente, com desempate por coordenada x do centróide
ascendente. A seleção do componente de interesse é controlada por
três modos configuráveis:

- `largest` — escolhe o componente de maior área (rank 0). É o modo
  por omissão e foi usado em todas as capturas validadas.
- `closest_to_center` — escolhe o componente cujo centróide está
  mais próximo do centro da imagem. Útil para capturar uma peça
  específica reposicionando a câmara, sem recurso a classificação.
- `manual_index` — escolhe o componente na posição
  `MANUAL_COMPONENT_INDEX` da lista ordenada.

Em qualquer modo, o número total de componentes válidos detetados é
registado nos metadados (campos `n_valid_components` e
`multiple_valid_components`), permitindo posterior diagnóstico de
ambiguidade na cena.

---

## 8. Geração da nuvem de pontos

A nuvem de pontos é construída por *backprojection* dos *pixels* da
máscara da peça selecionada, usando intrínsecos de câmara *pinhole*
(`mpp_x`, `mpp_y` derivados da focal, abertura e da distância à
superfície estimada `surface_z`).

Convenções:

- **Eixos X e Y**: centrados no centróide mundial da peça, expressos
  em metros.
- **Eixo Z**: representa a altura acima da superfície de suporte,
  calculado como `surface_z - depth[pixel]`. É sempre não-negativo.
- **Escala real preservada**: as coordenadas estão em metros e
  **não** são normalizadas para uma escala unitária. Esta decisão é
  fundamental para a correspondência peça-cavidade futura, em que o
  tamanho absoluto é parte da informação geométrica relevante.
- **Amostragem fixa**: cada nuvem de pontos contém exatamente
  `N_POINTS = 2048` pontos. Quando a máscara contém menos *pixels*
  do que `N_POINTS`, é feita amostragem com reposição (registado nos
  metadados).

A intrínseca é avaliada em `surface_z` (e não numa altura nominal da
câmara) para que a relação metros-por-pixel seja correta à
profundidade efetiva da peça.

---

## 9. Geração da pegada (*footprint*)

A pegada 2D é a projeção *top-down* da nuvem de pontos no plano XY,
renderizada numa tela quadrada de 256 *pixels* com resolução de
0,5 mm por *pixel*. A imagem é guardada com mapa de cores quente
para facilitar inspeção visual.

A pegada é o artefacto principal a ser consumido pela baseline
geométrica determinística da fase seguinte: a comparação por
sobreposição (IoU) ou Chamfer entre a pegada da peça e a pegada das
cavidades, sob diferentes rotações.

---

## 10. Saídas guardadas

Cada captura produz, por omissão, uma subpasta dentro de
`data/pieces_detected/<CAPTURE_NAME>/` com os seguintes ficheiros:

| Ficheiro | Conteúdo |
|---|---|
| `rgb.png` | Imagem a cores capturada. |
| `depth_vis.png` | Visualização colorida da imagem de profundidade. |
| `raw_piece_mask.png` | Máscara binária após o limiar de profundidade. |
| `piece_mask.png` | Máscara do componente ligado selecionado. |
| `piece_debug.png` | Sobreposição da máscara, caixa envolvente e centróide sobre a imagem RGB. |
| `piece_footprint.png` | Pegada 2D *top-down*. |
| `piece_pointcloud.npy` | Nuvem de pontos 3D em metros, *shape* `(N_POINTS, 3)`. |
| `piece_metadata.json` | Metadados completos da captura (parâmetros, intrínsecos, métricas, lista de componentes válidos). |

A política de escrita garante que, em caso de falha numa etapa
intermédia, **não** são produzidos ficheiros placebo: artefactos
inválidos ou de execuções anteriores são removidos no início e
apenas os artefactos efetivamente produzidos na execução atual são
gravados. O `piece_metadata.json` é sempre escrito, com
`success=False` e a mensagem de erro quando aplicável.

---

## 11. Procedimento de validação

Foi implementado um *script* independente,
`scripts/validate_piece_captures.py`, que corre fora do Isaac Sim
em Python convencional. Para cada subpasta de captura esperada
(`rectangle`, `square`, `circle`, `star`), a validação verifica:

1. presença dos ficheiros essenciais (`piece_metadata.json`,
   `piece_pointcloud.npy`, `piece_footprint.png`,
   `piece_debug.png`);
2. coerência dos metadados — em particular, `n_valid_components == 1`
   e `multiple_valid_components == false`;
3. estrutura da nuvem de pontos: dimensão 2, segunda dimensão igual
   a 3, pelo menos 100 pontos;
4. limites geométricos da nuvem: amplitudes em X e Y positivas,
   amplitude em Z não-negativa, ausência de NaN e de infinitos;
5. *footprint* legível e não vazio.

A validação produz três artefactos:

- `data/pieces_detected/validation_summary.json`
- `data/pieces_detected/validation_summary.csv`
- `data/pieces_detected/footprints_grid.png` (grelha 2 × 2 com os
  *footprints* das quatro peças, com rótulo da pasta).

---

## 12. Resumo dos resultados de validação

Todos os critérios passaram para as quatro peças capturadas. O
quadro abaixo resume as amplitudes da nuvem de pontos e a contagem
de pontos, extraídos diretamente de
`data/pieces_detected/validation_summary.json`.

| Peça      | Amplitude X (mm) | Amplitude Y (mm) | Amplitude Z (mm) | Pontos |
|-----------|------------------|------------------|------------------|--------|
| rectangle | 37,7             | 19,7             | 0,0              | 2048   |
| square    | 21,2             | 20,1             | 20,7             | 2048   |
| circle    | 21,2             | 19,7             | 0,0              | 2048   |
| star      | 19,9             | 17,7             | 26,3             | 2048   |

Limites em Z (em metros), também extraídos da validação:

| Peça      | Z mínimo | Z máximo |
|-----------|----------|----------|
| rectangle | 0,03050  | 0,03050  |
| square    | 0,00983  | 0,03050  |
| circle    | 0,03050  | 0,03050  |
| star      | 0,00418  | 0,03050  |

A amplitude Z nula nas peças `rectangle` e `circle` é coerente com
a hipótese de que a face superior dessas peças é estritamente
plana e a única visível numa vista *top-down*. Todos os *pixels*
visíveis projetam para o mesmo valor de profundidade ao nível da
quantização de `float32` do anotador, resultando em Z constante.
Não é, à luz da inspeção feita, um defeito do *pipeline*: é uma
propriedade da geometria observada combinada com a precisão do
*render*.

---

## 13. Problemas encontrados e correções

Esta secção documenta os problemas técnicos efetivamente observados
durante o desenvolvimento da Fase 1 e as correções aplicadas, para
que o relatório final possa apresentar o percurso de
desenvolvimento e não apenas o método final.

1. **Inicialização do orquestrador do Replicator inexistente**.
   A primeira versão do *script* invocava
   `rep.orchestrator.initialize_async()` antes do
   `step_async(...)`. No ambiente Isaac Sim 5.1 utilizado, esse
   método não existe e a captura falhava com `AttributeError`.
   *Correção*: remoção da chamada e adoção do padrão já validado
   nos *scripts* anteriores do projeto: criar *render product*,
   anexar anotadores, executar `await
   rep.orchestrator.step_async(rt_subframes=...)` e ler com
   `get_data()`.

2. **Resolução de caminhos via `__file__` no *Script Editor***.
   Inicialmente, o diretório de saída era derivado de
   `Path(__file__).resolve().parent.parent`. Quando o *script* é
   colado/executado no *Script Editor* do Isaac Sim, `__file__`
   pode resolver para um caminho temporário do tipo
   `/tmp/carb.../script_*.py`, fazendo com que as saídas fossem
   gravadas fora do repositório.
   *Correção*: definição explícita de `PROJECT_ROOT`, com
   variável de ambiente opcional `SHAPE_INSERTION_PROJECT_ROOT`
   para sobreposição em outros ambientes (ex. máquina de
   desenvolvimento). O caminho passou a ser estável
   independentemente do contexto de execução.

3. **Múltiplas peças visíveis simultaneamente**.
   Em capturas iniciais, várias peças permaneciam visíveis na
   cena. O *pipeline* selecionava o componente ligado de maior
   área, sem garantia sobre qual peça era escolhida.
   *Decisão metodológica para esta fase*: ocultar manualmente as
   peças não pretendidas no Isaac Sim e capturar uma peça de cada
   vez. Foram acrescentados modos de seleção configuráveis
   (`largest`, `closest_to_center`, `manual_index`) para tornar a
   escolha determinística sem recorrer a classificação de forma.

4. **Compatibilidade do formato dos anotadores**.
   Em diferentes versões do Replicator, `get_data()` pode retornar
   diretamente um *ndarray* ou um dicionário com chave `"data"`.
   Nas primeiras execuções obtiveram-se *crashes* do tipo
   `TypeError` por aplicar fatiamento direto a um dicionário.
   *Correção*: normalização defensiva — se o retorno for
   dicionário, converter para *ndarray* via
   `np.asarray(d["data"]).reshape(IMG_H, IMG_W, -1)` antes de
   usar; impressão única do tipo e *shape* para diagnóstico.

5. **Saídas espúrias após falha**.
   Numa versão intermédia, o bloco `finally` produzia *placeholders*
   com matrizes de zeros para evitar erros de escrita. Isto deixava
   ficheiros com aspeto de captura válida quando, na realidade, a
   captura tinha falhado.
   *Correção*: remoção de *placeholders*; passou a gravar-se apenas
   os artefactos efetivamente produzidos pela execução atual; o
   `piece_metadata.json` é sempre escrito, com `success=False` e
   mensagem de erro quando aplicável; ficheiros de execuções
   anteriores são removidos no início para não poderem ser
   confundidos com o resultado atual.

6. **Amplitude Z nula em peças prismáticas**.
   As peças `rectangle` e `circle` apresentaram amplitude Z
   exatamente igual a zero. Verificou-se ser uma propriedade
   conjunta da geometria observada (face superior estritamente
   plana) e da quantização de `float32` do anotador de
   profundidade, e não um defeito do *pipeline*. A amplitude Z é
   informação útil mas, nesta fase, não é estritamente necessária
   para a baseline geométrica baseada em pegada.

7. **Verificação de escala real**.
   Nas primeiras execuções, a função de intrínsecos da câmara era
   alimentada com a altura nominal da câmara em vez da
   profundidade efetiva da superfície. Isto introduzia um erro
   sistemático de escala em XY proporcional à diferença entre as
   duas profundidades.
   *Correção*: passou a usar-se a profundidade da superfície
   estimada (`surface_z`) para calcular metros-por-pixel,
   garantindo coerência métrica entre a *pixel grid* e o mundo.

---

## 14. Limitações da abordagem atual

1. **Vista única e topo plano dominante**: a partir de uma só
   captura *top-down*, peças prismáticas com face superior plana
   produzem nuvens de pontos com pouca ou nenhuma variação em Z.
   A representação resultante codifica essencialmente a pegada e a
   altura, mas não a forma lateral da peça.

2. **Seleção restrita a um componente**: o pipeline assume uma
   peça visível por captura. Se múltiplas peças estiverem visíveis,
   é selecionado apenas um componente segundo o critério
   configurado, sem qualquer raciocínio sobre identidade da peça.

3. **Sensibilidade aos limiares**: a estimação da superfície e a
   segmentação dependem de constantes que devem ser sintonizadas
   para a cena (`SURFACE_DEPTH_MIN/MAX`, `SURFACE_TOLERANCE`,
   `CC_MIN_AREA_PX`, `CC_MAX_AREA_PX`).

4. **Dependência da pose da câmara**: assume-se câmara
   aproximadamente vertical sobre a superfície. Desvios
   significativos invalidam a interpretação `world_z = surface_z -
   depth` como altura sobre a mesa.

5. **Intrínsecos hardcoded**: a focal e a abertura estão como
   constantes no *script*; têm de coincidir com os atributos do
   *prim* da câmara em USD. Discrepâncias produzem erro
   sistemático de escala em XY.

6. **Cobertura geométrica parcial**: faces laterais e inferiores da
   peça não são observáveis. A nuvem de pontos é, na prática, uma
   *2.5D heightmap* da face superior visível.

---

## 15. Relevância para o objetivo da tese

O objetivo central da tese é a aprendizagem de relações
perceção-ação baseadas em geometria, com o caso de estudo da
inserção de peças em cavidades. A perceção determinística aqui
descrita é o degrau inicial: fornece os representantes geométricos
das peças que serão posteriormente confrontados com cavidades para
inferir compatibilidade, rotação de inserção e pose aproximada.

Posicionamento desta fase no plano global:

- **Não substitui** a abordagem aprendida pretendida — fornece-lhe
  os artefactos de entrada e estabelece a baseline geométrica de
  referência.
- **Não classifica** formas. A saída do *pipeline* não é uma
  etiqueta como "este é um quadrado", mas uma representação
  geométrica reutilizável (máscara, pegada, nuvem de pontos,
  metadados).
- **Preserva escala real**, condição necessária para qualquer
  raciocínio posterior sobre inserção, em que o tamanho absoluto da
  peça e da cavidade é informação carregada de significado.
- **A representação por pegada 2D é adequada para a baseline
  geométrica determinística** de correspondência peça-cavidade,
  por exemplo via IoU ou distância de Chamfer sob rotações
  candidatas.

Para representações 3D mais ricas — necessárias caso se pretenda
modelar a peça por todas as faces — está prevista uma extensão
futura com captura *multi-view*, fora do âmbito desta fase.

---

## 16. Figuras a incluir mais tarde em LaTeX

| Identificador | Caminho atual | Legenda sugerida |
|---|---|---|
| `fig:rgb` | `data/pieces_detected/<peça>/rgb.png` | Imagem RGB capturada pela câmara virtual. |
| `fig:depth_vis` | `data/pieces_detected/<peça>/depth_vis.png` | Visualização colorida da imagem de profundidade. |
| `fig:raw_mask` | `data/pieces_detected/<peça>/raw_piece_mask.png` | Máscara binária resultante do limiar sobre a profundidade. |
| `fig:piece_debug` | `data/pieces_detected/<peça>/piece_debug.png` | Sobreposição da máscara selecionada, caixa envolvente e centróide. |
| `fig:footprint` | `data/pieces_detected/<peça>/piece_footprint.png` | Pegada 2D *top-down* da peça. |
| `fig:footprints_grid` | `data/pieces_detected/footprints_grid.png` | Grelha das pegadas das quatro peças capturadas. |

Sugere-se uma figura composta de quatro painéis (RGB, profundidade,
máscara *debug* e *footprint*) por peça representativa, mais a
grelha resumo das quatro pegadas. As métricas da Tabela 12.1 e
12.2 podem entrar como tabelas.

---

## 17. Dimensões CAD nominais das peças

Esta secção regista as dimensões CAD finais do conjunto
experimental utilizado a partir desta versão. Os valores são
canónicos e estão também armazenados em
`data/expected_cad_dimensions.json`, ficheiro que serve de
referência única para auditoria de escala. **Estes valores são
para validação/relato apenas — não são consumidos pelo algoritmo
de matching.**

Conjunto principal (após substituição da estrela por triângulo,
ver doc 03 — secção 11):

| Peça        | XY nominal (mm)            | Altura/extrusão (mm) |
|-------------|----------------------------|----------------------|
| quadrado    | 50 × 50                    | 105                  |
| retângulo   | 50 × 75                    | 105                  |
| triângulo   | base 50, altura geom. 50   | 105                  |
| círculo     | diâmetro 50                | 105                  |

A dimensão de extrusão (105 mm) das peças não é usada pela
Baseline 1, que é puramente baseada em pegada XY (ver
doc 03 — secção 12). É registada aqui porque será necessária
para fases futuras de perceção 3D, *multi-view* e execução de
inserção robotizada (em particular: 105 mm de peça vs. 75 mm
de profundidade de cavidade implica protrusão de 30 mm acima do
topo do tabuleiro).

A estrela permanece como caso de *stress* concava reservado para
futuro trabalho, registada em
`data/expected_cad_dimensions.json` em
`optional_stress_test_shapes`.

---

## 18. Atualização — estado atual da *pipeline* de captura

Esta secção substitui, para fins de leitura do estado **atual**, as
secções 4–14 acima (que permanecem como registo histórico do
desenvolvimento). Documenta a *pipeline* tal como existe após o
conjunto de alterações descritas na secção 13 mais as alterações
adicionais introduzidas posteriormente para resolver problemas de
controlo de câmara, estimação de superfície e projeção métrica.

### 18.1 Conjunto de formas final

O conjunto principal de peças usado a partir desta versão é:

- `rectangle`
- `square`
- `circle`
- `triangle` (substitui a `star`)

A `star` foi removida do conjunto principal por ser excessivamente
sensível a segmentação e a escala absoluta, conforme documentado
no doc 03 — secção 11. Permanece registada em
`data/expected_cad_dimensions.json`, em
`optional_stress_test_shapes`, como caso de *stress* concava
reservado para trabalho futuro.

### 18.2 Dimensões CAD finais (recapituladas)

| Peça        | XY nominal (mm)            | Extrusão (mm) |
|-------------|----------------------------|---------------|
| rectangle   | 75 × 50                    | 105           |
| square      | 50 × 50                    | 105           |
| circle      | diâmetro 50                | 105           |
| triangle    | base 50, alt. geom. 50     | 105           |

Para o tabuleiro e cavidades, ver doc 02 — secção 18; para
folga nominal (1 mm total, 0,5 mm por lado), ver
`data/expected_cad_dimensions.json`.

### 18.3 Controlo da câmara via *stage*

`scripts/capture_piece_detection.py` adopta agora a mesma
convenção de `scripts/capture_cavity_detection.py`:

- `SET_CAMERA_POSE = False` por omissão. O *script* **não** move
  a câmara: usa a pose autorizada no *stage* USD, que o
  utilizador posiciona manualmente no Isaac Sim.
- A pose mundial efetiva da câmara é lida via
  `get_camera_world_pose()`, impressa no início da execução, e
  registada em `piece_metadata.json` em `camera_pose` com
  `source = "stage"`.
- Se `SET_CAMERA_POSE = True`, as constantes `CAM_X/Y/Z` e
  `CAM_ROT_Z_DEG` são aplicadas via `setup_camera()` e
  `source = "config_override"`. Esta via é mantida apenas para
  reprodução determinística de capturas anteriores.

A pose efetiva é depois passada como `cam_xy` para a função de
*back-projection*, garantindo que as coordenadas mundiais XY
correspondem à pose realmente em vigor (e não a constantes de
configuração que poderiam não coincidir com o *stage*).

### 18.4 Estimativa automática da superfície de suporte por camadas de profundidade

A estimativa por modo dominante de profundidade falhou em
cenários com vários planos no campo de visão (por exemplo,
peça + tabuleiro local + uma segunda mesa/parede ao fundo). A
solução adoptada é a estimação por **camadas de profundidade**:

1. Restrição da análise a uma **ROI** centrada no campo de
   captura da peça (`PIECE_ROI_ENABLED = True`,
   `PIECE_ROI_MODE = "center_fraction"`,
   `PIECE_ROI_FRACTION = 0.60`).
2. Recolha dos *pixels* de profundidade válidos (positivos,
   finitos) dentro dessa ROI.
3. Cálculo de **limites adaptativos** a partir da própria
   distribuição (`AUTO_SURFACE_DEPTH_BOUNDS = True`):
   `lower = p01 − margem`, `upper = p99 + margem` com
   `SURFACE_DEPTH_MARGIN_M = 0,005` m. As constantes estáticas
   `SURFACE_DEPTH_MIN/MAX` deixam de funcionar como filtro
   rígido neste modo (mantêm-se em uso apenas no modo legado
   `dominant_depth`).
4. Construção de histograma com *bin* de 1 mm
   (`SURFACE_HIST_BIN_M`).
5. Extracção de máximos locais e fusão de picos próximos
   (`SURFACE_PEAK_MERGE_DISTANCE_M = 0,004` m).
6. Ordenação dos picos do mais próximo ao mais distante.
7. Selecção do pico de suporte:
   - picos próximos pequenos (fração ≤
     `PIECE_MAX_PEAK_FRACTION = 0,08`) são saltados como
     prováveis "topo da peça";
   - é aceite o primeiro pico com fração ≥
     `SUPPORT_MIN_PEAK_FRACTION = 0,10`;
   - se nenhum pico atinge esse limiar, recorre-se ao pico de
     maior fração e isto é registado em
     `selected_support_reason`.
8. Salvaguarda: se a máscara crua resultante cobrir mais de 50 %
   da ROI segmentada, é tentada uma **reseleção** com o pico
   imediatamente mais próximo — registada em
   `selected_support_reason` como `"... | RESELECTED (initial
   mask >50% of ROI)"`.

Diagnósticos disponíveis na consola e em metadados:

- listagem completa dos picos detectados (profundidade, contagem,
  fracção);
- pico seleccionado, identificado por `rank` e `reason`;
- aviso explícito `[WARNING] selected support appears to be the
  farthest layer ...` quando o pico escolhido é o último (sinal
  típico de fundo a ganhar);
- aviso `[WARNING] raw piece mask covers too much of ROI ...`;
- aviso específico para a configuração actual (`> 0,68 m`)
  indicando proximidade ao pano de fundo;
- imagem `depth_layers_debug.png` com a ROI de superfície
  (ciano), ROI de segmentação expandida (amarela) e painel de
  texto com a lista de picos, com o pico seleccionado destacado
  a verde.

### 18.5 Segmentação da peça

A regra mantém-se geometricamente correcta:

```
piece_mask = (depth > DEPTH_MIN_VALID) AND (depth < surface_z − SURFACE_TOLERANCE)
```

com `SURFACE_TOLERANCE = 0,004` m. Pixels mais próximos da
câmara do que a superfície (numa observação *top-down*) são
classificados como "acima do suporte".

Se `RESTRICT_PIECE_MASK_TO_ROI = True`, a máscara crua é também
restringida à ROI **expandida** por `PIECE_MASK_ROI_EXPAND_PX =
20` *pixels*, garantindo que objectos fora do campo de captura
nunca podem entrar na análise de componentes ligados.

### 18.6 Geração de *point cloud* e *footprint* — projecção por *pixel*

O cálculo de coordenadas mundiais XY foi alterado para
**projecção dependente da profundidade por *pixel***. A
versão anterior usava `mpp` calculado em `surface_z` para todos
os *pixels*, o que era aceitável para peças finas mas inflacionou
sistematicamente as dimensões para peças altas (105 mm).

Convenção actual (canónica para o modelo *pinhole*):

```
world_x = cam_x + (u − cx_px) / fx_px × depth_px
world_y = cam_y − (v − cy_px) / fy_px × depth_px
world_z = surface_z − depth_px            (altura acima do suporte)
```

onde `fx_px` e `fy_px` são distâncias focais expressas em
*pixels* (independentes da profundidade), `cam_x`/`cam_y` são as
coordenadas mundiais reais da câmara obtidas do *stage*, e
`depth_px` é a profundidade observada de cada *pixel* (distância
ao plano de imagem). XY é depois centrado no centróide da peça;
Z é mantido em valores absolutos (altura sobre o suporte).

A escala real é preservada. Z = 0 é tipicamente reportado para
faces superiores planas, devido à quantização em `float32` do
anotador — não constitui defeito do *pipeline*.

Diagnósticos novos por captura:

- `projection_depth_mode = "per_pixel_depth"`
- `support_surface_depth_m`
- `piece_depth_median_m`
- `piece_height_median_m`
- `piece_height_min_m`
- `piece_height_max_m`
- `xy_projection_note` (fórmula literal, para citação no
  relatório).

A pegada 2D continua a ser construída pela projecção *top-down*
da nuvem de pontos numa tela de 256 × 256 *pixels* a 0,5 mm/px.

### 18.7 Problemas encontrados nesta iteração

Resumo dos problemas observados e resolvidos durante esta
iteração da *pipeline*. Mais detalhe operacional na secção 13.

1. **Câmara movida pelo *script* sem necessidade.** Decisão:
   `SET_CAMERA_POSE = False` por omissão; usar a pose do *stage*.
2. **Estimador dominante a "agarrar" o fundo.** A profundidade
   dominante numa cena com várias mesas/paredes podia ser o
   plano mais distante. Decisão: estimador por camadas
   (`auto_depth_layers`) que prefere o pico grande mais próximo
   sem ser o topo da peça.
3. **Janela `SURFACE_DEPTH_MIN/MAX` demasiado estreita.** Os
   limites estáticos excluíam camadas úteis (peça, suporte
   local) e deixavam apenas o fundo no histograma. Decisão:
   limites adaptativos por p01/p99 da distribuição da própria
   ROI quando em modo `auto_depth_layers`.
4. **Inflação sistemática de XY (~1,5×) para peças de 105 mm de
   altura.** Causada pela utilização de `mpp(surface_z)` na
   *back-projection* uniforme. Decisão: projecção por *pixel*
   com `fx_px`/`fy_px` independentes da profundidade.

Histórico anterior (Fase 1 inicial) está registado na secção 13.

### 18.8 Validação atual (após correcções)

Resultado de
`scripts/validate_piece_captures.py` sobre as quatro pastas
`data/pieces_detected/{rectangle, square, circle, triangle}/`:

- 4/4 peças passam todos os critérios estruturais;
- todos os ficheiros obrigatórios presentes;
- nuvens de pontos com forma `(2048, 3)`, sem NaN nem infinitos;
- *footprints* legíveis e não vazios;
- exactamente um componente válido por captura
  (`n_valid_components = 1`).

Métricas geométricas medidas (extraídas de
`data/pieces_detected/validation_summary.csv` e dos metadados
individuais):

| Peça        | X medido (mm) | Y medido (mm) | Z span (mm) | `piece_height_median` (mm) |
|-------------|---------------|---------------|-------------|----------------------------|
| rectangle   | 49,8          | 69,4          | 0           | 104,5                      |
| square      | 49,8          | 46,4          | 0           | 104,5                      |
| circle      | 49,4          | 46,0          | 0           | 104,5                      |
| triangle    | 49,4          | 46,0          | 0           | 104,5                      |

Comparação com o CAD:

- **Altura da peça**: medida 104,5 mm vs CAD 105 mm ⇒ ≈ 0,5 % de
  desvio. Confirma que a estimativa de superfície (≈ 0,2995 m)
  e a projecção por *pixel* estão alinhadas.
- **Dimensão X**: erro ≤ ≈ 1,2 % em todos os casos (50 mm ⇒
  49,4–49,8 mm; 75 mm ⇒ não aplicável a X aqui pois a peça
  longa está orientada com 75 mm em Y).
- **Dimensão Y**: erro sistemático de aproximadamente
  −7 a −8 % (50 mm ⇒ 46,0–46,4 mm; 75 mm ⇒ 69,4 mm). Ver
  secção 18.10 abaixo.
- Z `span` = 0 mantém-se. Aceitável para correspondência por
  pegada na Baseline 1.

### 18.9 Limitações actuais

1. **Z span = 0 nas faces superiores planas** — propriedade
   conjunta da geometria observada e da quantização *float32*
   do anotador. Sem consequência para a Baseline 1 (correspondência
   por pegada 2D); requer abordagem complementar (multi-view ou
   depth com sub-pixel) para verificação vertical de inserção.
2. **Viés sistemático de Y (~7–8 %)** — descrito em detalhe na
   secção 18.10.
3. **Restricção a uma peça visível por captura** — premissa
   experimental; o *script* deteta múltiplos componentes mas
   selecciona um, controlado por
   `PIECE_SELECTION_MODE`.
4. **Sensibilidade aos parâmetros do estimador por camadas**
   (`SUPPORT_MIN_PEAK_FRACTION`, `PIECE_MAX_PEAK_FRACTION`,
   `SURFACE_PEAK_MERGE_DISTANCE_M`). Os valores actuais
   funcionam para a cena validada; cenas novas podem exigir
   reajuste.
5. **Cobertura geométrica parcial** — observação *top-down*
   única; faces laterais e inferiores não são observáveis.

### 18.10 Viés residual de Y (intrínsecos verticais) — *RESOLVIDO*

> **Nota:** esta secção descreve o problema tal como foi
> observado e diagnosticado **antes** da correcção. Para o
> registo da correcção aplicada e dos novos resultados
> validados, ver secção **18.12** abaixo.

A função `compute_intrinsics()` calcula a *focal* vertical em
*pixels* através de:

```
fov_v          = fov_h × (IMG_H / IMG_W)
tan_half_fov_y = tan(fov_v / 2)
fy_px          = (IMG_H / 2) / tan_half_fov_y
```

Esta escala **linear em graus/radianos** entre `fov_h` e `fov_v`
só é geometricamente exacta para FOVs muito pequenos. Para o
sensor actual (FOCAL = 24 mm, APERTURE = 36 mm, FOV horizontal
≈ 73,7°), o erro acumulado é não-desprezável: produz
`fy_px ≈ 459` em vez do valor correcto para *pixels* quadrados,
`fy_px = fx_px ≈ 426,7`. O rácio 459/426,7 ≈ 1,077 explica
exactamente a redução de ≈ 7,7 % observada nas dimensões
medidas em Y.

Correcção apropriada (em conformidade com *pixels* quadrados):

```
tan_half_fov_y = tan_half_fov_x × (IMG_H / IMG_W)
fy_px          = (IMG_H / 2) / tan_half_fov_y
                  → algebricamente igual a fx_px
```

Esta correcção **não** está aplicada no código no momento desta
nota, por instrução explícita de não modificar o *script*
enquanto a validação estrutural passar. Deve ser aplicada antes
de qualquer afirmação de escala absoluta no relatório final.
Para a Baseline 1 (correspondência relativa peça-cavidade), o
viés cancela parcialmente caso o *script* das cavidades use a
mesma fórmula — ver doc 02 — secção 18.

### 18.11 Próximas acções

Estado actualizado da sequência originalmente recomendada:

1. ~~Inspeccionar visualmente
   `data/pieces_detected/footprints_grid.png`.~~ — **feito**.
2. ~~Corrigir o cálculo de `fy_px` em
   `compute_intrinsics()`.~~ — **feito** (ver 18.12).
3. ~~Recapturar as quatro peças e re-validar com
   `scripts/validate_piece_captures.py`.~~ — **feito** (ver 18.12).
4. ~~Verificar se `scripts/capture_cavity_detection.py` partilha
   a mesma fórmula incorrecta.~~ — **feito**: partilhava; já
   foi corrigida pela mesma alteração (ver doc 02 — secção 19).
   **Pendente**: recapturar as cavidades com a fórmula corrigida
   e re-validar.
5. **Auditoria de escala** das amplitudes XY e altura medidas
   das cavidades contra `data/expected_cad_dimensions.json`
   (depois da recaptura).
6. **Re-executar Baseline 1** com o conjunto `triangle`
   (rectangle, square, circle, triangle), após o ponto 5.
7. **Documentar** os resultados actualizados no doc 03.

Esta sequência é tratada como pré-condição. Os resultados
actuais da Baseline 1 (com o conjunto que incluía a `star`)
permanecem registados como diagnóstico intermédio mas **não
constituem o resultado final**.

### 18.12 Correção dos intrínsecos e validação de escala

**Problema encontrado.** Após substituir a `star` por
`triangle`, recalibrar as dimensões CAD e revalidar
estruturalmente as quatro peças, observou-se que as nuvens de
pontos eram dimensionalmente inconsistentes com o CAD. As
dimensões em X aproximavam-se do esperado, mas as dimensões em
Y estavam **sistematicamente subestimadas em 7–8 %**.

**Evidência.** Medições efectuadas em
`data/pieces_detected/validation_summary.csv` antes da
correcção:

| Peça        | Y medido (mm) | Y CAD (mm) | erro    |
|-------------|---------------|------------|---------|
| square      | 46,4          | 50         | −7,2 %  |
| circle      | 46,0          | 50         | −8,0 %  |
| triangle    | 46,0          | 50         | −8,0 %  |
| rectangle   | 69,4          | 75         | −7,5 %  |

Em paralelo:
- a altura mediana (`piece_height_median`) estava correcta
  (104,5 mm vs CAD 105 mm), o que descartava erro na escala
  global de profundidade ou na estimativa do plano de suporte;
- a dimensão X estava correcta (≤ 1,2 % de erro), o que
  localizava o problema **exclusivamente na direcção vertical
  da imagem**.

**Causa.** A função `compute_intrinsics()` em
`scripts/capture_piece_detection.py` (e a função homónima em
`scripts/capture_cavity_detection.py`) calculava o FOV vertical
através de uma escala linear em radianos:

```
fov_v = fov_h × (IMG_H / IMG_W)
```

Esta aproximação é apenas válida para FOVs muito pequenos. Para
o sensor utilizado (FOCAL = 24 mm, APERTURE = 36 mm,
FOV horizontal ≈ 73,7°), produzia `fy_px ≈ 459` em vez do valor
geometricamente correcto para *pixels* quadrados,
`fy_px = fx_px ≈ 426,67`. O rácio 459 / 426,67 ≈ 1,0736
corresponde exactamente à redução de ≈ 7,4 % observada nas
medidas em Y.

**Correcção aplicada.** A escala linear foi substituída por
uma relação tangente-aspecto:

```
tan_half_fov_y = tan_half_fov_x × (IMG_H / IMG_W)
fy_px          = (IMG_H / 2) / tan_half_fov_y
                  → algebricamente igual a fx_px
```

Esta é a fórmula consistente com *pixels* quadrados, em que a
abertura vertical efectiva é `aperture_horizontal × (H/W)` e
a *focal* vertical em *pixels* iguala a horizontal. A mesma
correcção foi aplicada em ambos os *scripts* de captura. A
projecção XY da nuvem de pontos continua a ser por *pixel*
(secção 18.6); só os intrínsecos foram corrigidos.

A *pipeline* passou a expor:
- `intrinsics_model = "pinhole_tangent_aspect_corrected"` em
  `piece_metadata.json` e em `cavities_summary.json`;
- `fx_px` e `fy_px` em ambos os ficheiros de metadados;
- linha de consola por captura:
  `[intrinsics] fx_px=..., fy_px=..., mpp_x=..., mpp_y=...`.

**Comparação antes/depois.**

| Peça | Y antes (mm) | Y depois (mm) | Y CAD (mm) | erro depois |
|------|--------------|---------------|------------|-------------|
| square    | 46,4 | **49,8** | 50 | −0,4 %  |
| circle    | 46,0 | **49,4** | 50 | −1,2 %  |
| triangle  | 46,0 | **49,4** | 50 | −1,2 %  |
| rectangle | 69,4 | **74,5** | 75 | −0,7 %  |

`fx_px = fy_px = 426,6667` confirmado nos quatro
`piece_metadata.json`; simetria X/Y restaurada (square e circle
medem agora `49,8 × 49,8` mm e `49,4 × 49,4` mm
respectivamente).

**Resultado da validação após a correcção.**

`scripts/validate_piece_captures.py`: 4/4 peças passam todos os
critérios estruturais. Métricas-chave:

| Peça      | X (mm) | Y (mm) | Z span (mm) | `piece_height_median` (mm) |
|-----------|--------|--------|-------------|----------------------------|
| rectangle | 49,8   | 74,5   | 0           | 104,5                      |
| square    | 49,8   | 49,8   | 0           | 104,5                      |
| circle    | 49,4   | 49,4   | 0           | 104,5                      |
| triangle  | 49,4   | 49,4   | 0           | 104,5                      |

Todas as dimensões dentro de ≈ 1,2 % do CAD; altura mediana a
104,5 mm vs 105 mm (≈ 0,5 %).

**Interpretação.**

- A correcção fechou o viés sistemático de Y. As peças `square`
  e `circle` recuperaram simetria X/Y exacta no plano da
  pegada, e a `rectangle` apresenta agora a razão de aspecto
  esperada (74,5 / 49,8 ≈ 1,50, vs CAD 75 / 50 = 1,50).
- O problema **não** era exclusivamente de segmentação. A
  segmentação por camadas de profundidade estava correcta; o
  defeito era estritamente na geometria projectiva. Esta
  distinção é importante para o relatório, porque mostra que
  os dois subsistemas (segmentação e projecção) podem falhar
  independentemente e exigem auditorias separadas.
- O `piece_height_median = 104,5 mm` continuar correcto antes e
  depois do *fix* confirma também que a estimativa de superfície
  por `auto_depth_layers` já estava correcta — o viés de Y não
  afectava Z porque Z é calculado por subtracção directa de
  profundidade e não passa pela escala em *pixels*.

**Limitação remanescente.**

- Erro residual de até ≈ 1,2 % em XY, atribuível a quantização
  *pixel-a-pixel* nos limites da máscara e a anti-*aliasing* no
  rasterizador de profundidade do Replicator. Este erro é
  simétrico em X e Y, situado dentro da tolerância de
  engenharia, e sem correlação com forma da peça. Nenhuma
  acção adicional é proposta para esta limitação no âmbito da
  Baseline 1.
- O Z `span` permanece igual a zero por captura; é uma
  propriedade conjunta da geometria observada (face superior
  estritamente plana) e da quantização *float32* do anotador.
  Aceitável para a Baseline 1 (correspondência por pegada);
  requer abordagem complementar para verificação vertical de
  inserção.
- A correcção foi aplicada também a
  `scripts/capture_cavity_detection.py` mas as cavidades ainda
  precisam de ser **recapturadas** para que as nuvens de pontos
  guardadas reflictam a nova fórmula (os ficheiros `.npy`
  existentes continuam baseados nos intrínsecos antigos). Ver
  doc 03 — secção 16 para o protocolo de re-execução.

---

## Notas para o autor

Itens que devem ser registados manualmente, fora deste documento, e
que não estão capturados nos ficheiros de validação:

- **Verificação independente de uma peça**: medir fisicamente
  uma peça (preferencialmente o retângulo) e confirmar que as
  dimensões CAD da secção 17 estão corretas — esta verificação
  ancora a auditoria de escala que confronta as dimensões CAD
  com as amplitudes medidas em
  `data/pieces_detected/validation_summary.csv`.
- **Confirmação visual de `footprints_grid.png`** após a
  correcção de `fy_px` (ver 18.10–18.11).
- **Pose USD da câmara no momento da captura validada** (lida do
  campo `camera_pose` em `piece_metadata.json`, mas convém
  guardar uma captura de ecrã do inspector do *stage* para o
  relatório).
- **Pose física da câmara virtual no USD** (translação e
  orientação) no momento da captura validada.
- **Versão exata do Isaac Sim** e do contentor usado.
- **Eventuais alterações da iluminação** entre capturas, se
  relevantes.
- **Justificação do conjunto inicial de peças** (rectângulo,
  quadrado, círculo, estrela) — porquê estas e não outras.
