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

## 13. Limitações da abordagem atual

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

## 14. Relevância para o objetivo da tese

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

## 15. Figuras a incluir mais tarde em LaTeX

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

## Notas para o autor

Itens que devem ser registados manualmente, fora deste documento, e
que não estão capturados nos ficheiros de validação:

- **Dimensões CAD reais das peças** (em Fusion), para confirmar a
  escala medida na captura. Verificar pelo menos uma peça
  independentemente.
- **Pose física da câmara virtual no USD** (translação e
  orientação) no momento da captura validada.
- **Versão exata do Isaac Sim** e do contentor usado.
- **Eventuais alterações da iluminação** entre capturas, se
  relevantes.
- **Justificação do conjunto inicial de peças** (rectângulo,
  quadrado, círculo, estrela) — porquê estas e não outras.
