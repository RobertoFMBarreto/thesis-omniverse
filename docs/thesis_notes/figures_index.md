# Índice de figuras para o relatório de tese

> Documento de organização para futura conversão em LaTeX.
> Não copia, renomeia ou gera ficheiros — apenas regista quais imagens
> devem ser consideradas para inclusão e com que legenda.
> Estado: inicializado com a Fase 1 (deteção de peça em vista única).
> Data: 2026-05-01.

---

## Convenções

- **Figure ID**: identificador interno usado neste documento e
  reutilizável como `\label{...}` em LaTeX.
- **Source file**: caminho relativo à raiz do repositório.
- **Suggested LaTeX filename**: nome proposto para o ficheiro
  copiado/renomeado na pasta `figures/` do projeto LaTeX (a criar
  futuramente). Convenção: `figXX_<peca>_<tipo>.png`, sempre em
  minúsculas, sem espaços nem acentos.
- **Suggested caption**: legenda em português de Portugal,
  diretamente reutilizável em `\caption{...}`.
- **Related section**: referência à secção correspondente em
  `docs/thesis_notes/01_piece_detection_singleview.md` ou ao
  capítulo da tese.
- **Notes**: observações sobre qualidade, propósito ou avisos
  relevantes para a inclusão.

Os ficheiros de validação (`validation_summary.csv` e
`validation_summary.json`) não são figuras, mas estão registados
como **fontes de dados** para tabelas e métricas a apresentar no
relatório.

---

## Tabela de figuras — Fase 1: deteção de peça em vista única

| Figure ID | Source file | Suggested LaTeX filename | Suggested caption (pt-PT) | Related section | Notes |
|---|---|---|---|---|---|
| `fig:footprints_grid` | `data/pieces_detected/footprints_grid.png` | `fig01_footprints_grid.png` | Pegadas 2D *top-down* das quatro peças capturadas em vista única (rectângulo, quadrado, círculo, estrela), projetadas a partir da nuvem de pontos com escala real preservada. | Fase 1 — secção 12 e 15 | Resumo visual; candidata a figura de abertura da secção de validação. |
| `fig:rectangle_debug` | `data/pieces_detected/rectangle/piece_debug.png` | `fig02_rectangle_debug.png` | Sobreposição da máscara do componente selecionado, caixa envolvente e centróide para a peça rectangular. | Fase 1 — secção 7 | Ilustra o resultado da segmentação e seleção do componente ligado. |
| `fig:rectangle_footprint` | `data/pieces_detected/rectangle/piece_footprint.png` | `fig03_rectangle_footprint.png` | Pegada 2D *top-down* da peça rectangular, em escala real (0,5 mm/*pixel*). | Fase 1 — secção 9 | Útil para discutir a representação geométrica que alimentará a baseline determinística. |
| `fig:square_debug` | `data/pieces_detected/square/piece_debug.png` | `fig04_square_debug.png` | Sobreposição da máscara do componente selecionado, caixa envolvente e centróide para a peça quadrada. | Fase 1 — secção 7 | Comparável a `fig:rectangle_debug` para discutir consistência do *pipeline* entre peças. |
| `fig:square_footprint` | `data/pieces_detected/square/piece_footprint.png` | `fig05_square_footprint.png` | Pegada 2D *top-down* da peça quadrada, em escala real. | Fase 1 — secção 9 | Permite verificar visualmente a simetria X≈Y reportada nas métricas de validação. |
| `fig:circle_debug` | `data/pieces_detected/circle/piece_debug.png` | `fig06_circle_debug.png` | Sobreposição da máscara do componente selecionado, caixa envolvente e centróide para a peça circular. | Fase 1 — secção 7 | Útil para discutir o comportamento do *pipeline* com fronteiras curvas. |
| `fig:circle_footprint` | `data/pieces_detected/circle/piece_footprint.png` | `fig07_circle_footprint.png` | Pegada 2D *top-down* da peça circular, em escala real. | Fase 1 — secção 9 | Inspecionar quanto à discretização do contorno na resolução escolhida. |
| `fig:star_debug` | `data/pieces_detected/star/piece_debug.png` | `fig08_star_debug.png` | Sobreposição da máscara do componente selecionado, caixa envolvente e centróide para a peça em forma de estrela. | Fase 1 — secção 7 | Caso geometricamente mais exigente; útil para discutir vértices côncavos. |
| `fig:star_footprint` | `data/pieces_detected/star/piece_footprint.png` | `fig09_star_footprint.png` | Pegada 2D *top-down* da peça em forma de estrela, em escala real. | Fase 1 — secção 9 | Caso de teste mais informativo para a futura comparação por IoU/Chamfer entre rotações candidatas. |

---

## Tabela de figuras — Fase 2: deteção do tabuleiro e das cavidades

| Figure ID | Source file | Suggested LaTeX filename | Suggested caption (pt-PT) | Related section | Notes |
|---|---|---|---|---|---|
| `fig:cavity_rgb` | `data/cavities_detected/rgb.png` | `fig10_cavity_rgb.png` | Imagem RGB da cena com o tabuleiro e as cavidades, captada pela câmara virtual. | Fase 2 — secção 4 | Fornece a referência visual da cena antes de qualquer processamento. |
| `fig:cavity_depth_vis` | `data/cavities_detected/depth_vis.png` | `fig11_cavity_depth_vis.png` | Visualização colorida da imagem de profundidade da mesma cena. | Fase 2 — secção 4 | Permite comentar a ordenação de profundidade entre fundo, tabuleiro e cavidades. |
| `fig:board_mask` | `data/cavities_detected/board_mask.png` | `fig12_board_mask.png` | Máscara binária da face superior do tabuleiro, com buracos correspondentes às cavidades. | Fase 2 — secção 6 | Mostra que o tabuleiro detetado tem holes geometricamente coerentes com as cavidades. |
| `fig:board_region_mask` | `data/cavities_detected/board_region_mask.png` | `fig13_board_region_mask.png` | Tabuleiro preenchido (modo `contour`), domínio de pesquisa para a deteção de cavidades. | Fase 2 — secção 6 | Par com `fig:board_mask`: ilustra a diferença entre superfície detetada e domínio de pesquisa preenchido. |
| `fig:board_debug` | `data/cavities_detected/board_debug.png` | `fig14_board_debug.png` | Sobreposição RGB com o tabuleiro detetado tingido, contorno preenchido, caixa envolvente e centróide. | Fase 2 — secção 6 | Figura síntese da deteção automática do tabuleiro. |
| `fig:board_roi_auto_debug` | `data/cavities_detected/board_roi_auto_debug.png` | `fig15_board_roi_auto_debug.png` | Diagnóstico do processo de deteção automática do tabuleiro: candidatos de profundidade e parâmetros utilizados. | Fase 2 — secção 6 e 14 | Útil para a discussão dos problemas encontrados e parâmetros sintonizados. |
| `fig:raw_cavity_mask` | `data/cavities_detected/raw_cavity_mask.png` | `fig16_raw_cavity_mask.png` | Máscara binária após aplicação do critério de profundidade restrito ao domínio do tabuleiro. | Fase 2 — secção 8 | Estado pré-componentes-ligados; útil para discutir limpeza morfológica. |
| `fig:cavities_debug` | `data/cavities_detected/cavities_debug.png` | `fig17_cavities_debug.png` | Sobreposição RGB com cada cavidade detetada tingida e numerada (cavity_00 a cavity_03). | Fase 2 — secção 9 | Figura de síntese da deteção das cavidades com identificadores espaciais. |
| `fig:cavities_footprints_grid` | `data/cavities_detected/footprints_grid.png` | `fig18_cavities_footprints_grid.png` | Pegadas 2D *top-down* das cavidades detetadas, em grelha rotulada por identificador. | Fase 2 — secções 12 e 13 | Análogo da figura de Fase 1 para as cavidades. Útil para comparação visual posterior peça vs. cavidade. |

---

## Fontes de dados (não-figuras) para tabelas e métricas

| ID interno | Source file | Utilização sugerida | Notes |
|---|---|---|---|
| `data:validation_csv_pieces` | `data/pieces_detected/validation_summary.csv` | Origem da tabela de amplitudes e contagem de pontos das peças. | Formato plano, fácil de transformar em `\begin{tabular}`. |
| `data:validation_json_pieces` | `data/pieces_detected/validation_summary.json` | Origem detalhada (limites X/Y/Z exatos, *flags* de validação) para tabelas auxiliares ou texto. | Mais completo do que o CSV; preferir como fonte canónica para Fase 1. |
| `data:validation_csv_cavities` | `data/cavities_detected/validation_summary.csv` | Origem da tabela de áreas e amplitudes das cavidades. | Formato plano, fácil de transformar em `\begin{tabular}`. |
| `data:validation_json_cavities` | `data/cavities_detected/validation_summary.json` | Origem detalhada da validação das cavidades (limites X/Y/Z, *flags*). | Mais completo do que o CSV; preferir como fonte canónica para Fase 2. |
| `data:cavities_summary_json` | `data/cavities_detected/cavities_summary.json` | Origem dos parâmetros do *pipeline* (tabuleiro, profundidade da mesa, *flags* de deteção, lista de componentes rejeitados). | Útil para a secção de problemas encontrados/parâmetros sintonizados. |
| `data:cavities_run_log` | `data/cavities_detected/run_log.txt` | Registo da consola da execução validada (sobrescrito em cada execução). | Útil para citação literal no relatório, com o cuidado de ficar gravado fora do *log* atual antes de o sobrescrever. |

---

## Lacunas e figuras a considerar mais tarde

Itens **não** disponíveis ainda mas potencialmente relevantes para o
relatório, a registar quando forem produzidos:

- Figura comparativa **antes/depois da segmentação** (RGB original
  ao lado da máscara selecionada) por peça.
- Figura ilustrando a estimação da superfície de suporte
  (histograma da profundidade com pico anotado), tanto para a
  Fase 1 (mesa de peças) como para a Fase 2 (mesa/fundo e topo do
  tabuleiro).
- Figuras *per-cavity* dedicadas (`cavity_NN/cavity_debug.png` e
  `cavity_NN/cavity_footprint.png`), a registar quando se decidir
  destacar uma cavidade individual no relatório.
- Figura comparativa peça vs. cavidade correspondente, a ser
  usada na secção da baseline determinística (Fase 3, ainda não
  implementada).
- Diagrama do *pipeline* de perceção (a desenhar à parte, por
  exemplo em TikZ ou em ferramenta vetorial), para ser referenciado
  como `fig:pipeline_overview`.

---

## Convenções para futura migração para LaTeX

Quando este índice for materializado no projeto LaTeX:

1. Criar pasta `figures/` no projeto LaTeX.
2. Copiar cada `Source file` para `figures/<Suggested LaTeX filename>`.
   Não alterar a imagem; apenas renomear.
3. Inserir cada figura com:

   ```latex
   \begin{figure}[ht]
     \centering
     \includegraphics[width=0.7\linewidth]{figures/figXX_<peca>_<tipo>.png}
     \caption{<legenda em pt-PT>}
     \label{<Figure ID>}
   \end{figure}
   ```

4. Manter o `Figure ID` deste documento como `\label`, para que
   referências cruzadas no texto da tese sejam estáveis mesmo que
   os nomes dos ficheiros de imagem sejam reorganizados.
5. Não embeber as imagens com escala real arbitrária — preferir
   `width=0.7\linewidth` ou `width=0.45\linewidth` para grelhas de
   duas figuras lado-a-lado.
