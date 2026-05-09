# 06 — Baseline 2 Fase B: resultados da agregação determinística multi-vista

> Nota de implementação para futura conversão em secção LaTeX.
> Estado: resultado experimental — registo de evidência negativa.
> Data: 2026-05-09.

---

## 1. Objetivo

A Fase B da Baseline 2 testou se a agregação determinística
multi-vista de scores, sobre os artefactos por vista produzidos
pela Fase A (doc 04 — secções 1 e 2) e descritos pela proposta de
desenho (doc 05), podia melhorar a discriminação geométrica
piece-cavity face à Baseline 1 (doc 03 — secção 17). A questão
experimental foi formulada de forma estrita: dadas três vistas por
peça e por cavidade, e mantendo inalterado o cabeçote de scoring
da Baseline 1, uma agregação ao nível do score sobre máscaras
rasterizadas por vista produz margens rank-1 vs rank-2 mais largas
ou pelo menos preservadas?

A Fase B **não** introduziu fusão geométrica, **não** introduziu
reconstrução 3D, **não** introduziu estimação de pose, **não**
introduziu nenhum componente aprendido. O escopo é estritamente
agregação ao nível do score sobre máscaras rasterizadas por vista,
no espírito do doc 05 — secção 5.1.

---

## 2. Método

A Fase B reaproveita os artefactos por vista da Fase A
(`data/multiview_captures/pieces/<peça>/view_NN_<nome>/`) e
introduz, do lado das cavidades, a captura multi-vista equivalente
descrita em 3.2. As três vistas são `top_down`, `front_oblique` e
`side_oblique`, herdadas sem alteração da Fase A (doc 04 —
secção 1.4).

A agregação é uma média ponderada fixa, conforme doc 05 —
secção 5.1, com pesos:

```
VIEW_WEIGHTS = {top_down: 0.6, front_oblique: 0.2, side_oblique: 0.2}
```

A justificação dos pesos é geométrica: a vista `top_down` é a vista
mais próxima do regime validado pela Baseline 1 (doc 03 —
secção 17) e por isso recebe o peso dominante; as duas vistas
oblíquas contribuem em partes iguais, sem privilegiar nenhum eixo
horizontal. Os pesos foram fixados a priori e **não** foram
otimizados sobre os resultados.

Os descritores foram mantidos ao mínimo: por vista, o cabeçote de
scoring da Baseline 1 é aplicado tal como definido (doc 03 —
secção 6), produzindo `IoU`, `inside_ratio` e `outside_ratio` para
a rotação ótima encontrada pela pesquisa em grelha de 2°. Não
foram adicionados descritores novos da família discutida em
doc 05 — secção 4 (nem aspect ratio, nem circularidade, nem
momentos de Hu, nem `cv2.matchShapes`). A justificação é de
higiene experimental: a Fase B avalia exclusivamente o efeito da
representação multi-vista, com o cabeçote de scoring congelado.
Adicionar simultaneamente descritores novos confundiria duas
fontes de variação.

A constante de margem mínima `TIE_MARGIN = 0.01` é herdada da
Baseline 1 (doc 03 — secção 7), e o limiar `MIN_VIEW_POINTS = 50`
foi introduzido para descartar vistas cujo splat por profundidade
seja insuficiente para suportar uma máscara rasterizada
significativa.

A comparação experimental é direta contra os números finais da
Baseline 1 reportados em doc 03 — secção 17.4.

---

## 3. Iterações experimentais

A Fase B passou por três iterações distintas. Cada iteração
revelou um problema estrutural na representação que motivou a
seguinte. Esta secção regista a sequência cronológica como
diagnóstico, não como justificação retrospetiva.

### 3.1 Iteração A — primeira execução assimétrica

A primeira execução utilizou, do lado das peças, as três vistas da
Fase A; do lado das cavidades, manteve a representação top-down
única validada pela Baseline 1
(`data/cavities_detected/<cavity>/cavity_pointcloud.npy`).

Foi observado que as 4/4 atribuições rank-1 da Baseline 1 foram
preservadas, mas as margens rank-1 vs rank-2 estreitaram-se
consistentemente face à Baseline 1. O intervalo de IoU por par
manteve-se válido (0.339 a 0.980), mas verificou-se que o score
agregado era dominado pela componente `top_down` (peso 0.6), que
era a única componente a transportar sinal discriminativo
coerente; as componentes oblíquas adicionavam essencialmente ruído
direcional.

Constatou-se que a comparação era estruturalmente inválida: vistas
oblíquas das peças foram comparadas contra máscaras top-down das
cavidades, o que produz desfasamento geométrico sistemático. Não
pode ser concluído que a agregação multi-vista é prejudicial a
partir desta iteração; o que pode ser concluído é que a comparação
exige representação simétrica em ambos os lados.

### 3.2 Iteração B — captura simétrica de cavidades

Foi introduzido o script `scripts/capture_multiview_cavities.py`
para capturar as cavidades nas mesmas três vistas usadas pelas
peças. O controlo de visibilidade foi reaproveitado da Fase A
(doc 04 — secção 2.3): durante a captura de cavidades, todas as
quatro peças MVP foram escondidas para garantir que a única
geometria observável acima do plano do tabuleiro era o tabuleiro
em si, e a única geometria observável abaixo eram os interiores
das cavidades.

A iteração B subdividiu-se em três sub-iterações, cada uma a
tentar resolver um modo de falha distinto na construção das
máscaras de cavidade.

**B.1 — filtro Z inicial.** O primeiro filtro foi
`world_z < board_top - 0.001`, isto é, todos os pontos cuja
profundidade ficasse pelo menos 1 mm abaixo do topo do tabuleiro
seriam considerados pertencentes ao interior de uma cavidade.
Constatou-se que o filtro capturava toda a mesa e parte do chão
dentro do FOV, porque ambos estão geometricamente abaixo do
tabuleiro. As máscaras resultantes continham entre 200 000 e
290 000 pontos por cavidade, com bounding box que cobria toda a
ROI de captura. O fallback de convex-hull, herdado da Baseline 1
(doc 03 — secção 4), envolvia este splat amplo num quadrado quase
uniforme de aproximadamente 98 000 px por cavidade, indistinguível
entre as quatro cavidades. O resultado foi que todas as quatro
peças colapsaram para `cavity_00` por desempate por índice.

**B.2 — filtro XY ROI adicional.** Foi adicionado um filtro XY de
ROI quadrada de ±55 mm em torno do centro conhecido de cada
cavidade (`CAVITY_VIEW_ROI_HALF_SIZE_M = 0.055`), o que eliminou a
fuga para a mesa e para o chão. A contagem de pontos por cavidade
caiu para o intervalo 1 800 a 4 300, mas verificou-se que as
máscaras top-down resultantes continuavam a apresentar área
uniforme de aproximadamente 98 000 px. A causa é estrutural: o
splat por profundidade permanece esparso ao longo de toda a ROI,
e o convex-hull fallback envolve o splat esparso como um quadrado
quase uniforme. A sub-iteração corrigiu a fuga geométrica mas não
corrigiu a uniformidade da máscara final.

**B.3 — banda Z fina.** Foi imposta uma banda Z estreita
(`board_top − 5 mm < z < board_top − 1 mm`, isto é,
`CAVITY_DEPTH_MIN_BELOW_SURFACE_M = 0.001` e
`CAVITY_DEPTH_MAX_BELOW_SURFACE_M = 0.005`) com o objetivo de
isolar exclusivamente o aro da cavidade. As máscaras oblíquas
tornaram-se efetivamente distintas por cavidade. Foi observado
porém que as quatro vistas top-down ficaram MISSING: a banda é
demasiado estreita para captar amostras suficientes vindas
exatamente de cima, dado que a transição de profundidade no aro é
quase abrupta e muito poucos pixels caem precisamente dentro da
banda de 4 mm quando observados ao longo da normal da superfície.

A conclusão da iteração B é que a representação multi-vista de
cavidades é instável sob descritores baseados em máscara
rasterizada simples: sem ROI a representação tem fuga
geométrica; com Z-banda larga colapsa via convex-hull; com Z-banda
fina perde a vista top-down.

### 3.3 Iteração C — representação híbrida

Foi adoptada uma política híbrida controlada por uma constante por
vista, `CAVITY_VIEW_SOURCE`. Para `top_down`, o source é o
pointcloud validado da Baseline 1
(`cavity_opening_pointcloud.npy`); para as duas vistas oblíquas, o
source é a captura multi-vista da iteração B.3 (ROI XY mais banda
Z fina). Cada registo por vista persiste o campo `cavity_source`
∈ {`baseline1_validated_opening`, `multiview_roi_z_band`}, de
forma a tornar a origem geométrica auditável a partir dos
artefactos.

Sob esta política, a vista top-down recuperou áreas distintas por
cavidade — `cavity_00 = 60 277 px`, `cavity_01 = 20 091 px`,
`cavity_02 = 40 194 px`, `cavity_03 = 31 679 px` — coincidentes,
por construção, com as áreas usadas pela Baseline 1.

Os resultados rank-1 por peça foram:

- `rectangle → cavity_00`, score `0.493`, margem `0.107`
  (CORRETO);
- `square → cavity_02`, score `0.578`, margem `0.091` (CORRETO);
- `circle → cavity_03`, score `0.592`, margem `0.058` (CORRETO);
- `triangle → cavity_03`, score `0.500`, margem `0.006`
  (INCORRETO; ground truth `cavity_01`; sinalizador
  `low_margin = True`).

O sinalizador `per_view_disagreement` permaneceu em 4/4: para
todas as quatro peças, a vista top-down e as duas vistas oblíquas
escolhem cavidades diferentes em pelo menos uma das comparações.

Foi observado, no caso do triangle, que a vista top-down prefere
corretamente `cavity_01` (score por vista `0.863` contra `0.658`
para `cavity_03`), mas as máscaras oblíquas de `cavity_01` são
diminutas (319 px e 159 px nas duas oblíquas respetivamente). O
footprint do triangle excede estas máscaras, produzindo scores por
vista oblíqua negativos (`−0.057` e `−0.063`). As máscaras
oblíquas de `cavity_03` são consideravelmente maiores (15 381 px e
4 571 px), produzindo scores por vista oblíqua positivos
(`+0.409` e `+0.115`). A média ponderada
(`0.6 * 0.863 + 0.2 * (−0.057) + 0.2 * (−0.063)`) versus
(`0.6 * 0.658 + 0.2 * 0.409 + 0.2 * 0.115`) faz `cavity_03`
ultrapassar `cavity_01` por uma margem de `0.006` — abaixo do
limiar `TIE_MARGIN = 0.01`, donde o sinalizador `low_margin`.

---

## 4. Resultados

| peça | B1 melhor | B1 margem | B2 híbrido melhor | B2 híbrido margem | interpretação |
|---|---|---:|---|---:|---|
| rectangle | cavity_00 | 0.293 | cavity_00 | 0.107 | preservado, margem reduzida (drag oblíquo) |
| square    | cavity_02 | 0.168 | cavity_02 | 0.091 | preservado, margem reduzida |
| circle    | cavity_03 | 0.114 | cavity_03 | 0.058 | preservado, margem reduzida |
| triangle  | cavity_01 | 0.227 | cavity_03 | 0.006 | trocado (incorreto), margem em empate (`low_margin`) |

---

## 5. Discussão crítica

A Fase B da Baseline 2 **não** melhorou face à Baseline 1. Em três
das quatro peças a atribuição rank-1 foi preservada mas a margem
rank-1 vs rank-2 reduziu-se de forma sistemática; na quarta peça
(`triangle`), a atribuição rank-1 mudou de correta para incorreta,
com margem residual abaixo do limiar de empate. Verifica-se que o
efeito agregado das vistas oblíquas, na representação atual, é
introduzir desacordo sistemático em vez de reforço.

Deve ser sublinhado que isto **não** constitui evidência contra
perceção multi-vista em geral. O que estes resultados mostram é
mais restrito: a agregação ao nível do score, sobre máscaras
rasterizadas oblíquas que são por natureza parciais e dependentes
da vista, é insuficiente na representação atual. Em particular, a
representação atual não recupera a área de cavidade que seria
observável a partir de cada vista oblíqua de forma estável — o
diagnóstico das sub-iterações B.1, B.2 e B.3 demonstra que o
parâmetro Z-banda admite ou fuga geométrica ou colapso de máscara,
sem regime intermédio satisfatório para o conjunto MVP atual.

A Baseline 1 permanece, portanto, a baseline determinística de
referência mais forte para o regime piece-cavity sobre o conjunto
MVP convexo. A Fase B é registada como evidência negativa
controlada: o resultado é informativo precisamente porque a
representação foi mantida ao mínimo e os pesos foram fixados a
priori, evitando a tentação de afinar os hiperparâmetros até obter
um número favorável.

---

## 6. Limitações

1. **Captura sequencial.** A captura é feita por uma única câmara
   reposicionada programaticamente entre vistas (doc 04 —
   secções 1.4 e 2.6, ponto 7); não é um rig multi-câmara
   síncrono. Resultados específicos a este regime.
2. **Sem fusão.** As três vistas são agregadas exclusivamente ao
   nível do score escalar; não há fusão geométrica de pontos no
   referencial do mundo, no sentido proposto em doc 04 —
   secção 2.7, primeiro ponto.
3. **Sem reconstrução 3D.** Não foi produzida representação
   volumétrica nem TSDF; apenas máscaras 2D rasterizadas por vista.
4. **Sem estimação de pose.** A Fase B reporta apenas `(cavity,
   rotação no plano, score agregado)`; não há estimação de pose 6D
   nem de pose de inserção.
5. **Máscaras oblíquas parciais e dependentes da vista.** As
   máscaras oblíquas das cavidades, na representação da
   iteração C, refletem apenas a porção do aro observável a partir
   da pose oblíqua específica; não são representações canónicas da
   cavidade. Esta dependência é explicitamente observável nas
   contagens de pixels desproporcionais entre cavidades.
6. **Dependência das poses de câmara e dos limiares.** Os
   resultados são específicos a `TOP_DOWN_HEIGHT = 0.50 m`,
   `OBLIQUE_HEIGHT = 0.40 m`, `OBLIQUE_OFFSET = 0.30 m` (doc 04 —
   secção 1.4) e aos limiares
   `CAVITY_VIEW_ROI_HALF_SIZE_M = 0.055`,
   `CAVITY_DEPTH_MIN_BELOW_SURFACE_M = 0.001`,
   `CAVITY_DEPTH_MAX_BELOW_SURFACE_M = 0.005`,
   `MIN_VIEW_POINTS = 50` e `TIE_MARGIN = 0.01`. Não pode ser
   concluído que a observação se transporta para outras poses ou
   outros limiares sem nova validação.

---

## 7. Decisões tomadas

Foram tomadas, e ficam registadas, as seguintes decisões:

- **Não afinar descritores nem pesos.** Os pesos
  `top_down = 0.6, front_oblique = 0.2, side_oblique = 0.2` foram
  fixados a priori a partir da justificação geométrica de doc 05 —
  secção 5.1 e não foram alterados a posteriori para procurar um
  resultado mais favorável. Pelos mesmos motivos não foram
  introduzidos descritores adicionais (aspect ratio,
  circularidade, momentos de Hu, `cv2.matchShapes`).
- **Não adicionar componente aprendida.** A Fase B foi mantida
  estritamente determinística e geométrica, em conformidade com a
  filosofia de baseline do projeto (doc 03 — secção 1) e com os
  não-objetivos de doc 05 — secção 2.2.
- **Não migrar para fusão dentro da Fase B.** Fusão geométrica
  verdadeira de pontos no referencial do mundo é uma mudança de
  representação, não uma mudança de agregador, e foi
  deliberadamente excluída do escopo da Fase B.
- **Preservar como resultado negativo controlado.** O resultado é
  documentado em vez de descartado, porque é informativo: mostra
  o limite da agregação ao nível do score sobre máscaras
  rasterizadas oblíquas no regime atual.
- **Manter a Baseline 1 como baseline determinística de
  referência.** Para o conjunto MVP atual, qualquer comparação
  futura — determinística ou aprendida — toma os números de
  doc 03 — secção 17.4 como ponto de comparação.

---

## 8. Próximos passos

Se trabalho futuro vier a explorar perceção multi-vista de forma
mais profunda, deve ser uma **fase separada**, fora do escopo da
Fase B da Baseline 2 e fora do escopo do baseline determinístico
validado. Não é compromisso desta nota; são direções possíveis,
registadas para referência:

- **Fusão geométrica verdadeira** dos splats por vista no
  referencial do mundo, antes de qualquer rasterização ou cálculo
  de score, no espírito do primeiro ponto de doc 04 — secção 2.7.
- **Representação canónica** da cavidade (e da peça) que seja
  independente da pose de câmara, de forma a eliminar a
  dependência observada nas contagens de pixels oblíquos por
  cavidade na iteração C.
- **Captura multi-câmara síncrona** com câmaras estáticas
  authored em USD (doc 04 — secções 1.7 e 2.7), de forma a
  eliminar a dependência face ao regime sequencial atual.

Por ora, a Fase B fica registada como evidência negativa /
diagnóstica. A representação multi-vista, na forma de agregação
ao nível do score sobre máscaras rasterizadas, foi testada e não
produziu melhoria face à Baseline 1 sobre o conjunto MVP convexo.
