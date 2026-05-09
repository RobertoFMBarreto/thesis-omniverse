# 09 — Fase E: representações geométricas aprendidas a partir de footprints 2D estáveis (proposta de design)

> Nota de implementação para futura conversão em secção LaTeX.
> Estado: proposta de design — não implementada.
> Data: 2026-05-09.

---

## 1. Motivação

A Fase D (doc 08 — secções 9 e 10) deixou um diagnóstico
preciso. Sob a representação de features escalares herdada da
Baseline 1 (doc 03 — secção 17), e sob a profundidade CAD
uniforme das cavidades MVP, a fronteira de decisão aprendida
reduz-se essencialmente a limiares sobre `area_ratio` e `iou` —
exactamente o conjunto de features que o cabeçote determinístico
da Baseline 1 já combina analiticamente. A árvore reproduz 4/4
no MVP com margens nulas em três pares; a regressão logística
colapsa em 1/4. Verifica-se, portanto, que a Fase D não
demonstrou, sobre o conjunto MVP convexo, ganho mensurável face
à Baseline 1.

A questão que fica em aberto é representacional, não de modelo.
A pergunta de investigação que motiva esta proposta é, na sua
forma estrita:

> "Pode uma representação geométrica aprendida inferir
> compatibilidade de inserção para formas convexas não vistas,
> sem depender de descritores geométricos manualmente desenhados?"

A pergunta é sobre representação aprendida, não sobre perceção
aprendida. Mantém-se em 2D top-down — o regime único onde a
perceção determinística da tese é demonstrável e estável (doc 03
— secção 17.4; doc 06 — secção 5; doc 07 — secção 5). A Fase E
é apresentada aqui como **proposta de design**, não como
resultado experimental, e respeita os mesmos não-objectivos que
foram fixados nas Fases B, C e D: sem reconstrução 3D, sem
fusão volumétrica, sem perceção visual aprendida, sem controlo
robótico, sem execução de inserção.

---

## 2. Posicionamento científico

A Fase E é, em rigor, aprendizagem de **representação**. Não é:

- aprendizagem de **perceção** — a perceção mantém-se
  determinística e externamente validada pela Baseline 1;
- aprendizagem de **controlo robótico** — não há agente, não há
  política, não há trajectória, não há agarre;
- raciocínio 3D completo — a representação é fundamentalmente
  2D, sobre a face top-down observável de forma estável;
- reconstrução — não há TSDF, não há voxel grid, não há malha;
  as Fases B e C (docs 06 e 07) já documentaram que o lado 3D
  do problema não é observável de forma fiável com o sensor
  actual.

A perceção mantém-se determinística e externamente validada. A
componente aprendida actua exclusivamente na fase de
representação e ranking de compatibilidade. O cabeçote
determinístico de scoring da Baseline 1 (doc 03 — secção 6)
deixa de ser usado como decisor; passa a coexistir como
referência comparativa. As features escalares hand-crafted da
Fase D (doc 08 — secção 2) também deixam de ser consumidas
directamente; são substituídas por uma representação vectorial
aprendida sobre as máscaras 2D já produzidas pelo pipeline
determinístico.

---

## 3. Escolha de representação

A escolha primária recai sobre **SDF (Signed Distance Field)**
calculado sobre as máscaras 2D top-down já existentes. A
justificação é a seguinte:

- **Máscaras binárias** são a representação consagrada e
  validada pela Baseline 1; pertencem ao espaço onde a
  discriminação MVP está demonstrada. São porém pobres em
  estrutura local: o sinal é binário (codificado vs
  não-codificado), sem qualquer noção de proximidade ao bordo,
  e por essa razão são uma entrada com densidade de informação
  baixa para um encoder pequeno.
- **SDF** preserva proximidade ao bordo e codifica
  explicitamente a estrutura de folga (clearance) entre o
  interior da cavidade e o seu contorno, e simetricamente entre
  o exterior da peça e a sua silhueta. Constitui um sinal
  contínuo e localmente liso, com landscape de optimização mais
  suave do que o de uma máscara binária e com densidade de
  informação por pixel mais elevada. É a candidata primária
  desta proposta.
- **Point clouds 3D** são rejeitadas. As Fases B e C (docs 06
  e 07) demonstraram que as observações 3D laterais e oblíquas
  são incompletas, dependentes da pose de câmara e instáveis
  para fins de scoring. Reintroduzi-las como entrada nesta fase
  contradiria a evidência negativa já registada.
- **Voxel grids / TSDF** são rejeitadas. Não existe sensor
  volumétrico fiável no setup actual; a tentativa de
  reconstrução volumétrica foi feita na Fase B e produziu
  representações sem regime intermédio satisfatório (doc 06 —
  secção 3.2). É inconsistente reabrir essa via numa fase de
  representação.

A recomendação primária é, portanto, SDF como entrada de
encoder. Como ablação razoável, propõe-se executar uma segunda
configuração com máscaras binárias como entrada, para isolar o
contributo da representação SDF face à da arquitectura.

---

## 4. Pipeline de entrada

O stack de perceção fica **frozen**. A Fase E **não** modifica
qualquer componente do pipeline determinístico que sustenta a
Baseline 1 e a Fase D. As entradas são:

- **Peça**: máscara footprint top-down produzida por
  `rasterise_xy_to_mask` em
  `scripts/baseline1_geometric_matching.py:216`. A conversão
  para SDF é uma operação local sobre a máscara já existente,
  por exemplo via `cv2.distanceTransform` aplicada à máscara e
  ao seu complemento, com sinal a indicar interior/exterior.
- **Cavidade**: máscara opening derivada do
  `cavity_opening_pointcloud.npy` validado pela Baseline 1
  (doc 03 — secção 4). Conversão para SDF análoga à da peça.
- **Alinhamento**: as rotações candidatas no plano são
  herdadas do varrimento de 180 ângulos da Baseline 1 (doc 03
  — secção 6). **Não há estimação de pose aprendida** nesta
  fase; a busca em rotação permanece exaustiva e
  determinística, exactamente como na Baseline 1, para isolar
  a representação como única fonte de variação.

Decorre desta política que qualquer ganho ou regressão observado
na Fase E é atribuível à representação aprendida e não a
mudanças no pipeline de perceção ou no espaço de busca.

---

## 5. Formulação de aprendizagem

A formulação proposta é um **encoder siamês** (dual-encoder com
pesos partilhados). O fluxo conceptual é:

```
piece_mask_or_sdf  -> encoder -> embedding_p  (vector R^d)
cavity_mask_or_sdf -> encoder -> embedding_c  (vector R^d)
compatibility(piece, cavity) = cos_sim(embedding_p, embedding_c)
                                OR small MLP head([embedding_p; embedding_c])
```

A regra de ranking é uma extensão directa da Baseline 1:

```
para cada cavidade:
  para cada rotação candidata:
    score = compatibility(piece_rotated, cavity)
  cavity_score = max sobre rotações
ranquear cavidades por cavity_score descendente
```

Os embeddings são vectores de baixa dimensão, com `d ∈ {16, 32,
64}` como ponto de operação a escolher por validação. Os dois
encoders **partilham pesos** (siamese): esta restrição força a
rede a aprender uma noção comum de "forma encaixável" e impede
que o encoder de peça e o encoder de cavidade se especializem
em assinaturas independentes que poderiam memorizar o emparelhamento
em vez de o generalizar.

A cabeça de scoring começa pela versão sem parâmetros,
`cos_sim`, por duas razões: é auditável (é geometricamente uma
medida de alinhamento angular no espaço de embedding) e não
acrescenta parâmetros adicionais que possam memorizar pares.
Uma MLP head de pequena dimensão é admitida como ablação se a
baseline cosseno se revelar insuficiente; **não** é o decisor
por defeito.

---

## 6. Hipótese

A hipótese a testar é falsificável e tem a seguinte formulação
estrita:

> "Uma representação geométrica aprendida (embedding siamês
> treinado em formas convexas procedimentais) atinge
> generalização LOFO de ranking estritamente superior a (a) os
> modelos descritores hand-crafted da Fase D (doc 08 —
> secção 7), ou (b) o limiar determinístico da Baseline 1
> (doc 03 — secção 17.4), em particular em famílias geométricas
> não vistas e em máscaras perturbadas."

A hipótese **não** assume superioridade. É enunciada como
afirmação testável; a sua falsificação tem valor experimental
próprio, conforme detalhado em §14.

---

## 7. Estratégia de dataset

A Fase E reaproveita integralmente a geração procedimental da
Fase D (doc 08 — secção 5), de forma a manter controlo
experimental directo entre as duas fases:

- Cinco famílias procedimentais: `rectangle`, `ellipse`,
  `regular_polygon`, `convex_irregular_polygon`,
  `rounded_rectangle`.
- Vinte instâncias por família, mais quatro peças MVP hold-in
  (`rectangle`, `square`, `circle`, `triangle`).
- O `star` (côncavo) fica reservado como stress-test out-of-
  distribution. **Não** entra no conjunto de treino; é admitido
  apenas como teste OOD opcional, com a expectativa documentada
  de falha (ver §12).
- Os rótulos provêm da regra determinística de inserção parcial
  da Fase D.7 (doc 08 — secção 4):

```
insertion_required_mm = max(MIN_REQUIRED_INSERTION_MM = 5.0,
                             INSERTION_FRACTION = 0.25 * piece_height_mm)
depth_ok = (cavity_depth_mm >= insertion_required_mm − DEPTH_TOLERANCE_MM = 0.5)
            AND (cavity_depth_mm >= MIN_INSERTION_GUIDANCE_MM = 5.0)
label = lateral_ok AND depth_ok
```

Não há anotação manual. Os rótulos são derivados pela mesma
regra geométrica determinística usada na Fase D, sem alteração
dos pontos de operação fixos.

---

## 8. Rótulos e supervisão

Os rótulos são feasibility geométrica determinística (Phase D.7),
**não** acções robóticas, **não** dinâmica de contacto, **não**
trajectórias planeadas. A rede aprende uma representação de
compatibilidade que, quando combinada via `cos_sim` ou MLP head,
prediz essa label binária.

O treino é supervisão binária standard (cross-entropy sobre a
label de feasibility). A componente que é efectivamente
aprendida é o encoder; o critério de feasibility — a regra de
inserção parcial — não é aprendido nem afinado, é declarado a
priori. Esta separação é deliberada: garante que o encoder
herda a definição física da Fase D e que as comparações entre
Fase E e Fase D incidem exclusivamente sobre a qualidade da
representação.

---

## 9. Candidatos a modelo

Por ordem de preferência:

- **Primário: shallow CNN encoder.** Tipicamente 3 a 4 blocos
  convolucionais sobre máscara/SDF 2D (canvas comparável aos
  320×320 px @ 0.25 mm/px da Baseline 1), seguidos de pooling
  global e projecção para um embedding de `d ∈ {16, 32, 64}`.
  Treinável em CPU, auditável, reprodutível, com risco de
  memorização baixo.
- **Secundário (apenas como ablação se o primário não chegar):**
  encoder lightweight tipo ResNet, com 6 a 8 blocos e skip
  connections, mantido **abaixo de 1 M parâmetros** para
  preservar interpretabilidade de capacidade e tempo de treino.

São explicitamente **rejeitados**: transformers; modelos de
difusão; PointNet++ e variantes; modelos pré-treinados grandes.
A justificação é tripla: (i) o âmbito é controlado e o conjunto
procedimental é modesto (~150 peças), donde uma capacidade
elevada favorece memorização; (ii) a tese privilegia
interpretabilidade-primeiro, em linha com a higiene
experimental adoptada nas Fases B, C e D; (iii) não é assumida
GPU dedicada para esta fase. A escolha minimalista é coerente
com o princípio de baseline declarado em doc 03 — secção 1 e
com os não-objectivos das fases anteriores.

---

## 10. Plano de avaliação

Métricas primárias:

- **LOFO ranking accuracy** (top-1 sobre família held-out),
  computada por leave-one-family-out exactamente como em
  doc 08 — secção 7;
- **MRR** (Mean Reciprocal Rank) sobre as cavidades candidatas
  para cada peça do conjunto held-out;
- **Margem média rank-1 vs rank-2**, comparável à coluna de
  margens da Fase D (doc 08 — secção 9) e da Baseline 1 (doc 03
  — secção 17.4).

Cada métrica é reportada lado-a-lado com a Fase D (logreg e
tree, doc 08 — secção 7) e com a Baseline 1 (doc 03 —
secção 17.4), de forma a tornar a comparação directa.

Métricas secundárias: as experiências de robustez descritas
em §11.

Política de reporte: todos os resultados são reportados,
independentemente do sinal — positivo, neutro ou negativo. A
convenção é a mesma das Fases B, C e D, conforme doc 06 —
secção 7, doc 07 — secção 7 e doc 08 — secção 12.

---

## 11. Experiências de robustez (ponte para futuro)

A Fase E habilita um conjunto de perturbações controladas
sobre as máscaras de entrada, a aplicar tanto à entrada do
encoder aprendido como à da Baseline 1, de forma a comparar
degradação relativa. As perturbações propostas:

- **Erosão / dilatação morfológica** com raio entre 1 e 3 px
  — simula sub-segmentação ou over-segmentação no estágio de
  perceção determinística;
- **Ruído gaussiano de bordo**, com `σ_pixel` pequeno aplicado
  às coordenadas do contorno antes da rasterização;
- **Oclusão parcial**, mascarando uma faixa rectangular
  aleatória da máscara;
- **Segmentos de contorno em falta**, removendo um arco
  contínuo do bordo;
- **Offsets de centroide** entre `0.5 mm` e `2 mm` antes da
  centragem;
- **Variações de resolução de rasterização**, e.g.
  `0.20 / 0.25 / 0.30 mm/px`, mantendo o resto do pipeline
  inalterado.

A pergunta de robustez é estrita: a representação aprendida
degrada **mais graciosamente** do que a regra determinística
sob estas perturbações? A métrica é a queda relativa em top-1
LOFO em função da magnitude da perturbação.

Este conjunto de experiências constitui também a ponte natural
para uma fase futura de generalização (informalmente
referenciada como "Option 3" no caderno de campo), sem que a
Fase E se comprometa desde já com essa fase.

---

## 12. Modos de falha

Antecipa-se um conjunto de modos de falha, cada um com
mecanismo de detecção próprio:

- **Overfit a área de silhueta.** O encoder pode reduzir-se a
  um detector de tamanho, em analogia directa ao colapso da
  regressão logística da Fase D ("menor peça encaixa em maior
  cavidade"; doc 08 — secção 9). Detectável por comparação
  contra um classificador trivial baseado apenas em área.
- **Memorização de famílias.** O encoder pode aprender uma
  assinatura por família — `ellipse` vs `regular_polygon`
  é detectável por compactness, por exemplo. Detectável por
  LOFO.
- **Colapso para "maior cavidade ganha".** Mesmo modo de falha
  da Fase D logreg sob profundidade uniforme. Detectável por
  inspecção do ranking MVP.
- **Sensibilidade a artefactos de rasterização.** Máscaras com
  diferentes pitch podem produzir embeddings significativamente
  distintos. Detectável pela experiência de variação de
  resolução em §11.
- **Falta de transferência para formas côncavas.** O `star`
  provavelmente falha; é o resultado esperado e documentável,
  consistente com o domínio convexo declarado em doc 08 —
  secção 11, ponto 1.
- **Incapacidade de raciocinar sobre geometria 3D escondida.**
  Paredes da cavidade, faces laterais da peça e undercuts não
  são observáveis no regime top-down. A representação é
  fundamentalmente 2D; isto é uma propriedade do design, não
  um defeito a corrigir nesta fase.

---

## 13. Notas de redação para evitar overclaim

Para futura conversão em texto LaTeX, ficam fixadas as
seguintes formulações como aceitáveis:

- "representação aprendida de compatibilidade geométrica";
- "ranking de affordance baseado em embeddings";
- "encoder siamês treinado em formas convexas procedimentais".

E ficam fixadas, como exemplos a evitar pela razão indicada
entre parêntesis:

- "inteligência geral de inserção robótica" (a Fase E não
  envolve nem agente nem execução);
- "compreensão 3D de affordance" (a representação é 2D
  top-down);
- "compreensão completa de objectos" (não há reconstrução
  volumétrica);
- "controlo robótico" (fora do escopo da tese);
- "perceção aprendida" (a perceção é determinística; o que é
  aprendido é a representação de compatibilidade);
- "Phase E supera Baseline 1" (afirmação a reservar
  exclusivamente para depois de prova experimental, em coerência
  com a higiene das Fases B, C e D).

---

## 14. Critérios de saída

Todos os desfechos abaixo são considerados publicáveis,
seguindo a mesma convenção que tornou as Fases B, C e D
informativas:

- **Sucesso.** Melhoria estatisticamente significativa em
  ranking LOFO, ou em robustez à perturbação, face à Baseline 1
  e à Fase D.
- **Resultado neutro.** Paridade com a Baseline 1 mas com
  robustez melhorada — por exemplo, degradação mais graciosa
  sob perturbação. Publicável como contributo, dado que documenta
  uma propriedade não-trivial da representação aprendida.
- **Resultado negativo.** Os embeddings colapsam para
  heurísticas baseadas em área, ou falham em LOFO. Publicável
  como evidência negativa controlada, em coerência com a
  narrativa da tese: as Fases B, C e D já incluem resultados
  negativos ou positivos-mas-estreitos cuidadosamente registados
  (doc 06 — secção 5; doc 07 — secção 5; doc 08 — secção 10).

---

## 15. Recomendação

Recomenda-se a Fase E como exploração cientificamente útil pelas
seguintes razões:

- Investiga representações geométricas aprendidas **sem** exigir
  reconstrução 3D arbitrária instável, mantendo-se em 2D
  top-down — o regime único onde a perceção é fiável (doc 03
  — secção 17; docs 06 e 07);
- Estende naturalmente a Fase D (doc 08 — secções 5 e 10)
  preservando controlo experimental: mesma família de dados,
  mesmas labels determinísticas, mesma higiene de avaliação;
- Cria uma ponte limpa para estudos de robustez e generalização
  (futura Fase F / "Option 3") sem que a Fase E se comprometa
  desde já com esse escopo;
- O encoder siamês com `cos_sim` é a forma mais minimalista de
  aprendizagem de representação que o problema admite, e
  rejeita explicitamente alternativas grandes/escuras que
  introduziriam risco de memorização e custo de auditoria
  desproporcionados ao âmbito da tese.

A Fase E **não** se propõe substituir a Baseline 1 (doc 03 —
secção 17) como referência operacional sobre o conjunto MVP
convexo. Propõe-se como exploração de representação alinhada
com o objectivo declarado da tese — aprender relações
perceção-acção a partir de geometria — mantendo as restrições
de escopo que tornaram informativas as fases anteriores.
