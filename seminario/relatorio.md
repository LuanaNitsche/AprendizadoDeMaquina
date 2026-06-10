# Classificação de Tumores Mamários com KNN — Breast Cancer Wisconsin

## 1. Introdução

O câncer de mama é uma das neoplasias malignas de maior incidência mundial,
representando um sério desafio de saúde pública. Segundo dados da Organização
Mundial da Saúde, é o tipo de câncer mais diagnosticado entre mulheres em todo
o mundo [1]. O diagnóstico precoce é fator determinante para melhores
prognósticos, porém o processo tradicional — baseado em análise
histopatológica e avaliação subjetiva de especialistas — está sujeito a
variabilidade entre observadores e pode ser demorado. Nesse contexto, técnicas
de aprendizado de máquina surgem como alternativas promissoras para apoio ao
diagnóstico clínico, permitindo classificar tumores automaticamente a partir
de características mensuráveis extraídas de imagens citológicas [2].

Este trabalho investiga a seguinte pergunta de pesquisa: é possível classificar
tumores mamários como malignos ou benignos com alta acurácia utilizando
algoritmos clássicos de aprendizado de máquina? Para isso, utilizou-se o
Wisconsin Breast Cancer Dataset (WBCD), disponível no repositório UCI Machine
Learning Repository, composto por 569 amostras com 30 atributos numéricos
derivados de imagens de punção aspirativa por agulha fina (PAAF), rotuladas
como benigno (357 casos) ou maligno (212 casos) [3].

O algoritmo principal selecionado para este estudo foi o K-Nearest Neighbors
(KNN), um método de aprendizado supervisionado que classifica uma nova amostra
com base na maioria dos rótulos entre seus k vizinhos mais próximos no espaço
de atributos, sem assumir uma distribuição prévia dos dados [4]. Devido à sua
simplicidade, interpretabilidade e bom desempenho em problemas de
classificação, o KNN mostra-se adequado para a tarefa proposta. Como segundo
algoritmo de comparação, foi utilizado o **Random Forest**, um método de
ensemble baseado em múltiplas árvores de decisão, que permite contrastar uma
abordagem baseada em distância (KNN) com uma abordagem baseada em partições do
espaço de atributos, além de fornecer uma medida direta de importância das
features [5].

---

## 2. Objetivos

### Objetivo Geral

Avaliar a eficácia do algoritmo K-Nearest Neighbors (KNN) na classificação de
tumores mamários como benignos ou malignos, utilizando o Wisconsin Breast
Cancer Dataset, considerando diferentes técnicas de pré-processamento, seleção
de atributos e comparando seu desempenho com o de um segundo algoritmo
(Random Forest).

### Objetivos Específicos

* Implementar o algoritmo K-Nearest Neighbors (KNN) para classificação
  supervisionada de dados.
* Descrever e caracterizar o dataset utilizado (dimensões, balanceamento de
  classes e grupos de atributos).
* Aplicar e comparar técnicas de pré-processamento, incluindo normalização e
  padronização dos atributos.
* Avaliar o impacto do hiperparâmetro k no desempenho do modelo.
* Investigar o desempenho do modelo utilizando subconjuntos de atributos
  específicos.
* Analisar os resultados por meio de matrizes de confusão e métricas derivadas
  (precisão, revocação, especificidade e F1-score), além da acurácia.
* Comparar o KNN implementado manualmente com um segundo algoritmo de
  classificação (Random Forest), avaliando acurácia, matriz de confusão e
  importância dos atributos.
* Identificar a melhor combinação de pré-processamento, valor de k e atributos
  para maximizar o desempenho do classificador.

---

## 3. Descrição do Dataset

O Wisconsin Breast Cancer Dataset (WBCD) contém **569 amostras**, cada uma
descrita por **30 atributos numéricos** extraídos de imagens digitalizadas de
punção aspirativa por agulha fina (PAAF) de massas mamárias. A coluna `id`
(identificador, sem valor preditivo) foi removida, assim como uma coluna
extra 100% vazia presente no arquivo original.

### 3.1 Distribuição das classes

| Classe | Quantidade | Percentual |
|---|---|---|
| Benigno (B → 0) | 357 | 62,74% |
| Maligno (M → 1) | 212 | 37,26% |

O dataset apresenta um desbalanceamento moderado (aproximadamente 63% / 37%),
o que reforça a importância de analisar não apenas a acurácia, mas também
métricas como precisão, revocação e especificidade — daí a relevância da
matriz de confusão (Seção 5.3).

### 3.2 Grupos de atributos

Os 30 atributos correspondem a **10 características morfológicas do núcleo
celular**, cada uma resumida por **3 estatísticas** (média, erro padrão e
"pior" valor — isto é, a média dos três maiores valores observados na
amostra), totalizando 10 × 3 = 30 colunas (`_mean`, `_se`, `_worst`):

| Característica | Descrição |
|---|---|
| `radius` | Raio médio (média das distâncias do centro aos pontos do perímetro) |
| `texture` | Desvio padrão dos valores de tons de cinza (textura da imagem) |
| `perimeter` | Perímetro do núcleo |
| `area` | Área do núcleo |
| `smoothness` | Suavidade (variação local nos comprimentos dos raios) |
| `compactness` | Compacidade (perímetro² / área − 1,0) |
| `concavity` | Concavidade (severidade das porções côncavas do contorno) |
| `concave points` | Número/intensidade de pontos côncavos no contorno |
| `symmetry` | Simetria do núcleo |
| `fractal_dimension` | Dimensão fractal ("aproximação da linha costeira" − 1) |

### 3.3 Estatísticas descritivas (atributos `_mean`)

| Atributo | Média | Desvio | Mín | Máx |
|---|---|---|---|---|
| radius_mean | 14,13 | 3,52 | 6,98 | 28,11 |
| texture_mean | 19,29 | 4,30 | 9,71 | 39,28 |
| perimeter_mean | 91,97 | 24,30 | 43,79 | 188,50 |
| area_mean | 654,89 | 351,91 | 143,50 | 2501,00 |
| smoothness_mean | 0,10 | 0,01 | 0,05 | 0,16 |
| compactness_mean | 0,10 | 0,05 | 0,02 | 0,35 |
| concavity_mean | 0,09 | 0,08 | 0,00 | 0,43 |
| concave points_mean | 0,05 | 0,04 | 0,00 | 0,20 |
| symmetry_mean | 0,18 | 0,03 | 0,11 | 0,30 |
| fractal_dimension_mean | 0,06 | 0,01 | 0,05 | 0,10 |

A grande disparidade de escalas entre atributos (por exemplo, `area_mean`
chega a 2501, enquanto `smoothness_mean` não ultrapassa 0,16) confirma a
importância de avaliar técnicas de pré-processamento, já que o KNN é sensível
à escala das variáveis (atributos com valores maiores dominam o cálculo da
distância euclidiana).

---

## 4. Metodologia

1. **Carregamento e limpeza:** remoção das colunas `id` e da coluna 100% NaN;
   mapeamento do rótulo `M → 1`, `B → 0`.
2. **Divisão treino/teste:** 80%/20% com `train_test_split` e
   `random_state=42`, garantindo reprodutibilidade (455 amostras de treino,
   114 de teste).
3. **Pré-processamento:** três cenários avaliados — sem processamento,
   normalização min-max e padronização (z-score). Os parâmetros (mín/máx ou
   média/desvio) são ajustados apenas no conjunto de treino e aplicados
   também ao teste, evitando *data leakage*.
4. **Seleção de k:** para cada cenário, k variou de 1 a 8. O melhor k foi
   selecionado pela maior acurácia no conjunto de teste.
5. **Análise de subsets:** subconjuntos de 2 atributos testados com
   normalização e padronização — os pares `[0,1]` (radius_mean/texture_mean),
   `[1,2]` (texture_mean/perimeter_mean) e `[23,29]`
   (area_worst/fractal_dimension_worst).
6. **Análise de matriz de confusão:** para a melhor configuração com todos os
   atributos e para o melhor subconjunto, foram calculadas a matriz de
   confusão (VP, VN, FP, FN) e as métricas de precisão, revocação,
   especificidade e F1-score.
7. **Segundo algoritmo (Random Forest):** treinado com `sklearn` sobre os
   dados brutos (sem necessidade de normalização, por ser um modelo baseado em
   árvores), com `random_state=42`. Avaliado com as mesmas métricas do KNN e
   com a importância de cada atributo (`feature_importances_`).
8. **Implementação:** KNN implementado manualmente em Python (classe `KNN`),
   sem uso de bibliotecas de ML para o classificador principal, respeitando o
   objetivo pedagógico do experimento. O Random Forest, por ser o algoritmo de
   comparação, utiliza a implementação do `scikit-learn`.

---

## 5. Resultados

### 5.1 Pré-processamento (todas as 30 features)

| Pré-processamento | Melhor k | Acurácia |
|---|---|---|
| Sem processamento | k = 5 | 95,61% |
| Padronização (z-score) | k = 4 | 95,61% |
| Normalização (min-max) | k = 2 | **97,37%** |

A normalização min-max apresentou o melhor desempenho geral, confirmando a
sensibilidade do KNN à escala dos atributos.

### 5.2 Subconjuntos de 2 atributos

| Subset | Pré-proc. | k | Acurácia |
|---|---|---|---|
| [0,1] radius_mean / texture_mean | Normalização | 2 | 92,11% |
| [1,2] texture_mean / perimeter_mean | Normalização | 5 | 94,74% |
| [23,29] area_worst / fractal_dimension_worst | Normalização | 1 | **100,00%** |
| [0,1] radius_mean / texture_mean | Padronização | 8 | 92,11% |
| [1,2] texture_mean / perimeter_mean | Padronização | 5 | 94,74% |
| [23,29] area_worst / fractal_dimension_worst | Padronização | 1 | **100,00%** |

O par `[23,29]` (area_worst, fractal_dimension_worst) atingiu **100% de
acurácia** no conjunto de teste com k=1, superando o modelo treinado com as 30
features.

### 5.3 Matrizes de confusão (KNN)

**Melhor configuração com todas as features** — normalização, k=2, acurácia =
97,37%:

| | Previsto Benigno | Previsto Maligno |
|---|---|---|
| **Real Benigno** | 70 (VN) | 1 (FP) |
| **Real Maligno** | 2 (FN) | 41 (VP) |

* Precisão: 0,9762
* Revocação (sensibilidade): 0,9535
* Especificidade: 0,9859
* F1-score: 0,9647

Dos 114 casos de teste, o modelo errou apenas 3: classificou 1 tumor benigno
como maligno (falso positivo) e 2 tumores malignos como benignos (falsos
negativos). Em diagnóstico de câncer, falsos negativos são os erros mais
preocupantes (um tumor maligno não detectado), o que sugere que, apesar da
alta acurácia, ainda há espaço para investigar configurações que reduzam
especificamente a taxa de falsos negativos (revocação).

**Melhor subconjunto** — `[23,29]` (area_worst / fractal_dimension_worst),
normalização, k=1, acurácia = 100%:

| | Previsto Benigno | Previsto Maligno |
|---|---|---|
| **Real Benigno** | 71 (VN) | 0 (FP) |
| **Real Maligno** | 0 (FN) | 43 (VP) |

* Precisão: 1,0000
* Revocação: 1,0000
* Especificidade: 1,0000
* F1-score: 1,0000

Todos os 114 casos de teste foram classificados corretamente.

### 5.4 Comparação com Random Forest

Como segundo algoritmo, treinou-se um **Random Forest** (scikit-learn,
`random_state=42`) sobre os dados brutos, com as 30 features:

* **Acurácia:** 96,49%

| | Previsto Benigno | Previsto Maligno |
|---|---|---|
| **Real Benigno** | 70 (VN) | 1 (FP) |
| **Real Maligno** | 3 (FN) | 40 (VP) |

* Precisão: 0,9756
* Revocação: 0,9302
* Especificidade: 0,9859
* F1-score: 0,9524

**Features mais importantes (Random Forest):**

| Atributo | Importância |
|---|---|
| area_worst | 0,1539 |
| concave points_worst | 0,1447 |
| concave points_mean | 0,1062 |
| radius_worst | 0,0780 |
| concavity_mean | 0,0680 |

Com as 30 features, o KNN normalizado (97,37%) superou ligeiramente o Random
Forest (96,49%) neste conjunto de teste, e cometeu menos falsos negativos (2
vs. 3). Por outro lado, o Random Forest confirma de forma independente parte
do achado da Seção 5.2: `area_worst` é o atributo mais importante segundo o
modelo de árvores, validando que essa feature carrega forte poder
discriminativo — o mesmo atributo que, combinado com
`fractal_dimension_worst`, levou o KNN a 100% de acurácia.

---

## 6. Discussão

* **Pré-processamento:** a normalização min-max foi consistentemente melhor
  que a ausência de pré-processamento e que a padronização, o que é esperado
  para o KNN, já que ele depende diretamente da escala dos atributos para
  calcular distâncias.
* **Seleção de atributos:** o subconjunto `[23,29]` (area_worst,
  fractal_dimension_worst) atingiu 100% de acurácia com k=1, sugerindo que
  esses dois atributos sozinhos já separam quase perfeitamente as classes
  neste conjunto de teste. Fisicamente, isso é coerente: área grande +
  dimensão fractal elevada indicam um núcleo grande e com contorno irregular,
  características associadas a células cancerosas agressivas (malignas).
* **Comparação de algoritmos:** o KNN manual (com normalização) obteve
  desempenho ligeiramente superior ao Random Forest com todas as features
  (97,37% vs. 96,49%), mas o Random Forest fornece uma vantagem adicional —
  importância de atributos — que corrobora de forma independente a relevância
  de `area_worst`, reforçando a robustez do achado da Seção 5.2.
* **Matrizes de confusão:** todos os modelos avaliados (exceto o subconjunto
  `[23,29]`) cometeram pelo menos 1 falso negativo. Em um cenário clínico real,
  a revocação (capacidade de identificar corretamente os casos malignos) é tão
  ou mais importante que a acurácia geral, e deve ser considerada na escolha
  do modelo final.

---

## 7. Conclusão

O experimento demonstrou que a combinação **normalização min-max + KNN com
k=2** é a configuração mais robusta para o dataset completo, atingindo 97,37%
de acurácia, com apenas 1 falso positivo e 2 falsos negativos em 114 amostras
de teste. A limpeza dos dados — especialmente a remoção da coluna vazia — foi
decisiva para o funcionamento correto do algoritmo.

A descoberta mais relevante foi que apenas 2 atributos (`area_worst` e
`fractal_dimension_worst`) são suficientes para separar perfeitamente os
tumores malignos dos benignos no conjunto de teste utilizado, superando o
modelo treinado com as 30 features. Essa descoberta foi reforçada pela análise
de importância de atributos do Random Forest, que apontou `area_worst` como o
atributo mais relevante entre os 30.

A comparação com o Random Forest mostrou que o KNN manual, quando bem
pré-processado, é competitivo com (e neste caso superou) um algoritmo de
ensemble mais sofisticado, validando a abordagem pedagógica do trabalho.

Assim: área grande + dimensão fractal elevada = tumor com núcleo grande e
contorno irregular, características associadas a células cancerosas
agressivas (maligno).

---

## 8. Limitações

* Uso de uma única divisão treino-teste, o que pode introduzir viés nos
  resultados — validação cruzada (k-fold) traria estimativas mais robustas.
* Tamanho moderado do dataset (569 amostras) e ausência de validação externa
  com dados de outras instituições.
* O custo computacional do KNN cresce com o número de amostras, podendo
  limitar sua aplicação em cenários com grandes volumes de dados.
* O resultado de 100% de acurácia no subconjunto `[23,29]` é específico desta
  divisão treino/teste e deste dataset; não deve ser interpretado como garantia
  de desempenho perfeito em dados novos.

Ainda assim, os resultados obtidos demonstram que o KNN é uma abordagem eficaz
para a classificação de tumores mamários com base em atributos citológicos, e
que sua acurácia é competitiva com algoritmos de ensemble mais complexos como
o Random Forest.

---

## Referências Bibliográficas

[1] WORLD HEALTH ORGANIZATION. *Breast cancer*. Geneva: WHO, 2024.
Disponível em:
https://books.google.com.br/books?hl=pt-BR&lr=&id=q6QOEQAAQBAJ&oi=fnd&pg=PR5&dq=WORLD+HEALTH+ORGANIZATION.+Breast+cancer.+Geneva:+WHO,+2024.&ots=reVXYE5K6U&sig=B_fE8-yoPg_nAwbe7jjagK9ambk#v=onepage&q=WORLD%20HEALTH%20ORGANIZATION.%20Breast%20cancer.%20Geneva%3A%20WHO%2C%202024.&f=false
Acesso em: 20 maio 2026.

[2] KOUROU, K.; EXARCHOS, T. P.; EXARCHOS, K. P.; KARAMOUZIS, M. V.; FOTIADIS,
D. I. Machine learning applications in cancer prognosis and prediction.
*Computational and Structural Biotechnology Journal*, v. 13, p. 8–17, 2015.
Disponível em: https://www.sciencedirect.com/science/article/pii/S2001037014000464
Acesso em: 30 maio 2026.

[3] UCI Machine Learning Repository. *Breast Cancer Wisconsin (Diagnostic)
Data Set*. Disponível em:
https://www.kaggle.com/datasets/krupadharamshi/breast-cancer-dataset/data
Acesso em: abril 2025.

[4] COVER, T. M.; HART, P. E. Nearest neighbor pattern classification. *IEEE
Transactions on Information Theory*, v. 13, n. 1, p. 21–27, 1967. Disponível
em: https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf
Acesso em: 20 maio 2026.

[5] BREIMAN, L. Random Forests. *Machine Learning*, v. 45, n. 1, p. 5–32,
2001.

[6] MITCHELL, T. M. *Machine Learning*. McGraw-Hill, 1997. cap. 8 –
Instance-Based Learning. Disponível em:
https://www.cs.cmu.edu/~tom/files/MachineLearningTomMitchell.pdf
Acesso em: 17 maio 2026.
