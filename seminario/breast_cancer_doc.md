# Documentação — breast_cancer.py

Classificação do dataset **Breast Cancer Wisconsin** usando o algoritmo KNN implementado
manualmente (`knn.py`). O script descreve o dataset, compara três formas de
pré-processamento (nenhum, normalização e padronização), avalia subconjuntos de features
para encontrar o melhor valor de k, analisa matrizes de confusão e compara o KNN com um
Random Forest (scikit-learn).

---

## Constantes globais

| Constante | Valor | Descrição |
|---|---|---|
| `CAMINHO_DADOS` | `"data.csv"` | Caminho para o arquivo do dataset |
| `PROPORCAO_TESTE` | `0.2` | 20 % dos dados reservados para teste |
| `SEMENTE` | `42` | Semente do gerador aleatório (garante reprodutibilidade) |
| `FAIXA_K` | `range(1, 9)` | Valores de k testados: 1, 2, 3, 4, 5, 6, 7, 8 |
| `SUBCONJUNTOS` | ver abaixo | Subconjuntos de colunas avaliados |

```python
SUBCONJUNTOS = {
    "[0,1]":   [0, 1],
    "[1,2]":   [1, 2],
    "[23,29]": [23, 29],
}
```

---

## Funções

### `carregar_dados(caminho)`

Lê o CSV e prepara as matrizes de entrada e rótulos.

**Etapas internas:**
1. Lê o arquivo CSV com `pandas`.
2. Remove a coluna `id` (identificador sem valor preditivo).
3. Descarta colunas 100 % NaN.
4. Codifica o diagnóstico: `M` (Maligno) → `1`, `B` (Benigno) → `0`.
5. Separa features (`X`) do rótulo alvo (`y`).

**Retorna:** `X` (n_amostras × n_features), `y` (n_amostras,) e `nomes_features`
(lista com os nomes das 30 colunas, usada depois na importância de features do
Random Forest).

---

### `descrever_dataset(caminho)`

Imprime uma descrição geral do dataset, usada para a análise exploratória inicial:

1. Dimensões (569 amostras × 30 features).
2. Distribuição das classes (benigno/maligno, contagem e percentual).
3. Os 10 grupos de características (radius, texture, perimeter, ... ,
   fractal_dimension), cada um presente como `_mean`, `_se` e `_worst`.
4. Estatísticas descritivas (média, desvio padrão, mínimo e máximo) das 10
   features `_mean`.

Não retorna valor — apenas imprime no console.

---

### `dividir_dados(X, y, proporcao_teste, semente)`

Divide os dados em treino (80 %) e teste (20 %) usando `train_test_split` do scikit-learn.

A semente fixa garante que a mesma divisão seja gerada em toda execução, tornando os
resultados comparáveis.

**Retorna:** `treino_X`, `teste_X`, `treino_y`, `teste_y`.

---

### `aplicar_preprocessamento(knn, tipo, treino, teste)`

Aplica o pré-processamento escolhido **apenas com base nos dados de treino**, evitando
*data leakage* no conjunto de teste.

| `tipo` | Operação |
|---|---|
| `None` | Dados brutos, sem transformação |
| `"normalizar"` | Normalização min-max → escala os valores para o intervalo [0, 1] |
| `"padronizar"` | Padronização z-score → média 0 e desvio padrão 1 |

Os parâmetros (mínimo/máximo ou média/desvio) são calculados no treino e aplicados em
ambos os conjuntos.

**Retorna:** `(treino_transformado, teste_transformado)`.

---

### `buscar_melhor_k(knn, treino, rotulos_treino, teste, rotulos_teste)`

Itera sobre todos os valores de k definidos em `FAIXA_K` (1 a 8), classifica o conjunto
de teste com `knn.meuKnn` e calcula a acurácia de cada k.

Mantém controle do k com maior acurácia durante a iteração, guardando também o vetor de
rótulos previstos por esse k (usado depois na matriz de confusão).

**Retorna:** `(melhor_k, melhor_acuracia, melhores_previsoes)`.

---

### `avaliar_preprocessamentos(knn, treino_bruto, teste_bruto, rotulos_treino, rotulos_teste)`

Orquestra a comparação dos três tipos de pré-processamento (`None`, `"normalizar"`,
`"padronizar"`) usando os **dados completos** (todas as features).

Para cada tipo:
1. Aplica o pré-processamento via `aplicar_preprocessamento`.
2. Chama `buscar_melhor_k` para encontrar o k ótimo.
3. Imprime os resultados e armazena no dicionário de retorno.

**Retorna:** `dict[tipo → (melhor_k, melhor_acuracia, melhores_previsoes)]`.

---

### `avaliar_subconjuntos(knn, treino, teste, rotulos_treino, rotulos_teste, nome_preprocessamento)`

Avalia o desempenho do KNN para cada subconjunto de features definido em `SUBCONJUNTOS`,
usando os dados **já pré-processados** recebidos como parâmetro.

Para cada subconjunto:
1. Seleciona as colunas correspondentes de treino e teste.
2. Chama `buscar_melhor_k` sobre esse subconjunto reduzido.
3. Imprime e armazena o resultado.

**Retorna:** `dict[nome_subset → (melhor_k, melhor_acuracia, melhores_previsoes)]`.

---

### `imprimir_resultados_finais(resultados_preprocessamentos, resultados_subconjuntos)`

Consolida e imprime todos os resultados ao final da execução:
- Melhor k de cada pré-processamento nos dados completos.
- Melhor k de cada subconjunto de features, por pré-processamento.

---

### `analisar_matrizes_confusao(knn, resultados_preprocessamentos, resultados_subconjuntos, rotulos_teste)`

Identifica a melhor configuração com dados completos e o melhor subconjunto (maior
acurácia em cada caso) e, para cada um, chama `knn.matrizConfusao` com as previsões
correspondentes, exibindo VP/VN/FP/FN, precisão, revocação, especificidade e F1-score.

---

### `avaliar_random_forest(knn, treino, teste, rotulos_treino, rotulos_teste, nomes_features)`

Treina um `RandomForestClassifier` (scikit-learn, `random_state=SEMENTE`) com os dados
brutos (sem pré-processamento, pois árvores não são sensíveis à escala) e:

1. Calcula a acurácia no conjunto de teste.
2. Exibe a matriz de confusão via `knn.matrizConfusao`.
3. Lista as 5 features mais importantes (`feature_importances_`), pelo nome.

Funciona como o "segundo algoritmo" de comparação com o KNN manual.

---

### `executar()`

Ponto de entrada principal. Orquestra toda a pipeline:

```
descrever_dataset
carregar_dados
    └── dividir_dados
            ├── avaliar_preprocessamentos   (dados completos: None / normalizar / padronizar)
            ├── visualizaPontos             (scatter plot: treino bruto, dimensões 0 e 1)
            ├── avaliar_subconjuntos        (normalização → subsets [0,1], [1,2], [23,29])
            ├── avaliar_subconjuntos        (padronização → subsets [0,1], [1,2], [23,29])
            ├── visualizaPontos             (scatter plot: treino normalizado, dimensões 0 e 1)
            ├── imprimir_resultados_finais
            ├── analisar_matrizes_confusao  (melhor pré-processamento e melhor subconjunto)
            └── avaliar_random_forest       (segundo algoritmo, com matriz de confusão)
```

---

## Fluxo completo de execução

```
1. Descrição do dataset (dimensões, balanceamento de classes, grupos de features, estatísticas)
2. Leitura do CSV (569 amostras, 30 features numéricas + diagnóstico)
3. Divisão treino/teste: 455 amostras / 114 amostras
4. Avaliação com dados completos:
   - Sem pré-processamento
   - Com normalização min-max
   - Com padronização z-score
   → Para cada um: testa k = 1..8 e registra o melhor (com previsões)
5. Scatter plot dos dados brutos (features 0 e 1)
6. Avaliação de subconjuntos de features com normalização:
   - Colunas [0, 1]
   - Colunas [1, 2]
   - Colunas [23, 29]
7. Avaliação de subconjuntos de features com padronização:
   - Colunas [0, 1]
   - Colunas [1, 2]
   - Colunas [23, 29]
8. Scatter plot dos dados normalizados (features 0 e 1)
9. Impressão consolidada de todos os resultados
10. Matrizes de confusão (melhor pré-processamento e melhor subconjunto)
11. Random Forest: acurácia, matriz de confusão e features mais importantes
```

---

## Dataset

O Breast Cancer Wisconsin contém medidas extraídas de imagens de biópsia de tumores
mamários. Cada amostra possui 30 features numéricas (raio, textura, perímetro, área,
suavidade, etc., calculadas como média, erro padrão e pior valor). O alvo é binário:
`1` = maligno, `0` = benigno.
