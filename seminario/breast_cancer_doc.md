# Documentação — breast_cancer.py

Classificação do dataset **Breast Cancer Wisconsin** usando o algoritmo KNN implementado
manualmente (`knn.py`). O script compara três formas de pré-processamento (nenhum,
normalização e padronização) e avalia subconjuntos de features para encontrar o melhor
valor de k.

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

**Retorna:** `X` (n_amostras × n_features) e `y` (n_amostras,).

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

Mantém controle do k com maior acurácia durante a iteração.

**Retorna:** `(melhor_k, melhor_acuracia)`.

---

### `avaliar_preprocessamentos(knn, treino_bruto, teste_bruto, rotulos_treino, rotulos_teste)`

Orquestra a comparação dos três tipos de pré-processamento (`None`, `"normalizar"`,
`"padronizar"`) usando os **dados completos** (todas as features).

Para cada tipo:
1. Aplica o pré-processamento via `aplicar_preprocessamento`.
2. Chama `buscar_melhor_k` para encontrar o k ótimo.
3. Imprime os resultados e armazena no dicionário de retorno.

**Retorna:** `dict[tipo → (melhor_k, melhor_acuracia)]`.

---

### `avaliar_subconjuntos(knn, treino, teste, rotulos_treino, rotulos_teste, nome_preprocessamento)`

Avalia o desempenho do KNN para cada subconjunto de features definido em `SUBCONJUNTOS`,
usando os dados **já pré-processados** recebidos como parâmetro.

Para cada subconjunto:
1. Seleciona as colunas correspondentes de treino e teste.
2. Chama `buscar_melhor_k` sobre esse subconjunto reduzido.
3. Imprime e armazena o resultado.

**Retorna:** `dict[nome_subset → (melhor_k, melhor_acuracia)]`.

---

### `imprimir_resultados_finais(resultados_preprocessamentos, resultados_subconjuntos)`

Consolida e imprime todos os resultados ao final da execução:
- Melhor k de cada pré-processamento nos dados completos.
- Melhor k de cada subconjunto de features, por pré-processamento.

---

### `executar()`

Ponto de entrada principal. Orquestra toda a pipeline:

```
carregar_dados
    └── dividir_dados
            ├── avaliar_preprocessamentos   (dados completos: None / normalizar / padronizar)
            ├── visualizaPontos             (scatter plot: treino bruto, dimensões 0 e 1)
            ├── avaliar_subconjuntos        (normalização → subsets [0,1], [1,2], [23,29])
            ├── avaliar_subconjuntos        (padronização → subsets [0,1], [1,2], [23,29])
            ├── visualizaPontos             (scatter plot: treino normalizado, dimensões 0 e 1)
            └── imprimir_resultados_finais
```

---

## Fluxo completo de execução

```
1. Leitura do CSV (569 amostras, 30 features numéricas + diagnóstico)
2. Divisão treino/teste: 455 amostras / 114 amostras
3. Avaliação com dados completos:
   - Sem pré-processamento
   - Com normalização min-max
   - Com padronização z-score
   → Para cada um: testa k = 1..8 e registra o melhor
4. Scatter plot dos dados brutos (features 0 e 1)
5. Avaliação de subconjuntos de features com normalização:
   - Colunas [0, 1]
   - Colunas [1, 2]
   - Colunas [23, 29]
6. Avaliação de subconjuntos de features com padronização:
   - Colunas [0, 1]
   - Colunas [1, 2]
   - Colunas [23, 29]
7. Scatter plot dos dados normalizados (features 0 e 1)
8. Impressão consolidada de todos os resultados
```

---

## Dataset

O Breast Cancer Wisconsin contém medidas extraídas de imagens de biópsia de tumores
mamários. Cada amostra possui 30 features numéricas (raio, textura, perímetro, área,
suavidade, etc., calculadas como média, erro padrão e pior valor). O alvo é binário:
`1` = maligno, `0` = benigno.
