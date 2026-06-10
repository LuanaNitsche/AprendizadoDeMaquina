"""
Classifica o dataset Breast Cancer Wisconsin (maligno/benigno) com KNN implementado
manualmente, comparando pré-processamentos (nenhum, normalização, padronização),
valores de k e subconjuntos de features. Também avalia matrizes de confusão e
compara o KNN com um Random Forest (sklearn).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from knn import KNN

CAMINHO_DADOS = "data.csv"
PROPORCAO_TESTE = 0.2
SEMENTE = 42
FAIXA_K = range(1, 9)

SUBCONJUNTOS = {
    "[0,1]": [0, 1],
    "[1,2]": [1, 2],
    "[23,29]": [23, 29],
}


def carregar_dados(caminho: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Lê o CSV, remove a coluna id e converte o diagnóstico (M/B) em rótulos 1/0."""
    df = pd.read_csv(caminho)
    df = df.drop(columns=["id"])
    df = df.dropna(axis=1, how="all")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    X = df.drop(columns=["diagnosis"]).values
    y = df["diagnosis"].values
    nomes_features = df.drop(columns=["diagnosis"]).columns.tolist()
    return X, y, nomes_features


def descrever_dataset(caminho: str) -> None:
    """Imprime uma descrição do dataset: dimensões, balanceamento de classes,
    grupos de features e estatísticas descritivas das features '_mean'."""
    df = pd.read_csv(caminho)
    df = df.drop(columns=["id"])
    df = df.dropna(axis=1, how="all")

    print("=========================")
    print("DESCRIÇÃO DO DATASET")
    print("=========================")
    print(f"Total de amostras: {len(df)}")
    print(f"Total de features: {df.shape[1] - 1}")

    contagem = df["diagnosis"].value_counts()
    total = len(df)
    print("\nDistribuição das classes:")
    print(f"  Benigno (B): {contagem.get('B', 0)} ({contagem.get('B', 0) / total:.2%})")
    print(f"  Maligno (M): {contagem.get('M', 0)} ({contagem.get('M', 0) / total:.2%})")

    medidas = [c[: -len("_mean")] for c in df.columns if c.endswith("_mean")]
    print("\nAs 30 features são 10 medidas calculadas como média (_mean),")
    print("erro padrão (_se) e pior valor (_worst):")
    for medida in medidas:
        print(f"  - {medida}")

    colunas_mean = [c for c in df.columns if c.endswith("_mean")]
    print("\nEstatísticas descritivas das features '_mean' (média, desvio, mín, máx):")
    print(df[colunas_mean].describe().loc[["mean", "std", "min", "max"]].round(2).to_string())


def dividir_dados(
    X: np.ndarray,
    y: np.ndarray,
    proporcao_teste: float = PROPORCAO_TESTE,
    semente: int = SEMENTE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide X e y em treino/teste com proporção e semente fixas."""
    return train_test_split(X, y, test_size=proporcao_teste, random_state=semente)


def aplicar_preprocessamento(
    knn: KNN,
    tipo: str | None,
    treino: np.ndarray,
    teste: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica normalização, padronização ou nada (None) ao treino/teste, conforme `tipo`."""
    if tipo == "normalizar":
        return knn.normalizacao(treino, teste)
    if tipo == "padronizar":
        return knn.padronizacao(treino, teste)
    return treino, teste


def buscar_melhor_k(
    knn: KNN,
    treino: np.ndarray,
    rotulos_treino: np.ndarray,
    teste: np.ndarray,
    rotulos_teste: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    """Testa cada k em FAIXA_K com o KNN e retorna o k de maior acurácia,
    junto das previsões correspondentes (para análise posterior, ex.: matriz de confusão)."""
    melhor_k = 0
    melhor_acuracia = 0.0
    melhores_previsoes = None

    for k in FAIXA_K:
        rotulos_previstos = knn.meuKnn(treino, rotulos_treino, teste, k)
        acuracia = np.mean(rotulos_previstos == rotulos_teste)
        print(f"  k={k} -> acurácia={acuracia:.4f}")

        if acuracia > melhor_acuracia:
            melhor_acuracia = acuracia
            melhor_k = k
            melhores_previsoes = rotulos_previstos

    return melhor_k, melhor_acuracia, melhores_previsoes


def avaliar_preprocessamentos(
    knn: KNN,
    treino_bruto: np.ndarray,
    teste_bruto: np.ndarray,
    rotulos_treino: np.ndarray,
    rotulos_teste: np.ndarray,
) -> dict[str, tuple[int, float, np.ndarray]]:
    """Compara None/normalizar/padronizar nos dados completos, buscando o melhor k de cada um."""
    resultados = {}

    for tipo in [None, "normalizar", "padronizar"]:
        print(f"\n=========================")
        print(f"TIPO: {tipo}")
        print(f"=========================")

        treino, teste = aplicar_preprocessamento(knn, tipo, treino_bruto, teste_bruto)
        melhor_k, melhor_acuracia, melhores_previsoes = buscar_melhor_k(
            knn, treino, rotulos_treino, teste, rotulos_teste
        )
        resultados[str(tipo)] = (melhor_k, melhor_acuracia, melhores_previsoes)
        print(f"\n>>> Melhor k ({tipo}): {melhor_k} -> {melhor_acuracia:.4f}")

    return resultados


def avaliar_subconjuntos(
    knn: KNN,
    treino: np.ndarray,
    teste: np.ndarray,
    rotulos_treino: np.ndarray,
    rotulos_teste: np.ndarray,
    nome_preprocessamento: str,
) -> dict[str, tuple[int, float, np.ndarray]]:
    """Para cada par de colunas em SUBCONJUNTOS, busca o melhor k usando só essas features."""
    print(f"\n=========================")
    print(f"SUBCONJUNTOS ({nome_preprocessamento.upper()})")
    print(f"=========================")

    resultados = {}

    for nome, colunas in SUBCONJUNTOS.items():
        print(f"\n--- SUBSET {nome} ---")
        melhor_k, melhor_acuracia, melhores_previsoes = buscar_melhor_k(
            knn,
            treino[:, colunas],
            rotulos_treino,
            teste[:, colunas],
            rotulos_teste,
        )
        resultados[nome] = (melhor_k, melhor_acuracia, melhores_previsoes)
        print(f">>> Melhor {nome}: k={melhor_k} -> {melhor_acuracia:.4f}")

    return resultados


def imprimir_resultados_finais(
    resultados_preprocessamentos: dict[str, tuple[int, float, np.ndarray]],
    resultados_subconjuntos: dict[str, dict[str, tuple[int, float, np.ndarray]]],
) -> None:
    """Imprime um resumo final com os melhores k de cada pré-processamento e subconjunto."""
    print("\n=========================")
    print("RESULTADO FINAL")
    print("=========================")

    print("\nPré-processamentos (dados completos):")
    for tipo, (k, acuracia, _) in resultados_preprocessamentos.items():
        print(f"  {tipo}: melhor k={k} -> {acuracia:.4f}")

    for preprocessamento, resultados in resultados_subconjuntos.items():
        print(f"\nSubconjuntos com {preprocessamento}:")
        for subset, (k, acuracia, _) in resultados.items():
            print(f"  Subset {subset}: k={k} -> {acuracia:.4f}")


def analisar_matrizes_confusao(
    knn: KNN,
    resultados_preprocessamentos: dict[str, tuple[int, float, np.ndarray]],
    resultados_subconjuntos: dict[str, dict[str, tuple[int, float, np.ndarray]]],
    rotulos_teste: np.ndarray,
) -> None:
    """Mostra a matriz de confusão e métricas (precisão, revocação, especificidade, F1)
    para a melhor configuração com dados completos e para o melhor subconjunto."""
    print("\n=========================")
    print("ANÁLISE DE MATRIZES DE CONFUSÃO (KNN)")
    print("=========================")

    melhor_tipo = max(
        resultados_preprocessamentos, key=lambda t: resultados_preprocessamentos[t][1]
    )
    melhor_k, melhor_acuracia, melhores_previsoes = resultados_preprocessamentos[melhor_tipo]
    print(
        f"\n--- Melhor pré-processamento (dados completos): {melhor_tipo}, "
        f"k={melhor_k}, acurácia={melhor_acuracia:.4f} ---"
    )
    knn.matrizConfusao(melhores_previsoes, rotulos_teste)

    melhor_subset_nome = None
    melhor_subset_info = None
    for preprocessamento, resultados in resultados_subconjuntos.items():
        for nome_subset, (k, acuracia, previsoes) in resultados.items():
            if melhor_subset_info is None or acuracia > melhor_subset_info[1]:
                melhor_subset_nome = f"{nome_subset} ({preprocessamento})"
                melhor_subset_info = (k, acuracia, previsoes)

    melhor_k, melhor_acuracia, melhores_previsoes = melhor_subset_info
    print(
        f"\n--- Melhor subconjunto: {melhor_subset_nome}, "
        f"k={melhor_k}, acurácia={melhor_acuracia:.4f} ---"
    )
    knn.matrizConfusao(melhores_previsoes, rotulos_teste)


def avaliar_random_forest(
    knn: KNN,
    treino: np.ndarray,
    teste: np.ndarray,
    rotulos_treino: np.ndarray,
    rotulos_teste: np.ndarray,
    nomes_features: list[str],
) -> None:
    """Treina um Random Forest (sklearn) com os dados brutos como segundo algoritmo de
    comparação e mostra acurácia, matriz de confusão e features mais importantes."""
    print("\n=========================")
    print("RANDOM FOREST (segundo algoritmo)")
    print("=========================")

    modelo = RandomForestClassifier(random_state=SEMENTE)
    modelo.fit(treino, rotulos_treino)
    previsoes = modelo.predict(teste)
    acuracia = np.mean(previsoes == rotulos_teste)
    print(f"Acurácia: {acuracia:.4f}")

    knn.matrizConfusao(previsoes, rotulos_teste)

    importancias = sorted(
        zip(nomes_features, modelo.feature_importances_), key=lambda x: x[1], reverse=True
    )
    print("\nFeatures mais importantes:")
    for nome, importancia in importancias[:5]:
        print(f"  {nome}: {importancia:.4f}")


def executar() -> None:
    """Pipeline completa: descreve o dataset, avalia pré-processamentos, subconjuntos,
    matrizes de confusão, resultados finais e compara com Random Forest."""
    knn = KNN()

    descrever_dataset(CAMINHO_DADOS)

    X, y, nomes_features = carregar_dados(CAMINHO_DADOS)
    treino_bruto, teste_bruto, rotulos_treino, rotulos_teste = dividir_dados(X, y)

    resultados_preprocessamentos = avaliar_preprocessamentos(
        knn, treino_bruto, teste_bruto, rotulos_treino, rotulos_teste
    )

    knn.visualizaPontos(treino_bruto, rotulos_treino, 0, 1)

    treino_norm, teste_norm = knn.normalizacao(treino_bruto, teste_bruto)
    treino_pad, teste_pad = knn.padronizacao(treino_bruto, teste_bruto)

    resultados_subconjuntos = {
        "normalização": avaliar_subconjuntos(
            knn, treino_norm, teste_norm, rotulos_treino, rotulos_teste, "normalização"
        ),
        "padronização": avaliar_subconjuntos(
            knn, treino_pad, teste_pad, rotulos_treino, rotulos_teste, "padronização"
        ),
    }

    knn.visualizaPontos(treino_norm[:, [0, 1]], rotulos_treino, 0, 1)

    imprimir_resultados_finais(resultados_preprocessamentos, resultados_subconjuntos)

    analisar_matrizes_confusao(
        knn, resultados_preprocessamentos, resultados_subconjuntos, rotulos_teste
    )

    avaliar_random_forest(
        knn, treino_bruto, teste_bruto, rotulos_treino, rotulos_teste, nomes_features
    )


if __name__ == "__main__":
    executar()
