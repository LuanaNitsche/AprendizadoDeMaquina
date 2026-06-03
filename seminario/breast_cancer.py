import numpy as np
import pandas as pd
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


def carregar_dados(caminho: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(caminho)
    df = df.drop(columns=["id"])
    df = df.dropna(axis=1, how="all")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    X = df.drop(columns=["diagnosis"]).values
    y = df["diagnosis"].values
    return X, y


def dividir_dados(
    X: np.ndarray,
    y: np.ndarray,
    proporcao_teste: float = PROPORCAO_TESTE,
    semente: int = SEMENTE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=proporcao_teste, random_state=semente)


def aplicar_preprocessamento(
    knn: KNN,
    tipo: str | None,
    treino: np.ndarray,
    teste: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[int, float]:
    melhor_k = 0
    melhor_acuracia = 0.0

    for k in FAIXA_K:
        rotulos_previstos = knn.meuKnn(treino, rotulos_treino, teste, k)
        acuracia = np.mean(rotulos_previstos == rotulos_teste)
        print(f"  k={k} -> acurácia={acuracia:.4f}")

        if acuracia > melhor_acuracia:
            melhor_acuracia = acuracia
            melhor_k = k

    return melhor_k, melhor_acuracia


def avaliar_preprocessamentos(
    knn: KNN,
    treino_bruto: np.ndarray,
    teste_bruto: np.ndarray,
    rotulos_treino: np.ndarray,
    rotulos_teste: np.ndarray,
) -> dict[str, tuple[int, float]]:
    resultados = {}

    for tipo in [None, "normalizar", "padronizar"]:
        print(f"\n=========================")
        print(f"TIPO: {tipo}")
        print(f"=========================")

        treino, teste = aplicar_preprocessamento(knn, tipo, treino_bruto, teste_bruto)
        melhor_k, melhor_acuracia = buscar_melhor_k(
            knn, treino, rotulos_treino, teste, rotulos_teste
        )
        resultados[str(tipo)] = (melhor_k, melhor_acuracia)
        print(f"\n>>> Melhor k ({tipo}): {melhor_k} -> {melhor_acuracia:.4f}")

    return resultados


def avaliar_subconjuntos(
    knn: KNN,
    treino: np.ndarray,
    teste: np.ndarray,
    rotulos_treino: np.ndarray,
    rotulos_teste: np.ndarray,
    nome_preprocessamento: str,
) -> dict[str, tuple[int, float]]:
    print(f"\n=========================")
    print(f"SUBCONJUNTOS ({nome_preprocessamento.upper()})")
    print(f"=========================")

    resultados = {}

    for nome, colunas in SUBCONJUNTOS.items():
        print(f"\n--- SUBSET {nome} ---")
        melhor_k, melhor_acuracia = buscar_melhor_k(
            knn,
            treino[:, colunas],
            rotulos_treino,
            teste[:, colunas],
            rotulos_teste,
        )
        resultados[nome] = (melhor_k, melhor_acuracia)
        print(f">>> Melhor {nome}: k={melhor_k} -> {melhor_acuracia:.4f}")

    return resultados


def imprimir_resultados_finais(
    resultados_preprocessamentos: dict[str, tuple[int, float]],
    resultados_subconjuntos: dict[str, dict[str, tuple[int, float]]],
) -> None:
    print("\n=========================")
    print("RESULTADO FINAL")
    print("=========================")

    print("\nPré-processamentos (dados completos):")
    for tipo, (k, acuracia) in resultados_preprocessamentos.items():
        print(f"  {tipo}: melhor k={k} -> {acuracia:.4f}")

    for preprocessamento, resultados in resultados_subconjuntos.items():
        print(f"\nSubconjuntos com {preprocessamento}:")
        for subset, (k, acuracia) in resultados.items():
            print(f"  Subset {subset}: k={k} -> {acuracia:.4f}")


def executar() -> None:
    knn = KNN()

    X, y = carregar_dados(CAMINHO_DADOS)
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


if __name__ == "__main__":
    executar()
