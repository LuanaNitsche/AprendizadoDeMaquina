from knn import KNN
import numpy as np

knn = KNN()

grupoTrain, grupoTest, trainRots, testRots = knn.carregarDados(
    "grupoDados3.mat"
)

rotuloPrevisto = knn.meuKnn(grupoTrain, trainRots, grupoTest, k=1)
print("Q3.1 — k=1, sem normalização:")
knn.acuracia(rotuloPrevisto, testRots)

print("\nQ3.2 — diferentes k com normalização:")
grupoTrainN, grupoTestN, trainRots, testRots = knn.carregarDados(
    "grupoDados3.mat", normalizar=True
)

for k in range(1, 11):
    rotuloPrevisto = knn.meuKnn(grupoTrainN, trainRots, grupoTestN, k)
    acc = np.mean(rotuloPrevisto == testRots)
    print(f"k={k} -> acurácia={acc:.4f}")

for k in range(1, 11):
    rotuloPrevisto = knn.meuKnn(grupoTrainN, trainRots, grupoTestN, k)
    acc = np.mean(rotuloPrevisto == testRots)
    if acc >= 0.92:
        print(f"\nPrimeiro k com acurácia >= 92%: k={k}")
        knn.acuracia(rotuloPrevisto, testRots)
        knn.visualizaPontos(grupoTrainN, trainRots, 0, 1)
        break

"""
Q3.1: Aplique o kNN ao problema usando k = 1. Qual é a acurácia na classificação?
R: Com k = 1 e sem normalização, a acurácia obtida foi inferior aproximadamente 58%, 
bem inferior 92%. O kNN é sensível a escala das features, e o conjunto de dados 3 
possui atributos com magnitudes muito diferentes, o que distorce o cálculo de distância 
euclidiana e prejudica a classificação.

Q3.2: A acurácia pode ser igual a 92% com o kNN. Descubra por que o resultado atual
é muito menor. Ajuste o conjunto de dados ou k de tal forma que a acurácia se torne
92% e explique o que você fez e por quê.
R: O principal problema era a ausência de normalização. Como o kNN usa distância
euclidiana, features com valores grandes pesam mais no cálculo e tornam as demais
irrelevantes. Ao aplicar a normalização min-max (calculada sobre o treino e
aplicada ao teste para evitar data leakage), cada feature passa a contribuir
igualmente. Com os dados normalizados e ajustando o valor de k, a acurácia atinge
92%
"""
