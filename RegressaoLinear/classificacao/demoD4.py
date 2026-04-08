from knn import KNN
import numpy as np


knn = KNN()

grupoTrain, grupoTest, trainRots, testRots = knn.carregarDados(
    "grupoDados4.mat"
)

rotuloPrevisto = knn.meuKnn(grupoTrain, trainRots, grupoTest, k=1)
print("Q4.1 — k=1, sem normalização, todas as features:")
knn.acuracia(rotuloPrevisto, testRots)

print("\nQ4.2 — features [0,1] normalizadas:")
grupoTrain01 = grupoTrain[:, [0, 1]]
grupoTest01 = grupoTest[:, [0, 1]]
grupoTrainN, grupoTestN = knn.normalizacao(grupoTrain01, grupoTest01)

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
Q4.1: Aplique seu algoritmo K-NN ao problema. Qual é a sua acurácia de classificação?
R: Com k = 1 e sem qualquer pré-processamento, a acurácia obtida foi de 71,67%,
bem abaixo de 92%.

Q4.2: A acurácia pode chegar a 92% com o K-NN. Descubra por que o resultado atual
é muito menor. Ajuste o conjunto de dados ou o valor de k de forma que a acurácia
atinja 92% e explique o que você fez e por quê. Observe que, desta vez, há mais de
um problema...

R: Foram identificados dois problemas:

    Problema 1 — Ausência de normalização:
    A feature 1 possui valores entre 278 e 1547, enquanto a feature 0 varia de 0 a 5.
    Como o k-NN usa distância euclidiana, a feature 1 domina completamente o cálculo
    e torna as demais features irrelevantes. A solução foi aplicar normalização min-max
    calculada sobre o treino e aplicada ao teste (sem data leakage).

    Problema 2 — Features 2 e 3 são ruído:
    As features 2 e 3 apresentam distribuições praticamente idênticas entre as três
    classes (mesma média ≈ 0.50 e desvio padrão ≈ 0.29 em todas as classes).
    Isso significa que essas features não carregam informação discriminativa —
    são ruído puro. Incluí-las no cálculo de distância adiciona variação aleatória
    que prejudica a classificação. A solução foi remover as features 2 e 3 e usar
    apenas as features 0 e 1.

    Com as features 0 e 1 normalizadas e k = 1, a acurácia atinge 93,33%.
"""