from knn import KNN
import numpy as np


knn = KNN()

grupoTrain, grupoTest, trainRots, testRots = knn.carregarDados(
    "classificacao/grupoDados4.mat"
)

rotuloPrevisto = knn.meuKnn(grupoTrain, trainRots, grupoTest, k=1)
print("Q4.1 — k=1, sem normalização:")
knn.acuracia(rotuloPrevisto, testRots)

print("\nQ4.2 — diferentes k com normalização:")
grupoTrainN, grupoTestN, trainRots, testRots = knn.carregarDados(
    "classificacao/grupoDados4.mat", normalizar=True
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
Q4.1: Aplique seu algoritmo K-NN ao problema. Qual é a sua acurácia de classificação?
R: Com k = 1 e sem qualquer pré-processamento, a acurácia obtida foi bem abaixo de
92%. O kNN depende de distâncias euclidianas, portanto é muito sensível à escala
das features.

Q4.2: A acurácia pode chegar a 92% com o K-NN. Descubra por que o resultado atual
é muito menor. Ajuste o conjunto de dados ou o valor de k de forma que a acurácia
atinja 92% e explique o que você fez e por quê. Observe que, desta vez, há mais de
um problema...
R: Notamos dois problemas, o primeiro é a ausência de normalização: 
    as features possuem escalas muito diferentes. Sem normalizar, features com valores maiores 
    pesam mais no cálculo e tornam as demais irrelevantes. 
    A solução foi aplicar normalização min-max calculada sobre o treino e aplicada ao teste (sem data leakage).

    E o segundo problema é a escolha inadequada de k: 
    com k = 1, o modelo é excessivamente sensível a ruídos e pontos isolados próximos a fronteira de decisão. 
    Ao aumentar k, o modelo considera mais vizinhos, tornando a fronteira mais suave e robusta. 
    Com os dados normalizados e o k ajustado, a acurácia atinge 92%
"""
