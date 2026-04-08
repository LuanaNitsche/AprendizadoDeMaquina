import numpy as np
from knn import KNN

knn = KNN()

grupoTrain, grupoTest, trainRots, testRots = knn.carregarDados(
    "classificacao/grupoDados2.mat", normalizar=True
)

for k in range(1, 9):
    rotuloPrevisto = knn.meuKnn(grupoTrain, trainRots, grupoTest, k)
    acc = np.mean(rotuloPrevisto == testRots)
    print(f"k={k} -> acurácia={acc:.4f}")

rotuloK3 = knn.meuKnn(grupoTrain, trainRots, grupoTest, 3)
acuracia = knn.acuracia(rotuloK3, testRots)

knn.visualizaPontos(grupoTrain, trainRots, 0, 1)

"""
Q2.1: Aplique seu kNN a este problema. Qual é a sua acurácia de classificação?
R: Aplicando o algoritmo k-NN com k = 1, foi obtida uma acurácia de aproximadamente 96,67% (58 acertos em 60 amostras).

Q2.2: A acurácia pode ser igual a 98% com o kNN. Descubra por que o resultado atual é muito menor. Ajuste o conjunto de dados ou k de tal forma que a acurácia se torne 98% e explique o que você fez e por quê.
R: A acurácia inicial não atingiu 98% porque o k-NN com k = 1 é altamente sensível a ruídos e a pontos próximos da fronteira entre classes, o que pode causar classificações incorretas. Além disso os dados não estavam normalizados, o que pode afetar a performance do k-NN, já que ele é baseado em distâncias.

Para melhorar o desempenho, foram testados diferentes valores de k. Observou-se que, ao aumentar o número de vizinhos considerados, o modelo se torna mais robusto, reduzindo o impacto de ruídos e decisões isoladas. E também foi realizada a normalização dos dados, o que ajudou a equilibrar a influência de cada característica.

Com valores de k ≥ 3, a acurácia já atingiu aproximadamente 98,33%, e para k ≥ 8, a acurácia chegou a 100%, classificando corretamente todas as amostras de teste.

Portanto, o ajuste do parâmetro k foi suficiente para aumentar significativamente a acurácia do modelo.
"""
