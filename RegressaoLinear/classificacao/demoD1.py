from knn import KNN
import numpy as np

knn = KNN()
grupoTrain, grupoTest, trainRots, testRots = knn.carregarDados(
    "grupoDados1.mat"
)
grupoTrain, grupoTest = knn.normalizacao(grupoTrain, grupoTest)

# Teste com k=1
rotuloPrevisto = knn.meuKnn(grupoTrain, trainRots, grupoTest, k=1)
acuracia = knn.acuracia(rotuloPrevisto, testRots)


# Teste com k=10
rotuloPrevisto2 = knn.meuKnn(grupoTrain, trainRots, grupoTest, k=10)
acuracia = knn.acuracia(rotuloPrevisto2, testRots)

knn.visualizaPontos(grupoTrain, trainRots, 0, 1)

# ===== SOMENTE [0,1] =====
grupoTrain_sel = grupoTrain[:, [0, 1]]
grupoTest_sel = grupoTest[:, [0, 1]]

rotuloPrevisto = knn.meuKnn(grupoTrain_sel, trainRots, grupoTest_sel, k=1)
print("Acurácia [0,1]:", knn.acuracia(rotuloPrevisto, testRots))

# ===== SOMENTE [1,2] =====
grupoTrain_sel = grupoTrain[:, [1, 2]]
grupoTest_sel = grupoTest[:, [1, 2]]

rotuloPrevisto = knn.meuKnn(grupoTrain_sel, trainRots, grupoTest_sel, k=1)
print("Acurácia [1,2]:", knn.acuracia(rotuloPrevisto, testRots))


"""
Q1.1. Qual é a acurácia máxima que você consegue da classificação?
R: A acurácia máxima que consegui foi de 96% utilizando k=1 e 94% utilizando k=10.

Q1.2. É necessário ter todas as características (atributos) para obter a acurácia máxima para esta classificação?
    R: Não, não é necessário utilizar todas as características para obter a acurácia máxima. Nos testes realizados, a maior acurácia (96%) foi obtida utilizando todos os atributos. Ao reduzir o número de atributos, houve queda na acurácia, como 68% com os atributos [0,1] e 90% com [1,2]. Isso indica que, para este conjunto de dados específico, todos os atributos contribuem para o melhor desempenho do modelo. No entanto, em outros casos, a remoção de atributos irrelevantes pode melhorar a acurácia.
"""
