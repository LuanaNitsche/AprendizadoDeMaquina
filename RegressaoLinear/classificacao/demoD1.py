from knn import KNN
import numpy as np

knn = KNN()
grupoTrain, grupoTest, trainRots, testRots = knn.carregarDados(
    "grupoDados1.mat"
)

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


print("\n--- Teste variando k ---")
melhor_k = 0
melhor_acc = 0

for k in range(1, 11):
    rotuloPrevisto = knn.meuKnn(grupoTrain, trainRots, grupoTest, k)
    acc = np.mean(rotuloPrevisto == testRots)
    print(f"k={k} -> acurácia={acc:.4f}")

    if acc > melhor_acc:
        melhor_acc = acc
        melhor_k = k

print(f"\nMelhor k: {melhor_k} com acurácia: {melhor_acc:.4f}")

# ===== SOMENTE [1,2] =====
print("\n--- Atributos [0,1] variando k ---")

grupoTrain_01 = grupoTrain[:, [0, 1]]
grupoTest_01 = grupoTest[:, [0, 1]]

melhor_k_01 = 0
melhor_acc_01 = 0

for k in range(1, 9):
    rotuloPrevisto = knn.meuKnn(grupoTrain_01, trainRots, grupoTest_01, k)
    acc = np.mean(rotuloPrevisto == testRots)
    print(f"k={k} -> acurácia={acc:.4f}")

    if acc > melhor_acc_01:
        melhor_acc_01 = acc
        melhor_k_01 = k

print(f"Melhor k [0,1]: {melhor_k_01} com acurácia: {melhor_acc_01:.4f}")


print("\n--- Atributos [1,2] variando k ---")

grupoTrain_12 = grupoTrain[:, [1, 2]]
grupoTest_12 = grupoTest[:, [1, 2]]

melhor_k_12 = 0
melhor_acc_12 = 0

for k in range(1, 9):
    rotuloPrevisto = knn.meuKnn(grupoTrain_12, trainRots, grupoTest_12, k)
    acc = np.mean(rotuloPrevisto == testRots)
    print(f"k={k} -> acurácia={acc:.4f}")

    if acc > melhor_acc_12:
        melhor_acc_12 = acc
        melhor_k_12 = k

print(f"Melhor k [1,2]: {melhor_k_12} com acurácia: {melhor_acc_12:.4f}")


"""
Q1.1. Qual é a acurácia máxima que você consegue da classificação?
R: A acurácia máxima obtida foi de aproximadamente 96%, utilizando o algoritmo k-NN com dados previamente normalizados.

Ao testar diferentes valores de k, observou-se que valores menores, como k = 1, apresentaram melhor desempenho neste conjunto de dados. Isso ocorre porque os dados possuem boa separação entre as classes, permitindo que o vizinho mais próximo já seja suficiente para uma classificação correta na maioria dos casos.

Q1.2. É necessário ter todas as características (atributos) para obter a acurácia máxima para esta classificação?
    R: 
Não, não é necessário utilizar todas as características para realizar a classificação, porém isso impacta diretamente a acurácia do modelo.

Para analisar isso, foram realizados testes com diferentes subconjuntos de atributos e variação do parâmetro k.

Resultados:
Todos os atributos:
Melhor acurácia: 96% (k = 1 e k = 5)
Desempenho consistente para diferentes valores de k

Atributos [0,1]:
Melhor acurácia: 76% (k = 5 e k = 7)
Desempenho significativamente inferior
Mesmo ajustando k, não foi possível atingir alta acurácia

Atributos [1,2]:
Melhor acurácia: 96% (k = 7 e k = 8)
Desempenho equivalente ao uso de todos os atributos

Indica que essas duas dimensões já contêm informação suficiente para separar as classes
"""
