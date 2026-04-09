# DADOS 4

from knn import KNN
import numpy as np

knn = KNN()


# ----- Q4.1: aplicação direta (k=1, sem normalização) -----
grupoTrain, grupoTest, trainRots, testRots = knn.carregarDados(
    "grupoDados4.mat"
)
rotuloPrevisto = knn.meuKnn(grupoTrain, trainRots, grupoTest, k=1)
print("Q4.1 — k=1, todas as features, sem normalização:")
knn.acuracia(rotuloPrevisto, testRots)

# ----- Diagnóstico Q4.2: normalizar tudo ainda não chega a 92% -----
grupoTrainN, grupoTestN, trainRots, testRots = knn.carregarDados(
    "grupoDados4.mat", tipo="normalizar"
)
print("\nQ4.2 — todas as features normalizadas (varredura de k):")
for k in range(1, 11):
    pred = knn.meuKnn(grupoTrainN, trainRots, grupoTestN, k)
    acc = np.mean(pred == testRots)
    print(f"k={k:2d} -> acurácia={acc:.4f}")

# Ajuste: normalização + apenas dimensões 0 e 1 (as demais confundem o k-NN)
grupoTrain_sel = grupoTrain[:, [0, 1]]
grupoTest_sel = grupoTest[:, [0, 1]]
grupoTrain_selN, grupoTest_selN = knn.normalizacao(grupoTrain_sel, grupoTest_sel)

print("\nAjuste — features [0,1] + normalização (k que atinge ≥ 92%):")
for k in range(1, 11):
    pred = knn.meuKnn(grupoTrain_selN, trainRots, grupoTest_selN, k)
    acc = np.mean(pred == testRots)
    if acc >= 0.92:
        print(f"Primeiro k com acurácia ≥ 92%: k={k}")
        knn.acuracia(pred, testRots)
        knn.visualizaPontos(grupoTrain_selN, trainRots, 0, 1)
        break

# ===== Visualização [1,2] =====
grupoTrain_12 = grupoTrain[:, [1, 2]]
grupoTest_12 = grupoTest[:, [1, 2]]
grupoTrain_12N, grupoTest_12N = knn.normalizacao(grupoTrain_12, grupoTest_12)

print("\nVisualização — features [1,2]:")
knn.visualizaPontos(grupoTrain_12N, trainRots, 0, 1)

"""
Q4.1: Aplique seu algoritmo K-NN ao problema. Qual é a sua acurácia de classificação?
R: Com k = 1, todas as quatro características e sem normalização, a acurácia fica em
torno de 71,67%. O k-NN usa distância euclidiana; a segunda feature tem ordem de
grandeza muito maior que as outras, passando a dominar o cálculo e distorcendo os
vizinhos mais próximos.

Q4.2: A acurácia pode chegar a 92% com o K-NN. Descubra por que o resultado atual é
muito menor. Ajuste o conjunto de dados ou o valor de k de forma que a acurácia
atinja 92% e explique o que você fez e por quê. Observe que, desta vez, há mais de um
problema...
R: Há dois problemas principais. (1) Escala: sem normalização, uma feature com valores
muito grandes pesa desproporcionalmente na distância. Só normalizar já melhora, mas
com as quatro dimensões a acurácia ainda fica abaixo de 92% (no máximo cerca de 83%
nos testes com k). (2) Atributos que prejudicam o k-NN: as dimensões 2 e 3 adicionam
variação que não separa bem as classes neste espaço e acabam “puxando” vizinhos
errados quando usadas junto com as outras (mesma média ≈ 0.50 e desvio padrão ≈ 0.29 em todas as classes). Incluí-las no cálculo de distância adiciona variação aleatória
que prejudica a classificação. Ao manter apenas as features [0, 1],
aplicar normalização min-max (parâmetros do treino, aplicados ao teste) e usar por
exemplo k = 1 (ou outros k indicados no script), a acurácia sobe para cerca de 93%,
acima do alvo de 92%.

A análise visual dos dados também reforça esse comportamento.
Ao observar as dimensões [0,1], nota-se uma melhor separação entre as classes, o que favorece o desempenho do k-NN.
Por outro lado, ao considerar dimensões como [1,2], a separação entre as classes se torna menos evidente, indicando que nem todas as features contribuem positivamente para a classificação.

OBS: Os nomes das variáveis estão diferentes neste dataset:

grupoTest=testSet;
grupoTrain=trainSet;
testRots=testLabs;
trainRots=trainLabs;
"""
