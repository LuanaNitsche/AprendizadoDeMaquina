import numpy as np
from knn import KNN

knn = KNN()

#comparando normalização, padronização e sem ajuste
for tipo in [None, "normalizar", "padronizar"]:
    print(f"\n--- Tipo: {tipo} ---")

    grupoTrain, grupoTest, trainRots, testRots = knn.carregarDados(
        "grupoDados2.mat", tipo=tipo
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
R: Aplicando o algoritmo k-NN sem qualquer pré-processamento dos dados, foi obtida uma acurácia de aproximadamente 68,33% para k = 1.
Esse resultado ocorre porque o algoritmo k-NN é baseado em distância, sendo altamente sensível à escala dos atributos. Como os dados não estavam normalizados ou padronizados, atributos com maior magnitude influenciaram mais o cálculo da distância euclidiana, prejudicando a classificação.

Q2.2: A acurácia inicial foi baixa devido a dois fatores principais:
Ausência de pré-processamento dos dados
Como o k-NN utiliza distância euclidiana, atributos em escalas diferentes impactam desproporcionalmente o cálculo da distância. Isso faz com que algumas características dominem a decisão do modelo.
Uso de k = 1
Com k = 1, o modelo é altamente sensível a ruídos e exemplos próximos da fronteira entre classes, o que pode levar a classificações incorretas.

Melhorias aplicadas

Para melhorar o desempenho, foram realizadas as seguintes modificações:

Normalização dos dados
Os atributos foram escalados para o intervalo [0, 1], garantindo que todas as features contribuam de forma equilibrada no cálculo da distância.

Padronização dos dados
Também foi testada a padronização, onde os dados passam a ter média 0 e desvio padrão 1, reduzindo o impacto de diferentes escalas.

Ajuste do parâmetro k
Foram testados valores de k de 1 a 8, observando que valores maiores tornam o modelo mais robusto a ruídos.

Resultados obtidos
Sem pré-processamento: acurácia máxima ≈ 73,33%
Com normalização:
k ≥ 3 → ≈ 98,33%
k = 8 → 100%
Com padronização:
Melhor resultado → 98,33% (k = 3)
"""
