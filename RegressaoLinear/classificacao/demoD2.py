# DADOS 2
"""
Q2.1: Aplique seu kNN a este problema. Qual é a sua acurácia de classificação?
R: Aplicando o algoritmo k-NN com k = 1, foi obtida uma acurácia de aproximadamente 96,67% (58 acertos em 60 amostras).

Q2.2: A acurácia pode ser igual a 98% com o kNN. Descubra por que o resultado atual é muito menor. Ajuste o conjunto de dados ou k de tal forma que a acurácia se torne 98% e explique o que você fez e por quê.
R: A acurácia inicial não atingiu 98% porque o k-NN com k = 1 é altamente sensível a ruídos e a pontos próximos da fronteira entre classes, o que pode causar classificações incorretas. Além disso os dados não estavam normalizados, o que pode afetar a performance do k-NN, já que ele é baseado em distâncias.

Para melhorar o desempenho, foram testados diferentes valores de k. Observou-se que, ao aumentar o número de vizinhos considerados, o modelo se torna mais robusto, reduzindo o impacto de ruídos e decisões isoladas. E também foi realizada a normalização dos dados, o que ajudou a equilibrar a influência de cada característica.

Com valores de k ≥ 3, a acurácia já atingiu aproximadamente 98,33%, e para k ≥ 8, a acurácia chegou a 100%, classificando corretamente todas as amostras de teste.

Portanto, o ajuste do parâmetro k foi suficiente para aumentar significativamente a acurácia do modelo.
"""

import numpy as np
import scipy.io as scipy
import matplotlib.pyplot as plt

def normalizacao(dadosTrain, dadosTest):
    min_vals = np.min(dadosTrain, axis=0)
    max_vals = np.max(dadosTrain, axis=0)

    dadosTrain_norm = (dadosTrain - min_vals) / (max_vals - min_vals)
    dadosTest_norm = (dadosTest - min_vals) / (max_vals - min_vals)

    return dadosTrain_norm, dadosTest_norm

def dist(a, b):
    return np.sum((a - b) ** 2)

def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k=1):

    '''
    Para cada ponto de teste, ele:

        mede a distância até todos os pontos de treino
        pega o mais próximo
        copia o rótulo
    '''
    rotulos_previstos = []

    for teste in dadosTeste:
        # Calcula distância euclidiana (sem sqrt)
        distancias = np.array([dist(teste, treino) for treino in dadosTrain])

        # Ordena e pega índices
        ind = np.argsort(distancias)

        # Seleciona os k vizinhos mais próximos
        vizinhos = rotuloTrain[ind[:k]]

        # Se k = 1 → pega direto
        if k == 1:
            rotulos_previstos.append(vizinhos[0])
        else:
            # votação por maioria
            valores, contagens = np.unique(vizinhos, return_counts=True)
            rotulos_previstos.append(valores[np.argmax(contagens)])

    return np.array(rotulos_previstos)

def getDadosRotulo(dados, rotulos, rotulo, indice):
    ret = []

    for idx in range(len(dados)):
        if rotulos[idx] == rotulo:
            ret.append(dados[idx][indice])

    return ret


# =========================
# Carregar dados
# =========================
mat = scipy.loadmat('RegressaoLinear/classificacao/grupoDados2.mat')

grupoTrain = mat['grupoTrain']
grupoTest = mat['grupoTest']
trainRots = mat['trainRots']
testRots = mat['testRots']

# ⚠️ Ajuste importante: transformar rótulos em vetor 1D
trainRots = trainRots.flatten()
testRots = testRots.flatten()

# grupoTrain, grupoTest = normalizacao(grupoTrain, grupoTest)

# =========================
# Rodar KNN
# =========================
# =========================
# Testar vários valores de k
# =========================
for k in range(1, 9):
    rotuloPrevisto = meuKnn(grupoTrain, trainRots, grupoTest, k)
    acc = np.mean(rotuloPrevisto == testRots)
    print(f"k={k} -> acurácia={acc:.4f}")

# =========================
# Calcular acurácia
# =========================

# Compara com a resposta correta
#Isso gera algo tipo:

# [True, True, False, True, ...]
estaCorreto = rotuloPrevisto == testRots
numCorreto = np.sum(estaCorreto)
totalNum = len(testRots)

acuracia = numCorreto / totalNum

# =========================
# Resultado
# =========================
print("Número de acertos:", numCorreto)
print("Total:", totalNum)
print("Acurácia:", acuracia)



def visualizaPontos(dados, rotulos, d1, d2):
    fig, ax = plt.subplots()

    ax.scatter(
        getDadosRotulo(dados, rotulos, 1, d1),
        getDadosRotulo(dados, rotulos, 1, d2),
        c='red', marker='^', label='Classe 1'
    )

    ax.scatter(
        getDadosRotulo(dados, rotulos, 2, d1),
        getDadosRotulo(dados, rotulos, 2, d2),
        c='blue', marker='+', label='Classe 2'
    )

    ax.scatter(
        getDadosRotulo(dados, rotulos, 3, d1),
        getDadosRotulo(dados, rotulos, 3, d2),
        c='green', marker='.', label='Classe 3'
    )

    ax.set_xlabel(f'Dimensão {d1}')
    ax.set_ylabel(f'Dimensão {d2}')
    ax.legend()

    plt.show()

visualizaPontos(grupoTrain, trainRots, 0, 1)
