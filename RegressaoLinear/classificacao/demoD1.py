import numpy as np
import scipy.io as scipy
import matplotlib.pyplot as plt

# def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k):
"""
%Para cada exemplo de teste

% Calcule a distância entre o exemplo de teste e os dados de treinamento

% Ordene as distâncias. A ordem iX de cada elemento ordenado é importante:

% [distOrdenada ind] = sort(...);

% O rótulo previsto corresponde ao rótulo do exemplo mais próximo (iX(1))"""
...

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
        # Calcula distância euclidiana
        diferenca = dadosTrain - teste
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


"""
Para testar se você implementou a função corretamente, baixe o arquivo grupoDados1.mat - cada arquivo .mat contém 4 variáveis que são: grupoTest, grupoTrain, testRots, trainRots. Para baixar os arquivos .mat no Python você pode fazer o seguinte:

import scipy.io as scipy

mat = scipy.loadmat('grupoDados1.mat')

grupoTest = mat['grupoTest']

"""


"""
Então, verifique quantas classes foram previstas corretamente, isto é chamado de acurácia (accuracy):

rotuloPrevisto = meuKnn(grupoTrain, trainRots, grupoTest, 1);

estaCorreto = rotuloPrevisto == testRots;

numCorreto = sum(estaCorreto);

totalNum = length(testRots);

acurácia = numCorreto / totalNum
"""

# =========================
# Carregar dados
# =========================
mat = scipy.loadmat('RegressaoLinear/classificacao/grupoDados1.mat')

grupoTrain = mat['grupoTrain']
grupoTest = mat['grupoTest']
trainRots = mat['trainRots']
testRots = mat['testRots']

# ⚠️ Ajuste importante: transformar rótulos em vetor 1D
trainRots = trainRots.flatten()
testRots = testRots.flatten()

grupoTrain, grupoTest = normalizacao(grupoTrain, grupoTest)

# =========================
# Rodar KNN
# =========================
rotuloPrevisto = meuKnn(grupoTrain, trainRots, grupoTest, k=1)

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


"""
A acurácia deve ser de 96%. Agora, vamos estender a função a um classificador k-NN:

def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k)

%Para cada exemplo de teste

% Calcule a distância entre o exemplo de teste e os dados de treinamento

% Ordene as distâncias. A ordem iX de cada elemento ordenado é importante:

% [distOrdenada ind] = sort(...);

% Obter os rótulos correspondentes aos exemplos mais próximos k

% Agora, a moda dos rótulos correspondentes são os rótulos previstos (você pode usar a função mode).
"""

rotuloPrevisto2= meuKnn(grupoTrain, trainRots, grupoTest, k=10)

estaCorreto = rotuloPrevisto2 == testRots
numCorreto = np.sum(estaCorreto)
totalNum = len(testRots)

acuracia = numCorreto / totalNum

print("Número de acertos:", numCorreto)
print("Total:", totalNum)
print("Acurácia:", acuracia)

"""
Teste novamente no conjunto de dados 1 (grupoDados1.mat) e utilize k = 10 para uma acurácia igual a 94%.

É sempre bom visualizar graficamente seus dados. Para fazer isso, crie a seguinte função:

import matplotlib.pyplot as plt

 def getDadosRotulo(dados, rotulos, rotulo, indice):

    ret = []

    for idx in range(0, len(dados)):

        if(rotulos[idx] == rotulo):

            ret.append(dados[idx][indice])        

    return ret
"""


"""
def visualizaPontos(dados, rotulos, d1, d2):

    fig, ax = plt.subplots() 

    ax.scatter(getDadosRotulo(dados, rotulos, 1, d1), getDadosRotulo(dados, rotulos, 1, d2), c='red' , marker='^')

    ax.scatter(getDadosRotulo(dados, rotulos, 2, d1), getDadosRotulo(dados, rotulos, 2, d2), c='blue' , marker='+')

    ax.scatter(getDadosRotulo(dados, rotulos, 3, d1), getDadosRotulo(dados, rotulos, 3, d2), c='green', marker='.'   

    plt.show()
"""

def getDadosRotulo(dados, rotulos, rotulo, indice):
    ret = []

    for idx in range(len(dados)):
        if rotulos[idx] == rotulo:
            ret.append(dados[idx][indice])

    return ret


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



# DADOS 1
"""
Q1.1. Qual é a acurácia máxima que você consegue da classificação?
R: A acurácia máxima que consegui foi de 96% utilizando k=1 e 94% utilizando k=10.

Q1.2. É necessário ter todas as características (atributos) para obter a acurácia máxima para esta classificação?
    R: Não necessariamente. Algumas características podem ser irrelevantes ou redundantes. Remover atributos pode até melhorar o desempenho, especialmente no KNN, que é sensível à dimensionalidade e à escala dos dados.
"""


"""
No código deve ter 
4 funções: dist, meuKnn, visualizaPontos, normalizacao
3 scripts: demoD1, demoD2, demoD3, demoD4 (todas com comentários do que foi feito), e responda as perguntas nos comentários de cada script.
"""


# NOTE: 
# para cada exemplo de teste calcula a distancia de cada exemplo de treino (para cada linha ve a distancia com a matriz inteira de treino)
# precisa ordenar a distancia para dier quem é o menor (ou k distancias menores) distancai, mas mantendo o índice
