
# def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k):
"""
%Para cada exemplo de teste

% Calcule a distância entre o exemplo de teste e os dados de treinamento

% Ordene as distâncias. A ordem iX de cada elemento ordenado é importante:

% [distOrdenada ind] = sort(...);

% O rótulo previsto corresponde ao rótulo do exemplo mais próximo (iX(1))"""
...


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



# DADOS 1
"""
Q1.1. Qual é a acurácia máxima que você consegue da classificação?

Q1.2. É necessário ter todas as características (atributos) para obter a acurácia máxima para esta classificação?
"""


"""
No código deve ter 
4 funções: dist, meuKnn, visualizaPontos, normalizacao
3 scripts: demoD1, demoD2, demoD3, demoD4 (todas com comentários do que foi feito), e responda as perguntas nos comentários de cada script.
"""
def dist(a, b):
    ...


def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k):
    ...


def visualizaPontos(dados, rotulos, d1, d2):
    ...


def normalizacao(dados):
    ...


# NOTE: 
# para cada exemplo de teste calcula a distancia de cada exemplo de treino (para cada linha ve a distancia com a matriz inteira de treino)
# precisa ordenar a distancia para dier quem é o menor (ou k distancias menores) distancai, mas mantendo o índice
