from knn import KNN

knn = KNN()
grupoTrain, grupoTest, trainRots, testRots = knn.carregarDados(
    "classificacao/grupoDados1.mat"
)
grupoTrain, grupoTest = knn.normalizacao(grupoTrain, grupoTest)

# Teste com k=1
rotuloPrevisto = knn.meuKnn(grupoTrain, trainRots, grupoTest, k=1)
acuracia = knn.acuracia(rotuloPrevisto, testRots)


# Teste com k=10
rotuloPrevisto2 = knn.meuKnn(grupoTrain, trainRots, grupoTest, k=10)
acuracia = knn.acuracia(rotuloPrevisto2, testRots)

knn.visualizaPontos(grupoTrain, trainRots, 0, 1)


"""
Q1.1. Qual é a acurácia máxima que você consegue da classificação?
R: A acurácia máxima que consegui foi de 96% utilizando k=1 e 94% utilizando k=10.

Q1.2. É necessário ter todas as características (atributos) para obter a acurácia máxima para esta classificação?
    R: Não necessariamente. Algumas características podem ser irrelevantes ou redundantes. Remover atributos pode até melhorar o desempenho, especialmente no KNN, que é sensível à dimensionalidade e à escala dos dados.
"""
