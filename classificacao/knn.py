import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scipy


class KNN:
    """
    Classificador K-Nearest Neighbours.
    """

    def carregarDados(
        self, caminho: str, tipo: str = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carrega um arquivo .mat e retorna (grupoTrain, grupoTest, trainRots, testRots).

        Args:
            caminho (str): caminho para o arquivo .mat
            normalizar (bool): se True, aplica normalização min-max antes de retornar

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (grupoTrain, grupoTest, trainRots, testRots)
        """
        mat = scipy.loadmat(caminho)

        chaves = mat.keys()
        if "trainSet" in chaves:
            grupoTrain = mat["trainSet"]
            grupoTest = mat["testSet"]
            trainRots = mat["trainLabs"]
            testRots = mat["testLabs"]
        else:
            grupoTrain = mat["grupoTrain"]
            grupoTest = mat["grupoTest"]
            trainRots = mat["trainRots"]
            testRots = mat["testRots"]

        trainRots = trainRots.flatten()
        testRots = testRots.flatten()

        if tipo == "normalizar":
            grupoTrain, grupoTest = self.normalizacao(grupoTrain, grupoTest)
        elif tipo == "padronizar":
            grupoTrain, grupoTest = self.padronizacao(grupoTrain, grupoTest)

        return grupoTrain, grupoTest, trainRots, testRots

    def normalizacao(
        self, dadosTrain: np.ndarray, dadosTest: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalização min-max calculada sobre o conjunto de treino e
        aplicada a ambos os conjuntos (evita data leakage)

        Args:
            dadosTrain (np.ndarray): matriz de treino (n_treino x n_features)
            dadosTest (np.ndarray): matriz de teste (n_teste x n_features)
        Returns:
            tuple[np.ndarray, np.ndarray]: (dadosTrain_norm, dadosTest_norm)
        """
        min_vals = np.min(dadosTrain, axis=0)
        max_vals = np.max(dadosTrain, axis=0)

        dadosTrain_norm = (dadosTrain - min_vals) / (max_vals - min_vals)
        dadosTest_norm = (dadosTest - min_vals) / (max_vals - min_vals)

        return dadosTrain_norm, dadosTest_norm

    def padronizacao(
        self, dadosTrain: np.ndarray, dadosTest: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Padronização (z-score) usando média e desvio padrão do treino.

        Args:
            dadosTrain (np.ndarray)
            dadosTest (np.ndarray)

        Returns:
            tuple[np.ndarray, np.ndarray]
        """
        media = np.mean(dadosTrain, axis=0)
        desvio = np.std(dadosTrain, axis=0)

        dadosTrain_pad = (dadosTrain - media) / desvio
        dadosTest_pad = (dadosTest - media) / desvio

        return dadosTrain_pad, dadosTest_pad

    def dist(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Distância Euclidiana ao quadrado entre dois vetores

        Args:
            a (np.ndarray): vetor 1
            b (np.ndarray): vetor 2
        Returns:
            float: distância euclidiana ao quadrado
        """
        return np.sum((a - b) ** 2)

    def meuKnn(
        self,
        dadosTrain: np.ndarray,
        rotuloTrain: np.ndarray,
        dadosTeste: np.ndarray,
        k: int = 1,
    ) -> np.ndarray:
        """
        Classifica cada exemplo de dadosTeste usando os k vizinhos mais próximos.

        Args:
            dadosTrain (np.ndarray): matriz de treino  (n_treino  x n_features)
            rotuloTrain (np.ndarray): vetor de rótulos  (n_treino,)
            dadosTeste (np.ndarray): matriz de teste   (n_teste   x n_features)
            k (int): número de vizinhos a considerar (default=1)

        Returns:
            np.ndarray: vetor de rótulos previstos para os dadosTeste (n_teste,)
        """
        rotulos_previstos = []

        for teste in dadosTeste:
            distancias = np.array([self.dist(teste, treino) for treino in dadosTrain])

            ind = np.argsort(distancias)

            vizinhos = rotuloTrain[ind[:k]]

            valores, contagens = np.unique(vizinhos, return_counts=True)
            rotulos_previstos.append(valores[np.argmax(contagens)])

        return np.array(rotulos_previstos)

    def acuracia(self, rotuloPrevisto: np.ndarray, testRots: np.ndarray) -> float:
        """
        Calcula e imprime a acurácia, retornando o valor como float.

        Args:
            rotuloPrevisto (np.ndarray): vetor de rótulos previstos
            testRots (np.ndarray): vetor de rótulos verdadeiros

        Returns:
            float: acurácia (entre 0 e 1)
        """
        estaCorreto = rotuloPrevisto == testRots
        numCorreto = np.sum(estaCorreto)
        totalNum = len(testRots)
        acc = numCorreto / totalNum

        print(
            f"Acertos: {numCorreto}/{totalNum}  |  Acurácia: {acc:.4f} ({acc * 100:.2f}%)"
        )
        return acc

    def _getDadosRotulo(
        self, dados: np.ndarray, rotulos: np.ndarray, rotulo: int, indice: int
    ) -> list[float]:
        """
        Filtra os valores da coluna `indice` para a classe `rotulo`

        Args:
            dados (np.ndarray): matriz de dados
            rotulos (np.ndarray): vetor de rótulos
            rotulo (int): rótulo da classe a filtrar
            indice (int): índice da coluna a retornar

        Returns:
            list[float]: lista dos valores da coluna `indice` para os exemplos com rótulo igual a `rotulo`
        """
        return [
            dados[idx][indice] for idx in range(len(dados)) if rotulos[idx] == rotulo
        ]

    def visualizaPontos(
        self, dados: np.ndarray, rotulos: np.ndarray, d1: int, d2: int
    ) -> None:
        """
        Scatter plot das dimensões d1 e d2 para até 3 classes.

        Args:
            dados (np.ndarray): matriz de dados
            rotulos (np.ndarray): vetor de rótulos
            d1 (int): índice da dimensão a plotar no eixo x
            d2 (int): índice da dimensão a plotar no eixo y

        """
        classes = [
            (1, "red", "^", "Classe 1"),
            (2, "blue", "+", "Classe 2"),
            (3, "green", ".", "Classe 3"),
        ]

        _, ax = plt.subplots()
        for rotulo, cor, marcador, label in classes:
            x = self._getDadosRotulo(dados, rotulos, rotulo, d1)
            y = self._getDadosRotulo(dados, rotulos, rotulo, d2)
            if x:
                ax.scatter(x, y, c=cor, marker=marcador, label=label)

        ax.set_xlabel(f"Dimensão {d1}")
        ax.set_ylabel(f"Dimensão {d2}")
        ax.legend()
        plt.show()
