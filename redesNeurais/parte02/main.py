import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# backend não-interativo: salva figuras em arquivo sem precisar de display
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOMES_CLASSES = [
    "camisa/top", "calca", "casaco", "vestido", "jaqueta",
    "sandalia", "camiseta", "tenis", "bolsa", "bota",
]


def carregar_dados():
    # carrega o dataset Fashion-MNIST embutido no Keras (60.000 treino + 10.000 teste)
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    print(X_train_full.shape)  # (60000, 28, 28) — 60k imagens 28x28 pixels
    print(y_train_full.shape)  # (60000,) — rótulo inteiro de 0-9 para cada imagem
    print(X_test.shape)        # (10000, 28, 28)
    print(y_test.shape)        # (10000,)

    # separa os primeiros 5.000 exemplos para validação e normaliza para [0, 1]
    # dividir por 255 garante que todos os pixels fiquem no intervalo [0,1],
    # facilitando a convergência do gradiente
    X_valid = X_train_full[:5000] / 255.0
    X_train = X_train_full[5000:] / 255.0
    y_valid = y_train_full[:5000]
    y_train = y_train_full[5000:]

    # conjunto de teste também normalizado com o mesmo fator
    X_test = X_test / 255.0

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def visualizar_amostra(X_train, y_train, exemplo_n=5):
    print(NOMES_CLASSES[y_train[exemplo_n]])
    plt.imshow(X_train[exemplo_n], interpolation="nearest")
    plt.title(NOMES_CLASSES[y_train[exemplo_n]])
    plt.savefig("amostra.png")
    plt.close()


def criar_rna():
    # Sequential: pilha linear de camadas executadas em ordem
    nn = keras.models.Sequential()

    # Flatten: transforma a imagem 28x28 em um vetor de 784 valores — entrada da RNA
    nn.add(keras.layers.Flatten(input_shape=[28, 28]))

    # Dense(300, relu): camada oculta com 300 neurônios e ativação ReLU
    # ReLU evita o problema de vanishing gradient e é eficiente em camadas profundas
    nn.add(keras.layers.Dense(300, activation="relu"))

    # Dropout(0.1): desativa 10% dos neurônios aleatoriamente no treino
    # reduz overfitting ao evitar co-adaptação excessiva entre neurônios
    nn.add(keras.layers.Dropout(rate=0.1))

    # segunda camada oculta com 100 neurônios e ReLU
    nn.add(keras.layers.Dense(100, activation="relu"))

    # Dropout de 10% novamente para regularização
    nn.add(keras.layers.Dropout(rate=0.1))

    # camada de saída com 10 neurônios (1 por classe) e ativação softmax
    # softmax normaliza as saídas para probabilidades que somam 1,
    # ideal para classificação multiclasse (10 categorias de roupa)
    nn.add(keras.layers.Dense(10, activation="softmax"))

    nn.summary()
    return nn


def criar_cnn():
    # Sequential com arquitetura convolucional para extração hierárquica de features
    cnn = keras.models.Sequential([

        # InputLayer: define a forma da entrada — imagens 28x28 com 1 canal (tons de cinza)
        keras.layers.InputLayer(input_shape=(28, 28, 1)),

        # Conv2D(64 filtros, kernel 5x5, relu, padding same, glorot_uniform):
        # aplica 64 filtros de convolução 5x5 preservando dimensão espacial (padding=same)
        # glorot_uniform inicializa pesos para manter variância estável nas ativações
        keras.layers.Conv2D(64, 5, activation="relu", padding="same", kernel_initializer="glorot_uniform"),

        # MaxPool2D(2): reduz cada dimensão espacial pela metade (28x28 → 14x14)
        # retém a feature mais ativada em cada região 2x2, reduzindo parâmetros
        keras.layers.MaxPool2D(2),

        # duas Conv2D(128, 3x3): extraem features mais complexas a partir das anteriores
        # filtros menores (3x3) após pooling capturam padrões de maior abstração
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),

        # MaxPool2D(2): reduz de 14x14 para 7x7
        keras.layers.MaxPool2D(2),

        # duas Conv2D(258, 3x3): terceiro bloco convolucional com ainda mais filtros
        # mais filtros = maior capacidade de representar features abstratas
        keras.layers.Conv2D(258, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(258, 3, activation="relu", padding="same"),

        # MaxPool2D(2): reduz de 7x7 para 3x3
        keras.layers.MaxPool2D(2),

        # Flatten: transforma o volume 3x3x258 em vetor para entrada nas camadas densas
        keras.layers.Flatten(),

        # Dense(128, relu): camada densa para combinar as features extraídas
        keras.layers.Dense(128, activation="relu"),

        # Dropout(0.5): desativa 50% dos neurônios — regularização mais agressiva
        # justificada pela alta capacidade da rede, prevenindo overfitting
        keras.layers.Dropout(rate=0.5),

        # Dense(64, relu): segunda camada densa para refinamento da classificação
        keras.layers.Dense(64, activation="relu"),

        # Dropout(0.5): regularização na segunda camada densa
        keras.layers.Dropout(rate=0.5),

        # camada de saída com 10 neurônios e softmax — igual à RNA
        keras.layers.Dense(10, activation="softmax"),
    ])

    cnn.summary()
    return cnn


def compilar_modelo(modelo):
    # sparse_categorical_crossentropy: loss para classificação multiclasse
    # com rótulos inteiros (não one-hot encoded) — eficiente para 10 classes
    # sgd (Stochastic Gradient Descent): otimizador clássico com taxa padrão 0.01
    # accuracy: métrica de avaliação — percentual de acertos
    modelo.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"],
    )
    return modelo


def treinar_modelo(modelo, X_train, y_train, X_valid, y_valid, epochs):
    # fit: executa o treinamento por `epochs` épocas
    # validation_data permite monitorar overfitting comparando treino e validação a cada época
    history = modelo.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_valid, y_valid),
    )
    return history


def visualizar_historico(history, nome_arquivo):
    # plota as curvas de loss e accuracy de treino e validação por época
    # curvas próximas indicam boa generalização; gap grande indica overfitting
    pd.DataFrame(history.history).plot(figsize=(12, 8))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.title("Histórico de treinamento")
    plt.savefig(nome_arquivo)
    plt.close()
    print(f"Gráfico salvo em: {nome_arquivo}")


def avaliar_modelo(modelo, X_test, y_test):
    # evaluate: retorna loss e accuracy no conjunto de teste (dados nunca vistos)
    loss, accuracy = modelo.evaluate(X_test, y_test, verbose=1)
    print(f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
    return loss, accuracy


def prever(modelo, X_test, n=4):
    X_novo = X_test[:n]

    # predict: retorna vetor de probabilidades (softmax) para cada uma das 10 classes
    y_proba = modelo.predict(X_novo)
    print("Probabilidades por classe:\n", y_proba)

    # argmax seleciona o índice de maior probabilidade — a classe prevista
    y_pred = np.argmax(modelo.predict(X_novo), axis=-1)
    print("Índices previstos:", y_pred)
    print("Classes previstas:", np.array(NOMES_CLASSES)[y_pred])
    return y_pred


def preparar_dados_cnn(X):
    # CNN espera tensores 4D: (amostras, altura, largura, canais)
    # reshape adiciona a dimensão de canal (1 = tons de cinza)
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)


def main():
    X_train, y_train, X_valid, y_valid, X_test, y_test = carregar_dados()

    visualizar_amostra(X_train, y_train, exemplo_n=5)

    # --- RNA ---
    rna = criar_rna()
    rna = compilar_modelo(rna)
    history_rna = treinar_modelo(rna, X_train, y_train, X_valid, y_valid, epochs=5)
    visualizar_historico(history_rna, "historico_rna.png")
    avaliar_modelo(rna, X_test, y_test)
    y_pred_rna = prever(rna, X_test)
    print("Rótulos reais:", y_test[:4])
    print("Classes reais:", np.array(NOMES_CLASSES)[y_test[:4]])

    # --- CNN ---
    X_train_cnn = preparar_dados_cnn(X_train)
    X_valid_cnn = preparar_dados_cnn(X_valid)
    X_test_cnn = preparar_dados_cnn(X_test)

    cnn = criar_cnn()
    cnn = compilar_modelo(cnn)
    history_cnn = treinar_modelo(cnn, X_train_cnn, y_train, X_valid_cnn, y_valid, epochs=20)
    visualizar_historico(history_cnn, "historico_cnn.png")
    avaliar_modelo(cnn, X_test_cnn, y_test)


if __name__ == "__main__":
    main()
