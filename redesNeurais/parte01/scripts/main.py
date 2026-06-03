import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential


def carregar_dados():
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    return X, y


def separar_dados(X, y):
    X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    return X_treinamento, X_teste, y_treinamento, y_teste


def criar_rede_neural():
    #3. Sequential é usado pois a RNA é uma pilha linear de camadas, onde cada camada tem exatamente uma entrada e uma saída
    rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape=(30,)),  # 30 features do dataset
        #2.b. 16 neurônios: valor entre a entrada (30) e a saída (1), potência de 2 para eficiência computacional. equilibra capacidade e risco de overfitting no dataset
        tf.keras.layers.Dense(units=16, activation="relu", kernel_initializer="random_uniform"),
        #2.d. sigmoid: mapeia a saída para [0,1], interpretável como probabilidade. ideal para classificação binária (maligno/benigno)
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ])
    return rede_neural


def compilar_rede_neural(rede_neural):
    # 6. Otimizadores ajustam os pesos da rede para minimizar a função de perda durante o treinamento.
    # Adam combina Momentum (média móvel dos gradientes para acelerar direções consistentes) e RMSprop
    # (taxa de aprendizado adaptativa por parâmetro), convergindo de forma eficiente sem precisar ajustar manualmente a learning rate.
    rede_neural.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    return rede_neural


def treinar_rede_neural(rede_neural, X_treinamento, y_treinamento):
    # 7. Com 426 amostras e batch_size=10, são usados ceil(426/10) = 43 batches por época.
    # 8. O treinamento ocorre por 100 épocas, totalizando 4300 atualizações de pesos.
    return rede_neural.fit(X_treinamento, y_treinamento, batch_size=10, epochs=100)


def prever(rede_neural, X_teste):
    # 9. O resultado é um valor entre 0 e 1 porque a camada de saída usa sigmoid, que mapeia qualquer valor real para [0,1],
    # representando a probabilidade de a amostra ser da classe positiva (maligno).
    return rede_neural.predict(X_teste)


def converter_para_binario(previsoes):
    # 10. Aplica threshold de 0.5: probabilidade >= 0.5 → True (maligno), < 0.5 → False (benigno).
    return previsoes > 0.5


def avaliar_rede_neural(rede_neural, X_teste, y_teste):
    # 9b. O evaluate retorna a loss e a binary_accuracy nos dados de teste, indicando o desempenho real da rede
    # em amostras que ela nunca viu durante o treinamento.
    return rede_neural.evaluate(X_teste, y_teste)


def criar_rede_neural_duas_camadas():
    # 10. Total de parâmetros com 2 camadas ocultas:
    # Dense1: 30×16 + 16 biases = 496
    # Dense2: 16×16 + 16 biases = 272
    # Saída:  16×1  +  1 bias   =  17
    # Total: 785 parâmetros treináveis.
    rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape=(30,)),
        tf.keras.layers.Dense(units=16, activation="relu", kernel_initializer="random_uniform"),
        tf.keras.layers.Dense(units=16, activation="relu", kernel_initializer="random_uniform"),
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ])
    return rede_neural


def criar_rede_kfold():
    # 13. O K-Fold divide o dataset em k=10 partes (folds). A cada iteração, 1 fold é usado como teste
    # e os 9 restantes como treino, repetindo 10 vezes até que todos os folds sirvam de teste.
    # O resultado final é a média das 10 acurácias, reduzindo o viés de uma única divisão treino/teste.
    # k.clear_session() é necessário para limpar os pesos da sessão anterior a cada fold, evitando contaminação.
    k.clear_session()
    rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape=(30,)),
        tf.keras.layers.Dense(units=16, activation="relu", kernel_initializer="random_uniform"),
        tf.keras.layers.Dense(units=16, activation="relu", kernel_initializer="random_uniform"),
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ])
    otimizador = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
    rede_neural.compile(optimizer=otimizador, loss="binary_crossentropy", metrics=["binary_accuracy"])
    return rede_neural


def executar_kfold(X, y):
    classificador = KerasClassifier(model=criar_rede_kfold, epochs=100, batch_size=10)
    resultados = cross_val_score(estimator=classificador, X=X, y=y, cv=10, scoring="accuracy")
    # 14. Desvio padrão baixo indica que o modelo é estável entre os folds (pouca variância).
    # Desvio padrão alto sugere que a acurácia depende muito de qual fold foi usado — sinal de instabilidade.
    print(f"Acurácia média: {resultados.mean():.4f}")
    print(f"Desvio padrão:  {resultados.std():.4f}")
    return resultados


def criar_rede_com_dropout():
    # 14. Dropout desativa aleatoriamente 20% dos neurônios a cada passo de treino, forçando a rede a aprender
    # representações redundantes e reduzindo a dependência de neurônios específicos — principal técnica contra overfitting.
    # Esperado: acurácia média semelhante ou ligeiramente menor, mas desvio padrão menor (modelo mais estável/generalizável).
    k.clear_session()
    rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape=(30,)),
        tf.keras.layers.Dense(units=16, activation="relu", kernel_initializer="random_uniform"),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=16, activation="relu", kernel_initializer="random_uniform"),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ])
    otimizador = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
    rede_neural.compile(optimizer=otimizador, loss="binary_crossentropy", metrics=["binary_accuracy"])
    return rede_neural


def executar_kfold_com_dropout(X, y):
    classificador = KerasClassifier(model=criar_rede_com_dropout, epochs=100, batch_size=10)
    resultados = cross_val_score(estimator=classificador, X=X, y=y, cv=10, scoring="accuracy")
    print(f"Acurácia média (dropout): {resultados.mean():.4f}")
    print(f"Desvio padrão  (dropout): {resultados.std():.4f}")
    return resultados


def compilar_rede_neural_otimizada(rede_neural):
    # 11. learning_rate=0.001: controla o tamanho do passo na atualização dos pesos — muito alto diverge, muito baixo aprende devagar.
    # clipvalue=0.5: limita os gradientes ao intervalo [-0.5, 0.5] para evitar o problema de exploding gradients.
    otimizador = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
    rede_neural.compile(
        optimizer=otimizador,
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    return rede_neural


def criar_rede_tuning(optimizer, loss, kernel_initializer, activation, neurons):
    # 15. A RNA foi configurada para receber hiperparâmetros como argumentos, permitindo que o GridSearchCV
    # substitua cada combinação automaticamente. O KerasClassifier adapta o modelo Keras à interface do sklearn,
    # e o GridSearchCV testa todas as combinações do param_grid com validação cruzada (cv=5),
    # retornando a combinação de maior acurácia média.
    k.clear_session()
    rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape=(30,)),
        tf.keras.layers.Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ])
    rede_neural.compile(optimizer=optimizer, loss=loss, metrics=["binary_accuracy"])
    return rede_neural


def executar_tuning(X, y):
    classificador = KerasClassifier(model=criar_rede_tuning)
    parametros = {
        "batch_size": [10, 30],
        "epochs": [50],
        "model__optimizer": ["adam"],
        "model__loss": ["binary_crossentropy"],
        "model__kernel_initializer": ["random_uniform", "normal"],
        "model__activation": ["relu"],
        "model__neurons": [16],
    }
    grid_search = GridSearchCV(estimator=classificador, param_grid=parametros, scoring="accuracy", cv=5)
    grid_search = grid_search.fit(X, y)
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    # 16. Sim, é possível melhorar a acurácia ampliando o espaço de busca (mais otimizadores, learning rates,
    # número de neurônios e camadas), usando RandomizedSearchCV para explorar combinações aleatórias de forma
    # mais eficiente, ou aplicando técnicas como BatchNormalization e early stopping.
    print(f"Melhor acurácia:     {grid_search.best_score_:.4f}")
    return grid_search


def main():
    X, y = carregar_dados()
    X_treinamento, X_teste, y_treinamento, y_teste = separar_dados(X, y)

    print(X_treinamento.shape, y_treinamento.shape)
    print(X_teste.shape, y_teste.shape)

    rede_neural = criar_rede_neural()
    rede_neural.summary()
    #4. O summary exibe, para cada camada: nome da camada; Output Shape (None, N) onde None é o batch size e N é a quantidade
    # de neurônios da camada; e Param #, que é o total de parâmetros treináveis (pesos + biases).
    # Para a camada densa oculta: 30 entradas × 16 neurônios + 16 biases = 496 parâmetros.
    # Para a camada de saída: 16 entradas × 1 neurônio + 1 bias = 17 parâmetros.
    # Total: 513 parâmetros treináveis.
    rede_neural = compilar_rede_neural(rede_neural)
    treinar_rede_neural(rede_neural, X_treinamento, y_treinamento)

    previsoes = prever(rede_neural, X_teste)
    previsoes = converter_para_binario(previsoes)
    print(previsoes)

    avaliar_rede_neural(rede_neural, X_teste, y_teste)

    # --- Camadas e Otimização da RNA ---
    # 12. Adicionar camadas aumenta a capacidade da rede, mas em datasets pequenos pode causar overfitting
    # (boa acurácia no treino, pior no teste) ou não trazer ganho significativo.
    rede_neural_2 = criar_rede_neural_duas_camadas()
    rede_neural_2.summary()
    rede_neural_2 = compilar_rede_neural_otimizada(rede_neural_2)
    treinar_rede_neural(rede_neural_2, X_treinamento, y_treinamento)

    prever(rede_neural_2, X_teste)
    avaliar_rede_neural(rede_neural_2, X_teste, y_teste)

    # --- K-Fold Cross Validation ---
    executar_kfold(X, y)

    # --- Overfitting e Dropout ---
    executar_kfold_com_dropout(X, y)

    # --- Tuning dos Hiperparâmetros ---
    executar_tuning(X, y)


if __name__ == "__main__":
    main()
