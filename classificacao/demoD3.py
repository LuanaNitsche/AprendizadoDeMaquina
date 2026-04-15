from knn import KNN
import numpy as np

knn = KNN()

grupoTrain, grupoTest, trainRots, testRots = knn.carregarDados(
    "grupoDados3.mat"
)

rotuloPrevisto = knn.meuKnn(grupoTrain, trainRots, grupoTest, k=1)
print("Q3.1 — k=1, sem normalização:")
knn.acuracia(rotuloPrevisto, testRots)

print("\nQ3.2 — diferentes k com normalização:")
grupoTrainN, grupoTestN, trainRots, testRots = knn.carregarDados(
    "grupoDados3.mat", tipo="normalizar"
)

for k in range(1, 11):
    rotuloPrevisto = knn.meuKnn(grupoTrainN, trainRots, grupoTestN, k)
    acc = np.mean(rotuloPrevisto == testRots)
    print(f"[Normalização] k={k} -> acurácia={acc:.4f}")

for k in range(1, 11):
    rotuloPrevisto = knn.meuKnn(grupoTrainN, trainRots, grupoTestN, k)
    acc = np.mean(rotuloPrevisto == testRots)
    if acc >= 0.92:
        print(f"\nPrimeiro k com acurácia >= 92%: k={k}")
        knn.acuracia(rotuloPrevisto, testRots)
        knn.visualizaPontos(grupoTrainN, trainRots, 0, 1)
        break

print("\nQ3.3 — diferentes k com padronização:")
grupoTrainP, grupoTestP, trainRots, testRots = knn.carregarDados(
    "RegressaoLinear/classificacao/grupoDados3.mat", tipo="padronizar"
)

for k in range(1, 11):
    rotuloPrevisto = knn.meuKnn(grupoTrainP, trainRots, grupoTestP, k)
    acc = np.mean(rotuloPrevisto == testRots)
    print(f"[Padronização] k={k} -> acurácia={acc:.4f}")

"""
Q3.1: Aplique o kNN ao problema usando k = 1. Qual é a acurácia na classificação?
R: Com k = 1 e sem normalização, a acurácia obtida foi inferior aproximadamente 62%, 
bem inferior 92%. O kNN é sensível a escala das features, e o conjunto de dados 3 
possui atributos com magnitudes muito diferentes, o que distorce o cálculo de distância 
euclidiana e prejudica a classificação.

Q3.2: A acurácia pode ser igual a 92% com o kNN. Descubra por que o resultado atual
é muito menor. Ajuste o conjunto de dados ou k de tal forma que a acurácia se torne
92% e explique o que você fez e por quê.
R: 
A baixa acurácia inicial está diretamente relacionada à ausência de pré-processamento dos dados. E também ao valor de k = 1, que é muito sensível a ruídos e exemplos próximos da fronteira entre classes.

Para resolver esse problema, foram aplicadas duas técnicas:

Normalização

Padronização

Resultados observados
Sem pré-processamento: ~58%
Com normalização: acurácia ≥ 92% para determinados valores de k
Com padronização: desempenho semelhante ou próximo ao da normalização

Além disso:

Valores maiores de k tornam o modelo mais robusto a ruídos
Foi possível atingir a acurácia desejada (≥ 92%) após aplicar o pré-processamento e ajustar k
"""
