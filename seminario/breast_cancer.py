import pandas as pd
import numpy as np
from knn import KNN

# =========================
# 1. INICIALIZAÇÃO
# =========================
knn = KNN()

# =========================
# 2. CARREGAR DATASET
# =========================
df = pd.read_csv("seminario/data.csv")

df = df.drop(columns=["id"])
df = df.dropna(axis=1, how="all")  # remove colunas 100% NaN
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values

# =========================
# 3. DIVISÃO TREINO / TESTE
# =========================
# usando split manual simples (como no seu exemplo original do .mat)
from sklearn.model_selection import train_test_split

grupoTrain, grupoTest, trainRots, testRots = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. COMPARAÇÃO: SEM / NORMALIZAÇÃO / PADRONIZAÇÃO
# =========================
melhores = {}

for tipo in [None, "normalizar", "padronizar"]:
    print(f"\n=========================")
    print(f"TIPO: {tipo}")
    print(f"=========================")

    if tipo is None:
        train = grupoTrain
        test = grupoTest
    elif tipo == "normalizar":
        train, test = knn.normalizacao(grupoTrain, grupoTest)
    elif tipo == "padronizar":
        train, test = knn.padronizacao(grupoTrain, grupoTest)

    melhor_k = 0
    melhor_acc = 0

    for k in range(1, 9):
        rotuloPrevisto = knn.meuKnn(train, trainRots, test, k)
        acc = np.mean(rotuloPrevisto == testRots)

        print(f"k={k} -> acurácia={acc:.4f}")

        if acc > melhor_acc:
            melhor_acc = acc
            melhor_k = k

    melhores[tipo] = (melhor_k, melhor_acc)

    print(f"\n>>> Melhor k ({tipo}): {melhor_k} -> {melhor_acc:.4f}")

knn.visualizaPontos(train, trainRots, 0, 1)

# =========================
# 5. SUBCONJUNTOS (COM PRÉ-PROCESSAMENTO = NORMALIZAÇÃO)
# =========================
print("\n=========================")
print("SUBCONJUNTOS (NORMALIZAÇÃO)")
print("=========================")

train_n, test_n = knn.normalizacao(grupoTrain, grupoTest)

# ---------- subset [0,1]
print("\n--- SUBSET [0,1] ---")
best_k_01 = 0
best_acc_01 = 0

for k in range(1, 9):
    r = knn.meuKnn(train_n[:, [0, 1]], trainRots, test_n[:, [0, 1]], k)
    acc = np.mean(r == testRots)

    print(f"k={k} -> acurácia={acc:.4f}")

    if acc > best_acc_01:
        best_acc_01 = acc
        best_k_01 = k

print(f">>> Melhor [0,1]: k={best_k_01} -> {best_acc_01:.4f}")


# ---------- subset [1,2]
print("\n--- SUBSET [1,2] ---")
best_k_12 = 0
best_acc_12 = 0

for k in range(1, 9):
    r = knn.meuKnn(train_n[:, [1, 2]], trainRots, test_n[:, [1, 2]], k)
    acc = np.mean(r == testRots)

    print(f"k={k} -> acurácia={acc:.4f}")

    if acc > best_acc_12:
        best_acc_12 = acc
        best_k_12 = k

print(f">>> Melhor [1,2]: k={best_k_12} -> {best_acc_12:.4f}")

knn.visualizaPontos(train_n[:, [0, 1]], trainRots, 0, 1)


# =========================
# 5. SUBCONJUNTOS (COM PRÉ-PROCESSAMENTO = PADRONIZAÇÃO)
# =========================
print("\n=========================")
print("SUBCONJUNTOS (PADRONIZAÇÃO)")
print("=========================")

train_p, test_p = knn.padronizacao(grupoTrain, grupoTest)

# ---------- subset [0,1]
print("\n--- SUBSET [0,1] ---")
best_k_01 = 0
best_acc_01 = 0

for k in range(1, 9):
    r = knn.meuKnn(train_p[:, [0, 1]], trainRots, test_p[:, [0, 1]], k)
    acc = np.mean(r == testRots)

    print(f"k={k} -> acurácia={acc:.4f}")

    if acc > best_acc_01:
        best_acc_01 = acc
        best_k_01 = k

print(f">>> Melhor [0,1]: k={best_k_01} -> {best_acc_01:.4f}")

# ---------- subset [1,2]
print("\n--- SUBSET [1,2] ---")
best_k_12 = 0
best_acc_12 = 0

for k in range(1, 9):
    r = knn.meuKnn(train_p[:, [1, 2]], trainRots, test_p[:, [1, 2]], k)
    acc = np.mean(r == testRots)

    print(f"k={k} -> acurácia={acc:.4f}")

    if acc > best_acc_12:
        best_acc_12 = acc
        best_k_12 = k

print(f">>> Melhor [1,2]: k={best_k_12} -> {best_acc_12:.4f}")


# =========================
# 5. SUBCONJUNTOS (COM MELHOR PRÉ-PROCESSAMENTO 23 e 29)
# =========================
print("\n=========================")
print("SUBCONJUNTOS (PADRONIZAÇÃO)")
print("=========================")

train_p2, test_p2 = knn.padronizacao(grupoTrain, grupoTest)

# ---------- subset [23,29]
print("\n--- SUBSET [23,29] ---")
best_k_2329 = 0
best_acc_2329 = 0

for k in range(1, 9):
    r = knn.meuKnn(train_p2[:, [23, 29]], trainRots, test_p2[:, [23, 29]], k)
    acc = np.mean(r == testRots)

    print(f"k={k} -> acurácia={acc:.4f}")

    if acc > best_acc_2329:
        best_acc_2329 = acc
        best_k_2329 = k

print(f">>> Melhor [23,29]: k={best_k_2329} -> {best_acc_2329:.4f}")

print("\n=========================")
print("SUBCONJUNTOS (NORMALIZAÇÃO)")
print("=========================")

train_n2, test_n2 = knn.normalizacao(grupoTrain, grupoTest)

# ---------- subset [23,29]
print("\n--- SUBSET [23,29] ---")
best_k_2329 = 0
best_acc_2329 = 0

for k in range(1, 9):
    r = knn.meuKnn(train_n2[:, [23, 29]], trainRots, test_n2[:, [23, 29]], k)
    acc = np.mean(r == testRots)

    print(f"k={k} -> acurácia={acc:.4f}")

    if acc > best_acc_2329:
        best_acc_2329 = acc
        best_k_2329 = k

print(f">>> Melhor [23,29]: k={best_k_2329} -> {best_acc_2329:.4f}")


# =========================
# 6. RESULTADO FINAL
# =========================
print("\n=========================")
print("RESULTADO FINAL")
print("=========================")

for tipo, (k, acc) in melhores.items():
    print(f"{tipo}: melhor k={k} -> {acc:.4f}")

print(f"\nSubset [0,1]: k={best_k_01} -> {best_acc_01:.4f}")
print(f"Subset [1,2]: k={best_k_12} -> {best_acc_12:.4f}")
print(f"Subset [23,29]: k={best_k_2329} -> {best_acc_2329:.4f}")