# Trabalho 4 – Parte 2: Redes Neurais e CNN para Classificação de Imagens

## Objetivo

Classificar imagens do dataset **Fashion-MNIST** utilizando duas arquiteturas: uma Rede Neural Artificial (RNA) densa e uma Rede Neural Convolucional (CNN). O trabalho compara o desempenho das duas abordagens na tarefa de reconhecimento de 10 categorias de roupas e acessórios.

---

## Dataset: Fashion-MNIST

O Fashion-MNIST é um dataset da Zalando composto por imagens de artigos de moda em tons de cinza. Cada imagem tem tamanho **28×28 pixels** e pertence a uma das 10 classes:

| Índice | Classe       |
|--------|--------------|
| 0      | Camisa/Top   |
| 1      | Calça        |
| 2      | Casaco       |
| 3      | Vestido      |
| 4      | Jaqueta      |
| 5      | Sandália     |
| 6      | Camiseta     |
| 7      | Tênis        |
| 8      | Bolsa        |
| 9      | Bota         |

**Divisão dos dados:**

| Conjunto    | Tamanho |
|-------------|---------|
| Treinamento | 55.000  |
| Validação   | 5.000   |
| Teste       | 10.000  |

Os primeiros 5.000 exemplos do conjunto de treino original são separados como validação. Todos os valores de pixel são normalizados de `[0, 255]` para `[0, 1]` dividindo por 255 — isso estabiliza a convergência do gradiente durante o treinamento.

---

## Pré-processamento

```
X_valid = X_train_full[:5000] / 255.0
X_train  = X_train_full[5000:] / 255.0
X_test   = X_test / 255.0
```

Para a CNN, os dados precisam de uma dimensão extra de canal, pois camadas Conv2D esperam tensores 4D `(amostras, altura, largura, canais)`:

```
X_train_cnn = X_train.reshape(..., 1)   # → (55000, 28, 28, 1)
```

---

## Modelo 1 – RNA (Rede Neural Artificial)

Arquitetura sequencial totalmente conectada (densa):

```
Flatten(28×28)   →  vetor de 784 valores
Dense(300, relu)
Dropout(0.1)
Dense(100, relu)
Dropout(0.1)
Dense(10, softmax)
```

### Detalhes das camadas

- **Flatten:** converte a imagem 2D (28×28) em um vetor 1D de 784 entradas, exigido pelas camadas densas.
- **Dense(300, relu):** primeira camada oculta com 300 neurônios. A ativação ReLU (`max(0, x)`) evita o problema de *vanishing gradient* e é computacionalmente eficiente.
- **Dropout(0.1):** desativa aleatoriamente 10% dos neurônios durante o treinamento, reduzindo *overfitting* por impedir co-adaptação excessiva entre neurônios.
- **Dense(100, relu):** segunda camada oculta com 100 neurônios, refinando as representações aprendidas.
- **Dropout(0.1):** mesma função de regularização aplicada à segunda camada oculta.
- **Dense(10, softmax):** camada de saída com um neurônio por classe. O *softmax* converte os logits em probabilidades que somam 1, indicando a confiança do modelo para cada categoria.

### Treinamento

- **Loss:** `sparse_categorical_crossentropy` — adequada para classificação multiclasse com rótulos inteiros (não *one-hot*).
- **Otimizador:** SGD (*Stochastic Gradient Descent*) com taxa de aprendizado padrão 0.01.
- **Métrica:** acurácia.
- **Épocas:** 5.

---

## Modelo 2 – CNN (Rede Neural Convolucional)

Arquitetura com três blocos convolucionais seguidos de camadas densas:

```
InputLayer(28, 28, 1)
Conv2D(64 filtros, 5×5, relu, padding=same)
MaxPool2D(2)                             → 14×14

Conv2D(128 filtros, 3×3, relu, padding=same)
Conv2D(128 filtros, 3×3, relu, padding=same)
MaxPool2D(2)                             → 7×7

Conv2D(258 filtros, 3×3, relu, padding=same)
Conv2D(258 filtros, 3×3, relu, padding=same)
MaxPool2D(2)                             → 3×3

Flatten()
Dense(128, relu)
Dropout(0.5)
Dense(64, relu)
Dropout(0.5)
Dense(10, softmax)
```

### Detalhes das camadas

**Bloco 1 – Extração de features de baixo nível:**
- **Conv2D(64, 5×5, relu, padding=same, glorot_uniform):** 64 filtros de tamanho 5×5 detectam bordas, texturas e padrões locais. `padding=same` mantém a dimensão espacial (28×28). O inicializador `glorot_uniform` distribui os pesos de forma a manter a variância estável nas ativações.
- **MaxPool2D(2):** reduz cada dimensão pela metade (28×28 → 14×14), mantendo apenas o valor máximo em cada janela 2×2 — diminui parâmetros e torna a rede menos sensível a translações.

**Bloco 2 – Features de nível intermediário:**
- **Conv2D(128, 3×3) × 2:** dois filtros consecutivos com 128 canais extraem padrões mais complexos (ex.: formas geométricas parciais). Filtros 3×3 após o primeiro *pooling* são suficientes para capturar padrões no espaço reduzido.
- **MaxPool2D(2):** 14×14 → 7×7.

**Bloco 3 – Features de alto nível:**
- **Conv2D(258, 3×3) × 2:** aumenta a capacidade de representação com 258 filtros, combinando features abstratas aprendidas nos blocos anteriores.
- **MaxPool2D(2):** 7×7 → 3×3.

**Classificação:**
- **Flatten:** transforma o volume 3×3×258 em vetor de entrada para as camadas densas.
- **Dense(128, relu) + Dropout(0.5):** combina as features extraídas. O *dropout* de 50% é mais agressivo que na RNA, justificado pela maior capacidade da CNN.
- **Dense(64, relu) + Dropout(0.5):** refinamento adicional antes da classificação final.
- **Dense(10, softmax):** saída probabilística para as 10 classes.

### Treinamento

- **Loss / Otimizador / Métrica:** idênticos à RNA.
- **Épocas:** 20 — mais épocas compensam a maior complexidade da rede.

---

## Regularização

| Técnica         | RNA       | CNN       | Efeito                                         |
|-----------------|-----------|-----------|------------------------------------------------|
| Dropout         | 10%       | 50%       | Previne *overfitting* desativando neurônios    |
| Normalização    | ÷ 255     | ÷ 255     | Estabiliza o gradiente e acelera convergência  |
| Validação split | 5.000 ex. | 5.000 ex. | Monitora generalização a cada época            |

---

## Avaliação

Ambos os modelos são avaliados com `model.evaluate()` no conjunto de teste, reportando *loss* e acurácia em dados nunca vistos durante o treinamento. O histórico de treinamento é visualizado em gráficos que mostram as curvas de *loss* e acurácia de treino e validação ao longo das épocas — um grande afastamento entre as curvas indica *overfitting*.

A CNN tende a superar a RNA nessa tarefa porque aproveita a estrutura espacial das imagens: camadas convolucionais compartilham pesos entre posições, tornando a rede invariante a pequenas translações e muito mais eficiente no uso de parâmetros do que uma rede totalmente conectada.

---

## Fluxo Resumido

```
Carregar Fashion-MNIST
        ↓
Normalizar pixels [0, 1]
        ↓
      ┌─────────────────────────────┐
      │  RNA (5 épocas)             │  CNN (20 épocas)
      │  Flatten → Dense → Softmax  │  Conv → Pool → Dense → Softmax
      └─────────────────────────────┘
        ↓
Avaliar em teste (loss + acurácia)
        ↓
Gerar gráfico do histórico de treinamento
        ↓
Prever classes de amostras novas
```
