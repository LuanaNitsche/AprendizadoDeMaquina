# Roteiro do Vídeo — Trabalho 4 RNA (Parte 1 e Parte 2)

Duração alvo: **7-8 minutos**, dividido em 3 blocos de **~2:30 min** cada.
Sugestão: cada pessoa compartilha a tela já com o notebook correspondente aberto na seção que vai explicar.

---

## PESSOA 1 — Introdução + Parte 1: Construção da Rede Neural (~2:30)

> Abrir: `parte01/scripts/TRABALHO4_RNA_PARTE1_RESOLUCAO.ipynb`

"Olá! Neste vídeo vamos apresentar o Trabalho 4 de Redes Neurais Artificiais, dividido em duas partes: a primeira é uma rede neural para diagnóstico de câncer de mama, e a segunda compara uma RNA simples com uma CNN para classificação de imagens. Eu vou explicar a primeira parte.

Usamos a base **Breast Cancer Wisconsin (Diagnostic)**, disponível no `sklearn`. Ela tem 569 amostras, cada uma com 30 características numéricas extraídas de exames — como raio, textura e perímetro de núcleos celulares — e o rótulo indica se o tumor é maligno ou benigno. Dividimos os dados em 75% para treino e 25% para teste, totalizando 426 amostras de treino e 143 de teste.

Para a arquitetura, criamos uma rede `Sequential`, ou seja, uma pilha linear de camadas, onde a saída de uma é a entrada da próxima. A camada de entrada recebe os 30 atributos. Em seguida temos uma camada oculta densa com 16 neurônios, ativação `ReLU` e inicializador `random_uniform` — 16 foi escolhido por ficar entre o número de entradas e saídas, equilibrando capacidade de aprendizado e risco de overfitting. A camada de saída tem 1 neurônio com ativação `sigmoid`, que transforma o resultado em uma probabilidade entre 0 e 1, ideal para problemas binários.

Ao todo, essa rede tem **513 parâmetros treináveis** — pesos e bias somados.

Por fim, compilamos o modelo com o otimizador **Adam**, que combina momentum com taxa de aprendizado adaptativa, a função de perda `binary_crossentropy`, adequada para classificação binária, e a métrica `binary_accuracy` para acompanhar o desempenho durante o treino."

---

## PESSOA 2 — Parte 1: Treinamento, Validação e Otimização (~2:30)

> Continuar no mesmo notebook, seções de treinamento, K-Fold, Dropout e GridSearchCV

"Agora vou explicar o treinamento e as etapas de validação e otimização do modelo.

Treinamos a rede por **100 épocas**, com `batch_size=10`. Como temos 426 amostras de treino, cada época processa cerca de 43 lotes — isso significa mais de 4 mil atualizações de pesos ao final do treinamento. Depois, usamos `predict` para gerar as probabilidades e aplicamos um limiar de 0,5: valores acima disso são classificados como malignos. Na avaliação com os dados de teste, o modelo atingiu cerca de **95% de acurácia**.

Para melhorar a robustez da avaliação, fizemos duas coisas. Primeiro, adicionamos uma **segunda camada oculta** com mais 16 neurônios, totalizando 785 parâmetros, e ajustamos o otimizador Adam com `learning_rate=0.001` e `clipvalue=0.5`, que limita os gradientes para evitar que eles explodam durante o treino.

Segundo, aplicamos **validação cruzada K-Fold com 10 divisões**: o conjunto de dados é dividido em 10 partes, e a cada rodada uma parte vira teste e as outras nove viram treino. No final, calculamos a média e o desvio padrão das 10 acurácias — isso dá uma visão muito mais confiável do desempenho real do que um único treino e teste.

Para combater o **overfitting**, usamos **Dropout de 20%** nas camadas ocultas, que desativa aleatoriamente parte dos neurônios a cada passo de treino, forçando a rede a generalizar melhor.

Por fim, usamos o **GridSearchCV** para testar automaticamente combinações de hiperparâmetros — como tamanho do batch, inicializador de pesos e número de neurônios — e encontrar a configuração com melhor acurácia média via validação cruzada."

---

## PESSOA 3 — Parte 2: RNA vs CNN no Fashion-MNIST + Conclusão (~2:30 a 3:00)

> Abrir: `parte02/Trabalho_4_RNA_e_CNN_PARTE2.ipynb`

"Na segunda parte do trabalho, mudamos de problema: agora é classificação de imagens usando o dataset **Fashion-MNIST**, que tem 60 mil imagens de treino e 10 mil de teste, em escala de cinza, 28x28 pixels, divididas em 10 categorias de roupas e acessórios, como camiseta, calça, tênis e bolsa.

Antes de treinar, normalizamos os pixels dividindo por 255, deixando os valores entre 0 e 1, o que ajuda a rede a convergir mais rápido.

Primeiro, construímos uma **RNA simples**: uma camada `Flatten` que transforma a imagem 28x28 em um vetor de 784 valores, depois uma camada densa de 300 neurônios com `ReLU`, um Dropout de 10%, outra densa de 100 neurônios com `ReLU`, mais um Dropout de 10%, e a camada de saída com 10 neurônios e ativação `softmax` — que gera uma probabilidade para cada uma das 10 classes. Treinamos com o otimizador `sgd` e `sparse_categorical_crossentropy` por 5 épocas.

Depois, construímos uma **CNN (Rede Neural Convolucional)**, mais adequada para imagens. Ela tem três blocos de camadas convolucionais — `Conv2D` com 64, depois 128 e depois 258 filtros — intercaladas com camadas de `MaxPooling`, que reduzem a dimensão espacial mantendo as características mais importantes. Depois do `Flatten`, temos camadas densas de 128 e 64 neurônios com Dropout de 50% para evitar overfitting, e a saída também com `softmax` para as 10 classes. Essa CNN foi treinada por 20 épocas.

Comparando os dois modelos, a CNN apresenta desempenho superior, pois as camadas convolucionais conseguem capturar padrões espaciais — como bordas, texturas e formas — que a RNA densa, ao apenas 'achatar' a imagem em um vetor, acaba perdendo.

Para concluir: na Parte 1 vimos como uma RNA simples já resolve bem um problema de classificação binária com dados tabulares, e técnicas como K-Fold, Dropout e GridSearchCV ajudam a validar e otimizar o modelo. Na Parte 2, vimos que para imagens, CNNs são mais eficazes que RNAs densas, graças às camadas convolucionais e de pooling. Obrigado(a) por assistir!"

---

## Dicas de gravação
- Cada parte tem ~300-330 palavras, ritmo médio de fala (~130-140 palavras/min) → ~2:20-2:30 por pessoa.
- Testem a leitura em voz alta antes de gravar para ajustar o tempo.
- Podem cortar trechos secundários (ex.: detalhes de parâmetros) se precisarem encurtar.
