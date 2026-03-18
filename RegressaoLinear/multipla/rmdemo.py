import math
import re
import matplotlib.pyplot as plt


def calcula_beta_multipla(matriz:list[list]):
    """
    Calcula o beta da 
    """
    ...


def regressao_multipla(matriz:list[list]):
    """
    Calcula os coef beta0 e beta1 da reta de regressao multipal
    Beta:
    - faz a transposta da matriz vezes a propria matriz
    - faz o inverso da matriz resultante
    - multiplica o resultado pela matriz transposta
    - multiplica por y

    y = matriz vezes beta
    
    """
    for vetor in matriz:
        for var in vetor:
            