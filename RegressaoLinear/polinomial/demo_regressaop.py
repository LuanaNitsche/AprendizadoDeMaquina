import pandas as pd


def retorna_dados(caminho:str):
    """
    Recebe um caminho para leitura com pandas
    """
    return pd.read_csv(caminho, header=None)


if __name__ == "__main__":
    data = retorna_dados(r'polinomial\data_preg.csv')
    print(data)
