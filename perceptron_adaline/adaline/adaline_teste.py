"""Teste do Adaline — item 4 do trabalho.

Reutiliza os 5 modelos treinados em adaline_treino.py (mesmos seeds)
e classifica as 15 amostras de perceptron_adaline/data/adaline_teste.csv.
"""
import csv
from pathlib import Path

import numpy as np

from perceptron_adaline.adaline.adaline_treino import executar_treinamentos

BASE_DIR = Path(__file__).resolve().parent.parent
TESTE_CSV = BASE_DIR / "data" / "adaline_teste.csv"


def carregar_teste(path=TESTE_CSV):
    ids, X = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["Amostra"]))
            X.append([float(row["x1"]), float(row["x2"]), float(row["x3"]), float(row["x4"])])
    return ids, np.array(X)


def classificar():
    resultados = executar_treinamentos()
    ids, Xt = carregar_teste()
    matriz = []
    for r in resultados:
        pred = r["modelo"].prever(Xt).ravel().astype(int).tolist()
        matriz.append(pred)
    # matriz: 5 (T) x 15 (amostras) -> transpor para 15 x 5
    tabela = np.array(matriz).T.tolist()
    return ids, Xt, tabela


def imprimir_tabela(ids, Xt, tabela):
    print("Amostra | x1 x2 x3 x4 | y(T1) y(T2) y(T3) y(T4) y(T5)")
    for i, xt, linha in zip(ids, Xt, tabela):
        print(f"{i}: [{xt[0]:.4f} {xt[1]:.4f} {xt[2]:.4f} {xt[3]:.4f}] -> {linha}")


if __name__ == "__main__":
    ids, Xt, tabela = classificar()
    imprimir_tabela(ids, Xt, tabela)
