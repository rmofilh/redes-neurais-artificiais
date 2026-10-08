"""Teste do PMC — Fase 3.

Retreina em memoria (mesma seed do treino => mesmos pesos) e avalia
as 20 amostras de data/conjunto_validacao.csv com MSE e tabela d vs o.
"""
import csv
from pathlib import Path

import numpy as np

from pmc.pmc_treino import executar_treinamento

BASE_DIR = Path(__file__).resolve().parent.parent
VALID_CSV = BASE_DIR / "data" / "conjunto_validacao.csv"


def carregar_validacao(path=VALID_CSV):
    ids, X, y = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ids.append(int(row["Amostra"]))
            X.append([float(row["x1"]), float(row["x2"]), float(row["x3"])])
            y.append(float(row["d"]))
    return ids, np.array(X), np.array(y)


def carregar_modelo():
    modelo, _ = executar_treinamento(log_cada=500)
    return modelo


def avaliar():
    ids, X, y = carregar_validacao()
    modelo = carregar_modelo()
    pred = modelo.prever(X)
    mse = float(((y - pred) ** 2).mean())
    return ids, y, pred, mse


def imprimir_tabela(ids, y, pred, mse):
    print(f"MSE validacao (20 padroes): {mse:.6f}")
    print("Amostra | desejado | obtido | erro")
    for i, d, o in zip(ids, y, pred):
        print(f"{i:3d} | {d:.4f} | {o:.4f} | {d - o:+.4f}")


if __name__ == "__main__":
    ids, y, pred, mse = avaliar()
    imprimir_tabela(ids, y, pred, mse)
