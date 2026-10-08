"""Teste PMC1 — item 5 (erro relativo medio % e variancia nos 20 de validacao).

Retreina os 5 modelos em memoria (seeds fixas => mesmos pesos do treino),
sem nenhum .npz em disco.
"""
import csv
from pathlib import Path

import numpy as np

from pmc1.pmc1_treino import executar_5_treinamentos

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


def avaliar():
    ids, X, y = carregar_validacao()
    resultados = executar_5_treinamentos()
    tabela, resumes = [], []
    for r in resultados:
        pred = r["modelo"].prever(X)
        er = np.abs(y - pred) / np.abs(y) * 100.0
        resumes.append({"T": r["T"], "pred": pred,
                        "erm": float(er.mean()), "var": float(er.var())})
        tabela.append(pred)
    return ids, y, np.array(tabela).T, resumes


def imprimir(ids, y, tabela, resumes):
    print("Amostra | d | " + " ".join(f"y(T{k})" for k in range(1, 6)))
    for i, d, linha in zip(ids, y, tabela):
        print(f"{i:3d} | {d:.4f} | " + " ".join(f"{v:.4f}" for v in linha))
    for r in resumes:
        print(f"{r['T']}: ERM={r['erm']:.3f}% var={r['var']:.4f}")


if __name__ == "__main__":
    ids, y, tabela, resumes = avaliar()
    imprimir(ids, y, tabela, resumes)
