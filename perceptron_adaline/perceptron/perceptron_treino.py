"""Treino do perceptron — itens 1 e 2 do trabalho.

- Regra supervisionada com (d - y): w += eta*(d-y)*x, w0 += eta*(d-y)*(-1)
- eta = 0.01, x0 = -1 fixo, ativacao bipolar
- 5 treinamentos com seeds 10,20,30,40,50 e init U[0,1]
- Gera tabela do item 2 (pesos iniciais/finais + epocas)
"""
import csv
from pathlib import Path

import numpy as np

from perceptron_adaline.perceptron.perceptron_modelo import Perceptron

SEEDS = [10, 20, 30, 40, 50]
TAXA = 0.01
MAX_EPOCAS = 1000
BASE_DIR = Path(__file__).resolve().parent.parent
TREINO_CSV = BASE_DIR / "data" / "oleo_treinamento.csv"


def carregar_treino(path=TREINO_CSV):
    X, y = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row["x1"]), float(row["x2"]), float(row["x3"])])
            y.append(float(row["d"]))
    return np.array(X), np.array(y)


def executar_treinamentos():
    X, y = carregar_treino()
    resultados = []
    for i, seed in enumerate(SEEDS, start=1):
        np.random.seed(seed)
        modelo = Perceptron(taxa_aprendizado=TAXA, max_epocas=MAX_EPOCAS)
        # Inicializacao fora da classe (b) fora), valores entre 0 e 1
        modelo.pesos = np.random.uniform(0, 1, X.shape[1])
        modelo.bias = float(np.random.uniform(0, 1))
        n_epocas, pesos_finais, bias_final = modelo.treinar(X, y)

        pred = modelo.prever(X).ravel()
        acc = float((pred == y).mean())

        resultados.append(
            {
                "T": f"T{i}",
                "seed": seed,
                "w0i": modelo.bias_inicial,
                "w1i": modelo.pesos_iniciais[0],
                "w2i": modelo.pesos_iniciais[1],
                "w3i": modelo.pesos_iniciais[2],
                "w0f": float(bias_final),
                "w1f": float(pesos_finais[0]),
                "w2f": float(pesos_finais[1]),
                "w3f": float(pesos_finais[2]),
                "epocas": n_epocas,
                "acc": acc,
                "modelo": modelo,
            }
        )
    return resultados


def imprimir_tabela(resultados):
    print("Treinamento | w0i w1i w2i w3i | w0f w1f w2f w3f | Epocas | Acc")
    for r in resultados:
        print(
            f"{r['T']} (seed {r['seed']}): "
            f"[{r['w0i']:.4f} {r['w1i']:.4f} {r['w2i']:.4f} {r['w3i']:.4f}] -> "
            f"[{r['w0f']:.4f} {r['w1f']:.4f} {r['w2f']:.4f} {r['w3f']:.4f}] "
            f"epocas={r['epocas']} acc={r['acc']:.3f}"
        )


if __name__ == "__main__":
    res = executar_treinamentos()
    imprimir_tabela(res)
