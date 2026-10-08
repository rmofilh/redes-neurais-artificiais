"""Treino PMC1 — itens 1 e 2 (5 treinamentos, eta=0.1, eps=1e-6).

Pesos ficam so em memoria (sem .npz). Historico de EQM de cada treino
e salvo em pmc1/resultados/eqm_T*.csv para documentar o item 2/3.
"""
import csv
from pathlib import Path

import numpy as np

from pmc1.pmc1_modelo import PMC1

SEEDS = [10, 20, 30, 40, 50]
ETA = 0.1
EPS = 1e-6
MAX_EPOCAS = 5000
BASE_DIR = Path(__file__).resolve().parent.parent
TREINO_CSV = BASE_DIR / "data" / "conjunto_treinamento.csv"
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"


def carregar_treino(path=TREINO_CSV):
    X, y = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            X.append([float(row["x1"]), float(row["x2"]), float(row["x3"])])
            y.append(float(row["d"]))
    return np.array(X), np.array(y)


def treinar_um(seed, eta=ETA, eps=EPS, max_epocas=MAX_EPOCAS):
    X, y = carregar_treino()
    modelo = PMC1(n_entradas=3, n_ocultos=10, seed=seed)
    historico = []
    eqm_ant = None
    epocas = 0
    for epoca in range(1, max_epocas + 1):
        idx = np.random.permutation(len(X))
        soma_e = 0.0
        for i in idx:
            e, _ = modelo.backward_step(X[i], y[i], eta=eta)
            soma_e += e
        eqm = (2.0 * soma_e) / len(X)
        historico.append(eqm)
        epocas = epoca
        if eqm_ant is not None and abs(eqm_ant - eqm) < eps:
            break
        eqm_ant = eqm
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(RESULTADOS_DIR / f"eqm_T{SEEDS.index(seed) + 1}.csv", historico, delimiter=",")
    return {"seed": seed, "epocas": epocas, "eqm": historico[-1],
            "historico": historico, "modelo": modelo}


def executar_5_treinamentos():
    resultados = []
    for k, seed in enumerate(SEEDS, start=1):
        r = treinar_um(seed)
        r["T"] = f"T{k}"
        resultados.append(r)
        print(f"{r['T']} (seed {seed}): EQM={r['eqm']:.6f} epocas={r['epocas']}")
    return resultados


if __name__ == "__main__":
    executar_5_treinamentos()
