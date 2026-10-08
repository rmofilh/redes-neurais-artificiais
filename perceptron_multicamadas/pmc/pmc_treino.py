"""Treino do PMC com atalho W3 — Fase 2.

- N=3 fixo pelos CSVs, N1=5, eta=0.01, logistica (ver docs/answers/PMC_respostas.md)
- Estocastico: embaralha os P=200 padroes a cada epoca
- Pesos ficam so em memoria (retorna modelo, sem .npz em disco)
"""
import csv
from pathlib import Path

import numpy as np

from pmc.pmc_modelo import PMC

N_ENTRADAS = 3
N_OCULTOS = 5
ETA = 0.01
MAX_EPOCAS = 1000
SEED = 10
BASE_DIR = Path(__file__).resolve().parent.parent
TREINO_CSV = BASE_DIR / "data" / "conjunto_treinamento.csv"


def carregar_treino(path=TREINO_CSV):
    X, y = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            X.append([float(row["x1"]), float(row["x2"]), float(row["x3"])])
            y.append(float(row["d"]))
    return np.array(X), np.array(y)


def executar_treinamento(seed=SEED, eta=ETA, max_epocas=MAX_EPOCAS, log_cada=100):
    X, y = carregar_treino()
    modelo = PMC(n_entradas=N_ENTRADAS, n_ocultos=N_OCULTOS, seed=seed)
    historico_mse = []
    for epoca in range(1, max_epocas + 1):
        idx = np.random.permutation(len(X))
        soma_e = 0.0
        for i in idx:
            e, _ = modelo.backward_step(X[i], y[i], eta=eta)
            soma_e += e
        mse = (2.0 * soma_e) / len(X)  # E=1/2(d-o)^2 -> MSE = media de (d-o)^2
        historico_mse.append(mse)
        if epoca == 1 or epoca % log_cada == 0 or epoca == max_epocas:
            print(f"epoca {epoca:4d}/{max_epocas}  MSE={mse:.6f}")
    return modelo, historico_mse


if __name__ == "__main__":
    executar_treinamento()
