"""Teste do mapa de Kohonen — item 2 do trabalho.

- Carrega as 12 amostras de hopfield_kohonen/data/kohonen_teste.csv
  (tabela do enunciado, sem rotulo).
- Para cada um dos 5 treinamentos (T1..T5, seeds 10-50), apresenta cada amostra
  a rede, acha o neuronio vencedor (menor norma euclidiana) e atribui a classe
  dominante desse neuronio (mapeamento do item 1).
- Imprime a tabela de classificacao por treinamento + o consenso
  (voto majoritario entre T1..T5) e o neuronio vencedor em cada modelo.
"""
import csv
from pathlib import Path

import numpy as np

from hopfield_kohonen.kohonen.kohonen_treino import (
    BASE_DIR,
    executar_treinamentos,
)

TESTE_CSV = BASE_DIR / "data" / "kohonen_teste.csv"


def carregar_teste(path=TESTE_CSV):
    X, amostras = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            X.append([float(row["x1"]), float(row["x2"]), float(row["x3"])])
            amostras.append(int(row["amostra"]))
    return np.array(X), amostras


def classificar_amostras(resultados, X_teste):
    """Para cada treinamento: vencedor -> classe dominante do neuronio."""
    por_modelo = {}
    for r in resultados:
        rede = r["rede"]
        dominante = r["dominante"]
        linhas = []
        for x in X_teste:
            j = rede.vencedor(x)
            linhas.append({"neuronio": j + 1, "classe": dominante[j]})
        por_modelo[r["T"]] = linhas
    return por_modelo


def imprimir_tabela(por_modelo, resultados, amostras, X_teste):
    """Tabela do item 2: classe por modelo + consenso + vencedores."""
    modelos = list(por_modelo)

    print("=== Classificacao das 12 amostras de teste (item 2) ===")
    print("Amostra |    x1   |    x2   |    x3   | " + " | ".join(f"{m:>4}" for m in modelos) + " | Consenso")
    for i, (x, n) in enumerate(zip(X_teste, amostras)):
        classes = [por_modelo[m][i]["classe"] for m in modelos]
        # consenso por voto majoritario entre os modelos
        votos = {}
        for c in classes:
            votos[c] = votos.get(c, 0) + 1
        letra, n_votos = max(sorted(votos.items()), key=lambda kv: kv[1])
        consenso = f"{letra}{'*' if n_votos < len(modelos) else ''}"
        print(
            f"{n:>7} | {x[0]:.4f} | {x[1]:.4f} | {x[2]:.4f} | "
            + " | ".join(f"{c:>4}" for c in classes)
            + f" | {consenso:>8}"
        )

    print("\nNeuronio vencedor de cada amostra em cada modelo:")
    print("Amostra | " + " | ".join(f"{m:>9}" for m in modelos))
    for i, n in enumerate(amostras):
        neuronios = [por_modelo[m][i]["neuronio"] for m in modelos]
        print(f"{n:>7} | " + " | ".join(f"{c:>9}" for c in neuronios))

    print("\nLegenda: '*' = discordancia entre os 5 modelos; o consenso e o voto majoritario.")
    print("Classe do neuronio vencedor vem do mapa do item 1 (classe dominante por vitorias).")


if __name__ == "__main__":
    resultados, _, _ = executar_treinamentos()
    X_teste, amostras = carregar_teste()
    por_modelo = classificar_amostras(resultados, X_teste)
    imprimir_tabela(por_modelo, resultados, amostras, X_teste)
