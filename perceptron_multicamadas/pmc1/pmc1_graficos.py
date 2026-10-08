"""Graficos PMC1 — item 3 (2 treinos mais lentos, lado a lado, nao superpostos)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pmc1.pmc1_treino import executar_5_treinamentos

BASE_DIR = Path(__file__).resolve().parent.parent
SAIDA_PNG = BASE_DIR / "docs" / "answers" / "pmc1_eqm_lentos.png"


def gerar_graficos(saida=SAIDA_PNG):
    resultados = executar_5_treinamentos()
    lentos = sorted(resultados, key=lambda r: r["epocas"], reverse=True)[:2]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, r in zip(axes, lentos):
        ax.plot(range(1, len(r["historico"]) + 1), r["historico"])
        ax.set_title(f"{r['T']} (seed {r['seed']}) — {r['epocas']} épocas")
        ax.set_xlabel("Época")
        ax.grid(True)
    axes[0].set_ylabel("EQM")
    fig.suptitle("PMC1 — EQM por época (2 treinos mais lentos)")
    fig.tight_layout()
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=150)
    plt.close(fig)
    print(f"Grafico salvo em {saida} : {[r['T'] for r in lentos]}")
    return saida


if __name__ == "__main__":
    gerar_graficos()
