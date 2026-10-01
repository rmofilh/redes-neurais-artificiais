"""Graficos do Adaline — item 3 do trabalho.

- Reutiliza executar_treinamentos() de adaline_treino.py (mesmos seeds/constantes)
- Plota EQM por epoca de T1 (seed 10) e T2 (seed 20) lado a lado na mesma figura
- Salva em perceptron_adaline/docs/answers/adaline_eqm_t1_t2.png para embutir no markdown
"""
from pathlib import Path

import matplotlib.pyplot as plt

from perceptron_adaline.adaline.adaline_treino import executar_treinamentos

BASE_DIR = Path(__file__).resolve().parent.parent
SAIDA_PNG = BASE_DIR / "docs" / "answers" / "adaline_eqm_t1_t2.png"


def gerar_graficos(saida=SAIDA_PNG):
    resultados = executar_treinamentos()
    t1 = next(r for r in resultados if r["T"] == "T1")
    t2 = next(r for r in resultados if r["T"] == "T2")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, r in zip(axes, (t1, t2)):
        eqm = r["modelo"].historico_eqm
        epocas = range(1, len(eqm) + 1)
        ax.plot(epocas, eqm)
        ax.set_title(f"{r['T']} (seed {r['seed']}) — {r['epocas']} épocas")
        ax.set_xlabel("Época")
        ax.grid(True)
    axes[0].set_ylabel("EQM")
    fig.suptitle("Adaline — EQM por época (T1 e T2)")
    fig.tight_layout()
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=150)
    plt.close(fig)
    print(f"Grafico salvo em {saida}")
    return saida


if __name__ == "__main__":
    gerar_graficos()
