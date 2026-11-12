"""Graficos do Kohonen — itens 1, 2 e 3 do trabalho.

- kohonen_mapa.png: mapa de classes do grid 4x4 para cada treinamento
  (T1..T5) + o consenso (item 1) — evidencia a estabilidade entre treinos
- kohonen_convergencia.png: curvas de convergencia dos 5 treinamentos:
  erro medio de quantizacao por epoca e maior variacao de peso por epoca
  (visualiza por que o criterio 1e-6 nao dispara com eta fixa)
- kohonen_dispersao.png: dispersao das 120 amostras (x1x x2 e x2x x3)
  coloridas pela classe real, com os pesos finais marcados (T1 em X colorido
  pela classe do neuronio; T2..T5 vazados em cinza) — preservacao topologica
Salva em hopfield_kohonen/docs/answers/ para embutir no resposta.md.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from hopfield_kohonen.kohonen.kohonen_treino import (
    carregar_treino,
    executar_treinamentos,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SAIDA_DIR = BASE_DIR / "docs" / "answers"
CORES = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "-": "#d9d9d9"}


def _painel_grid(ax, dominante, titulo):
    """Desenha o grid 4x4 com a classe dominante de cada neuronio."""
    for j in range(16):
        lin, col = j // 4, j % 4
        letra = dominante[j]
        ax.add_patch(
            Rectangle(
                (col, 3 - lin), 1, 1,
                facecolor=CORES[letra[0]], edgecolor="black", linewidth=1.1,
            )
        )
        ax.text(col + 0.5, 3 - lin + 0.70, str(j + 1), ha="center", va="center",
                fontsize=7.5, color="0.25")
        ax.text(col + 0.5, 3 - lin + 0.34, letra, ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if letra != "-" else "0.45")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(titulo, fontsize=10)


def gerar_mapa(resultados, saida=SAIDA_DIR / "kohonen_mapa.png"):
    """Item 1: mapa de classes dos 5 treinamentos (a orientacao varia com a init)."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0))
    for ax, r in zip(axes.flat, resultados):
        _painel_grid(ax, r["dominante"], f"{r['T']} (seed {r['seed']})")
    axes.flat[5].axis("off")  # painel livre: legenda
    handles = [
        Patch(facecolor=CORES["A"], edgecolor="black", label="Classe A (amostras 1–20)"),
        Patch(facecolor=CORES["B"], edgecolor="black", label="Classe B (amostras 21–60)"),
        Patch(facecolor=CORES["C"], edgecolor="black", label="Classe C (amostras 61–120)"),
        Patch(facecolor=CORES["-"], edgecolor="black", label="Neurônio sem vitória"),
    ]
    axes.flat[5].legend(handles=handles, loc="center", fontsize=9, frameon=False)
    fig.suptitle("Kohonen — conjuntos de neurônios por classe no grid 4×4 (item 1)\n"
                 "a estrutura se repete nos 5 treinos (A: 20 vitórias em 1–2 neurônios; B: 40 em 5–6; "
                 "C: 60 em 4–7; regiões conexas),\nmas a orientação no grid depende da inicialização "
                 "(propriedade do Kohonen)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(saida, dpi=150)
    plt.close(fig)
    return saida


def gerar_convergencia(resultados, saida=SAIDA_DIR / "kohonen_convergencia.png"):
    """Convergencia: EQ por epoca e max |Delta w| por epoca (5 treinamentos)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.6), sharex=True)
    for r in resultados:
        epocas = range(1, r["rede"].epocas + 1)
        ax1.plot(epocas, r["rede"].eq_por_epoca, linewidth=1.1, alpha=0.85, label=r["T"])
        ax2.plot(epocas, r["rede"].maxdw_por_epoca, linewidth=1.1, alpha=0.85, label=r["T"])
    ax1.set_yscale("log")
    ax1.set_ylabel("EQ = média de ½‖W* − x‖²")
    ax1.set_title("Kohonen — convergência dos 5 treinamentos (η = 0.001 fixa, 20.000 épocas)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")
    ax2.set_ylim(0, 1.05e-3)
    ax2.axhline(1e-6, color="black", linestyle="--", linewidth=1.2)
    ax2.annotate("critério de parada (1e-6): nunca alcançado",
                 xy=(12000, 1e-6), xytext=(3000, 2.2e-4), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    ax2.annotate("platô ≈ 4.7e-4 = η × distância das atualizações\n"
                 "(com η fixa, a oscilação em torno dos centróides não cessa)",
                 xy=(15000, 4.71e-4), xytext=(6000, 7.4e-4), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    ax2.set_ylabel("max |Δw| na época")
    ax2.set_xlabel("Época")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(saida, dpi=150)
    plt.close(fig)
    return saida


def gerar_dispersao(resultados, X, rotulos, saida=SAIDA_DIR / "kohonen_dispersao.png"):
    """Amostras coloridas pela classe real + pesos finais marcados."""
    r1 = resultados[0]
    W1 = r1["rede"].W
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    pares = [(0, 1, "x1", "x2"), (1, 2, "x2", "x3")]
    for ax, (i, k, li, lk) in zip(axes, pares):
        for cls in ("A", "B", "C"):
            idx = [n for n, v in enumerate(rotulos) if v == cls]
            ax.scatter(X[idx, i], X[idx, k], s=16, color=CORES[cls], alpha=0.55,
                       label=f"amostras {cls}")
        for r in resultados[1:]:  # pesos finais T2..T5: cinza vazado
            W = r["rede"].W
            ax.scatter(W[:, i], W[:, k], s=26, facecolors="none",
                       edgecolors="0.55", linewidths=1.0)
        for j in range(16):  # pesos finais T1: X colorido pela classe do neuronio
            letra = r1["dominante"][j]
            ax.scatter(W1[j, i], W1[j, k], marker="X", s=140, color=CORES[letra[0]],
                       edgecolors="black", linewidths=0.8, zorder=5)
            ax.annotate(str(j + 1), (W1[j, i], W1[j, k]), textcoords="offset points",
                        xytext=(7, 4), fontsize=7.5, color="0.2")
        ax.set_xlabel(li)
        ax.set_ylabel(lk)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    from matplotlib.lines import Line2D
    extras = [
        Line2D([], [], marker="X", linestyle="", markersize=10, markeredgecolor="black",
               markerfacecolor="0.7", label="pesos finais T1 (X = neurônio 1–16)"),
        Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="none",
               markeredgecolor="0.55", label="pesos finais T2–T5"),
    ]
    handles, _ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles + extras, fontsize=8, loc="upper left")
    fig.suptitle("Kohonen — amostras × pesos finais: preservação topológica dos 3 grupos", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(saida, dpi=150)
    plt.close(fig)
    return saida


if __name__ == "__main__":
    resultados, X, rotulos = executar_treinamentos()
    print(f"Grafico salvo em {gerar_mapa(resultados)}")
    print(f"Grafico salvo em {gerar_convergencia(resultados)}")
    print(f"Grafico salvo em {gerar_dispersao(resultados, X, rotulos)}")
