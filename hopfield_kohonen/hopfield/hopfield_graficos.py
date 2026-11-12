"""Graficos do Hopfield — itens 1, 2 e 3 do trabalho.

- hopfield_padroes.png: os 4 padroes armazenados (grade 9x5, branco = -1, escuro = +1)
- hopfield_situacoes.png: as 12 situacoes de transmissao no trio do enunciado
  (Transmitida | Distorcida | Limpa), com o destino de cada recuperacao
- hopfield_ruido.png: taxa de recuperacao correta x nivel de ruido (item 3),
  curva agregada + curva por padrao + taxa de estados espurios
Salva em hopfield_kohonen/docs/answers/ para embutir no resposta.md.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hopfield_kohonen.hopfield.hopfield_modelo import grade_ascii, lado_a_lado
from hopfield_kohonen.hopfield.hopfield_treino import (
    carregar_padroes,
    executar_recuperacoes,
)
from hopfield_kohonen.hopfield.hopfield_teste import varrer_ruido

BASE_DIR = Path(__file__).resolve().parent.parent
SAIDA_DIR = BASE_DIR / "docs" / "answers"


def _celula(ax, vetor, titulo="", rotulo_y=""):
    """Desenha um padrao 9x5 como imagem com grade (escuro = +1)."""
    ax.imshow(np.asarray(vetor, dtype=float).reshape(9, 5), cmap="gray_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 9, 1), minor=True)
    ax.grid(which="minor", color="0.55", linewidth=0.6)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    if titulo:
        ax.set_title(titulo, fontsize=9)
    if rotulo_y:
        ax.set_ylabel(rotulo_y, fontsize=8)


def gerar_padroes(saida=SAIDA_DIR / "hopfield_padroes.png"):
    X, _ = carregar_padroes()
    fig, axes = plt.subplots(1, 4, figsize=(8, 3.4))
    for k, (ax, xk) in enumerate(zip(axes, X), start=1):
        _celula(ax, xk, titulo=f"Padrão {k}")
    fig.suptitle("Padrões armazenados (escuro = +1, branco = −1)")
    fig.tight_layout()
    fig.savefig(saida, dpi=150)
    plt.close(fig)
    return saida


def gerar_situacoes(prefixo="hopfield_situacoes"):
    """Dois paineis com as 12 situacoes (6 por figura), trio do enunciado.

    S = situacao, P = padrao original, sim = simulacao; o destino da
    recuperacao (ok / espurio / outro padrao) vai no rotulo de cada linha.
    """
    _, _, _, resultados = executar_recuperacoes()
    salvos = []
    for parte in range(2):
        bloco = resultados[parte * 6:(parte + 1) * 6]
        fig, axes = plt.subplots(len(bloco), 3, figsize=(5.4, 2.15 * len(bloco)))
        for ax_col, titulo in zip(axes[0], ("Transmitida", "Distorcida", "Limpa (recuperada)")):
            ax_col.set_title(titulo, fontsize=10)
        for ax_linha, r in zip(axes, bloco):
            if r["correta"]:
                destino = "ok"
            else:
                destino = "→ espúrio" if r["destino"] is None else f"→ Padrão {r['destino']}"
            rotulo = f"S{r['amostra']} (P{r['padrao']}·sim{r['simulacao']})\n{destino}"
            for ax, vetor in zip(ax_linha, (r["transmitida"], r["distorcida"], r["limpa"])):
                _celula(ax, vetor, rotulo_y=rotulo)
        fig.suptitle(
            "Situações de transmissão "
            f"{parte * 6 + 1}–{(parte + 1) * 6} (S = situação, P = padrão, sim = simulação)",
            fontsize=10, y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        saida = SAIDA_DIR / f"{prefixo}_{parte * 6 + 1}_{(parte + 1) * 6}.png"
        fig.savefig(saida, dpi=150)
        plt.close(fig)
        salvos.append(saida)
    return salvos


def gerar_ruido(saida=SAIDA_DIR / "hopfield_ruido.png"):
    _, tabela = varrer_ruido()
    niveis = [t["nivel"] for t in tabela]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(niveis, [100 * t["correto"] for t in tabela], "o-", color="black",
            linewidth=2.2, label="Correto (total)")
    cores = {1: "tab:blue", 2: "tab:red", 3: "tab:green", 4: "tab:orange"}
    for k in (1, 2, 3, 4):
        ax.plot(niveis, [100 * t["por_padrao"][k]["correto"] for t in tabela],
                "--", color=cores[k], alpha=0.85, label=f"Correto P{k}")
    ax.plot(niveis, [100 * t["espurio"] for t in tabela], "s:", color="gray",
            label="Estado espúrio (total)")
    ax.axvline(20, color="0.3", linestyle=":", linewidth=1)
    ax.annotate("nível do enunciado (20%)", xy=(20, 100), xytext=(24, 93), fontsize=8)
    ax.set_xlabel("Nível de ruído (%)")
    ax.set_ylabel("Taxa (%)")
    ax.set_title("Hopfield — recuperação correta × nível de ruído\n(50 repetições × 4 padrões por nível)")
    ax.set_xticks(niveis)
    ax.set_ylim(-3, 104)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(saida, dpi=150)
    plt.close(fig)
    return saida


if __name__ == "__main__":
    print(f"Grafico salvo em {gerar_padroes()}")
    for caminho in gerar_situacoes():
        print(f"Grafico salvo em {caminho}")
    print(f"Grafico salvo em {gerar_ruido()}")
