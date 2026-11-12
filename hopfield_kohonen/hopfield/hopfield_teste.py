"""Teste do Hopfield — item 3 do trabalho (efeito do ruido excessivo).

Varre o nivel de ruido de 0% a 100% (passos de 10%). Para cada nivel, cada um
dos 4 padroes e distorcido em REPETICOES situacoes (seeds fixos por repeticao)
e a rede tenta recuperar o padrao. Cada estado final e classificado em:
  correto       -> estado final = padrao original apresentado
  outro padrao  -> estado final = outro padrao armazenado
  espurio       -> estado estavel que nao corresponde a nenhum padrao armazenado
  nao convergiu -> atingiu o teto de varreduras sem estabilizar
No fim imprime a tabela por nivel (e por padrao) e o diagnostico estrutural
(energias dos padroes armazenados e overlaps entre eles) que fundamenta a
explicacao do item 3 no resposta.md.
"""
import csv
from pathlib import Path

import numpy as np

from hopfield_kohonen.hopfield.hopfield_modelo import Hopfield, binarizar
from hopfield_kohonen.hopfield.hopfield_treino import (
    BETA,
    SEED_ORDEM,
    carregar_padroes,
)

BASE_DIR = Path(__file__).resolve().parent.parent
NIVEIS_RUIDO = list(range(0, 101, 10))
REPETICOES = 50
SEED_BASE = 100  # seed do sorteio de pixels: rep r usa np.random.seed(SEED_BASE + r)
TETO = 200


def varrer_ruido():
    """Devolve (tabela_nivel, tabela_padrao_nivel) com as taxas de cada classe."""
    X, _ = carregar_padroes()
    rede = Hopfield(beta=BETA, max_varreduras=TETO)
    rede.treinar(X)

    tabela = []
    for nivel in NIVEIS_RUIDO:
        por_padrao = {k + 1: {"correto": 0, "outro": 0, "espurio": 0, "nao_conv": 0}
                      for k in range(len(X))}
        varreduras_tot = 0
        total = REPETICOES * len(X)
        for rep in range(REPETICOES):
            np.random.seed(SEED_BASE + rep)  # mesmo sorteio base por nivel
            for k, xk in enumerate(X):
                s0 = rede.distorcer(xk, nivel)
                sf = rede.lembrar(s0, seed=SEED_ORDEM)
                limpo = binarizar(sf)
                varreduras_tot += rede.n_varreduras
                reg = por_padrao[k + 1]
                if (limpo == xk).all():
                    reg["correto"] += 1
                elif rede.convergiu:
                    if any((limpo == xj).all() for j, xj in enumerate(X) if j != k):
                        reg["outro"] += 1
                    else:
                        reg["espurio"] += 1
                else:
                    reg["nao_conv"] += 1
        agreg = {
            "correto": sum(p["correto"] for p in por_padrao.values()) / total,
            "outro": sum(p["outro"] for p in por_padrao.values()) / total,
            "espurio": sum(p["espurio"] for p in por_padrao.values()) / total,
            "nao_conv": sum(p["nao_conv"] for p in por_padrao.values()) / total,
        }
        tabela.append(
            {
                "nivel": nivel,
                **agreg,
                "media_varreduras": varreduras_tot / total,
                "por_padrao": {k: {c: v / REPETICOES for c, v in p.items()}
                               for k, p in por_padrao.items()},
            }
        )
    return rede, tabela


def imprimir_diagnostico_estrutural(rede, X):
    """Energia de cada padrao armazenado e matriz de overlaps — base do item 3."""
    print("=== Energia dos padroes armazenados (E = -1/2 s' W s) ===")
    for k, xk in enumerate(X, 1):
        print(f"  P{k}: E = {rede.energia(xk):.4f}")
    print("\n=== Overlaps m_jk = (P_j . P_k)/45 ===")
    print("      " + "  ".join(f"P{k}" for k in range(1, len(X) + 1)))
    for j, xj in enumerate(X, 1):
        linha = "  ".join(f"{float(xj @ xk / len(xj)):+.3f}" for xk in X)
        print(f"  P{j}  {linha}")


def imprimir_tabela_ruido(tabela):
    print("\n=== Taxa de recuperacao x nivel de ruido (por nivel, agregado) ===")
    print("Nivel | Correto | Outro padrao | Espurio | Nao convergiu | Varreduras (media)")
    for t in tabela:
        print(
            f"{t['nivel']:>4}% | {100 * t['correto']:>6.1f}% | {100 * t['outro']:>11.1f}% | "
            f"{100 * t['espurio']:>6.1f}% | {100 * t['nao_conv']:>12.1f}% | "
            f"{t['media_varreduras']:>12.1f}"
        )
    print("\n=== Taxa de recuperacao correta por padrao (%) ===")
    print("Nivel |   P1   |   P2   |   P3   |   P4")
    for t in tabela:
        celulas = " | ".join(
            f"{100 * t['por_padrao'][k]['correto']:>5.1f}" for k in sorted(t["por_padrao"])
        )
        print(f"{t['nivel']:>4}% | {celulas}")


if __name__ == "__main__":
    X, _ = carregar_padroes()
    rede, tabela = varrer_ruido()
    imprimir_diagnostico_estrutural(rede, X)
    imprimir_tabela_ruido(tabela)
