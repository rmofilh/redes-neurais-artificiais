"""Treino e recuperacao na rede de Hopfield — itens 1 e 2 do trabalho.

- Armazena os 4 padroes (grade 9x5, 45 bits; branco = -1, escuro = +1) de
  hopfield_kohonen/data/hopfield_treinamento.csv pela regra do produto externo.
- Apresenta a rede as 12 situacoes de transmissao de
  hopfield_kohonen/data/hopfield_teste.csv (3 por padrao, 9/45 = 20% dos
  pixels corrompidos) e recupera cada uma pela dinamica assincrona.
- Para cada situacao imprime o trio do exemplo do enunciado:
  Imagem Transmitida -> Imagem Distorcida -> Imagem Limpa (recuperada).
"""
import csv
from pathlib import Path

import numpy as np

from hopfield_kohonen.hopfield.hopfield_modelo import (
    Hopfield,
    binarizar,
    grade_ascii,
    lado_a_lado,
)

BASE_DIR = Path(__file__).resolve().parent.parent
TREINO_CSV = BASE_DIR / "data" / "hopfield_treinamento.csv"
TESTE_CSV = BASE_DIR / "data" / "hopfield_teste.csv"
BETA = 1000.0  # beta muito grande -> tanh satura em ±1
SEED_ORDEM = 10  # seed da ordem aleatoria das atualizacoes assincronas
N_PIXELS = 45


def carregar_padroes(path=TREINO_CSV):
    """Padroes limpos: matriz (4, 45) e rotulos [1..4]."""
    X, rotulos = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            X.append([float(row[f"x{i}"]) for i in range(1, N_PIXELS + 1)])
            rotulos.append(int(row["padrao"]))
    return np.array(X), rotulos


def carregar_situacoes(path=TESTE_CSV):
    """As 12 situacoes de transmissao ja fornecidas (amostra, padrao, sim, s0)."""
    situacoes = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            situacoes.append(
                {
                    "amostra": int(row["amostra"]),
                    "padrao": int(row["padrao_original"]),
                    "simulacao": int(row["simulacao"]),
                    "s0": np.array([float(row[f"x{i}"]) for i in range(1, N_PIXELS + 1)]),
                }
            )
    return situacoes


def executar_recuperacoes():
    """Arma W e recupera as 12 situacoes; devolve rede, padroes e resultados."""
    X, rotulos = carregar_padroes()
    rede = Hopfield(beta=BETA)
    rede.treinar(X)

    resultados = []
    for sit in carregar_situacoes():
        alvo = X[sit["padrao"] - 1]  # imagem transmitida (padrao original)
        s0 = sit["s0"]  # imagem distorcida que chega na recepcao
        sf = rede.lembrar(s0, seed=SEED_ORDEM)
        limpo = binarizar(sf)  # imagem limpa recuperada

        destino = next(
            (k + 1 for k, xk in enumerate(X) if (limpo == xk).all()), None
        )
        resultados.append(
            {
                "amostra": sit["amostra"],
                "padrao": sit["padrao"],
                "simulacao": sit["simulacao"],
                "corrompidos": int((s0 != alvo).sum()),
                "transmitida": alvo,
                "distorcida": s0,
                "limpa": limpo,
                "correta": bool((limpo == alvo).all()),
                "destino": destino,  # padrao armazenado alcancado (ou None = espurio)
                "varreduras": rede.n_varreduras,
                "convergiu": rede.convergiu,
                "energia_inicial": rede.energias[0],
                "energia_final": rede.energias[-1],
            }
        )
    return rede, X, rotulos, resultados


def imprimir_situacoes(resultados):
    """Item 2: para cada situacao mostra o trio transmitida/distorcida/limpa."""
    for r in resultados:
        pct = 100.0 * r["corrompidos"] / N_PIXELS
        print(
            f"=== Situacao {r['amostra']} — padrao {r['padrao']}, simulacao "
            f"{r['simulacao']} ({r['corrompidos']}/45 pixels corrompidos = {pct:.0f}%) ==="
        )
        # Cabecalho curto para manter colunas alinhadas com as grades 9x5
        print(lado_a_lado("Transmitida", "Distorcida", "Limpa"))
        print(
            lado_a_lado(
                grade_ascii(r["transmitida"]),
                grade_ascii(r["distorcida"]),
                grade_ascii(r["limpa"]),
            )
        )
        destino_txt = f"padrao {r['destino']}" if r["destino"] else "estado espurio"
        ok_txt = "OK" if r["correta"] else "DIVERGE"
        print(
            f"-> {ok_txt}: estado final = {destino_txt} em {r['varreduras']} varreduras | "
            f"E inicial {r['energia_inicial']:.4f} -> E final {r['energia_final']:.4f}"
        )
        print()


def imprimir_tabela(resultados):
    """Resumo numerico das 12 situacoes (base da tabela do resposta.md)."""
    print("Situacao | Padrao | Sim | Corrompidos | Recuperou | Varreduras | E inicial | E final")
    for r in resultados:
        print(
            f"{r['amostra']:>8} | {r['padrao']:>6} | {r['simulacao']:>3} | "
            f"{r['corrompidos']:>11} | {'sim' if r['correta'] else 'NAO':>9} | "
            f"{r['varreduras']:>10} | {r['energia_inicial']:>9.4f} | {r['energia_final']:>8.4f}"
        )
    acertos = sum(1 for r in resultados if r["correta"])
    print(f"\n{acertos}/{len(resultados)} situacoes recuperaram o padrao original.")


if __name__ == "__main__":
    rede, X, rotulos, resultados = executar_recuperacoes()
    imprimir_situacoes(resultados)
    imprimir_tabela(resultados)
