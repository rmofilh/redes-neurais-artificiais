"""Treino do mapa de Kohonen — itens 1 e 2 do trabalho.

- 120 amostras de hopfield_kohonen/data/kohonen_treinamento.csv
  (analise do enunciado: classes A = amostras 1-20, B = 21-60, C = 61-120).
- 5 treinamentos (T1..T5) com seeds 10, 20, 30, 40, 50; pesos iniciais U[0,1]
  sorteados FORA da classe, no script, com np.random.seed(seed) (padrao da casa).
- Grade 4x4 (16 neuronios numerados 1..16 linha a linha), vencedor pela menor
  norma euclidiana, vizinhanca de raio 1 (vencedor + 4 ortogonais) e
  eta = 0.001 FIXA (o enunciado nao pede decaimento).
- Parada: max |Delta w| da epoca < 1e-6; teto de 20000 epocas.
- Para cada treinamento imprime o resumo da execucao, o mapa de classes do
  grid (item 1) e a tabela de pesos finais; ao fim, a concordancia entre os
  5 mapas (estabilidade dos conjuntos de neuronios por classe).
"""
import csv
from pathlib import Path

import numpy as np

from hopfield_kohonen.kohonen.kohonen_modelo import Kohonen, mapa_texto

SEEDS = [10, 20, 30, 40, 50]
TAXA = 0.001  # eta fixa, conforme o enunciado
TOLERANCIA = 1e-6  # parada: max |Delta w| da epoca
MAX_EPOCAS = 20000  # teto de seguranca
LINHAS, COLUNAS = 4, 4
BASE_DIR = Path(__file__).resolve().parent.parent
TREINO_CSV = BASE_DIR / "data" / "kohonen_treinamento.csv"


def carregar_treino(path=TREINO_CSV):
    X, rotulos = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            X.append([float(row["x1"]), float(row["x2"]), float(row["x3"])])
            rotulos.append(row["classe"])
    return np.array(X), rotulos


def executar_treinamentos():
    """Executa os 5 treinamentos independentes; devolve (resultados, X, rotulos)."""
    X, rotulos = carregar_treino()
    resultados = []
    for i, seed in enumerate(SEEDS, start=1):
        np.random.seed(seed)
        W0 = np.random.uniform(0, 1, (LINHAS * COLUNAS, X.shape[1]))  # U[0,1] fora da classe
        rede = Kohonen(eta=TAXA, tolerancia=TOLERANCIA, max_epocas=MAX_EPOCAS)
        rede.treinar(X, W0)
        dominante, contagens = rede.mapear_rotulos(X, rotulos)
        resultados.append(
            {
                "T": f"T{i}",
                "seed": seed,
                "W0": W0,
                "rede": rede,
                "dominante": dominante,
                "contagens": contagens,
                "eq_inicial": rede.eq_por_epoca[0],
                "eq_final": rede.eq_por_epoca[-1],
                "maxdw_final": rede.maxdw_por_epoca[-1],
                "epocas": rede.epocas,
                "criterio": rede.criterio_atingido,
                "ultima_mudanca": rede.ultima_mudanca,
            }
        )
    return resultados, X, rotulos


def imprimir_resumo(resultados):
    print("=== Resumo dos 5 treinamentos (eta=0.001 fixa, vizinhanca raio 1) ===")
    print("Treino | Seed | Epocas | Parada            | EQ inicial | EQ final | max|dw| final | Ultima mudanca")
    for r in resultados:
        parada = "criterio 1e-6" if r["criterio"] else f"teto ({r['epocas']} epocas)"
        print(
            f"{r['T']:>6} | {r['seed']:>4} | {r['epocas']:>6} | {parada:>17} | "
            f"{r['eq_inicial']:>10.5f} | {r['eq_final']:>8.5f} | {r['maxdw_final']:>13.2e} | "
            f"epoca {r['ultima_mudanca']}"
        )


def imprimir_mapas(resultados):
    for r in resultados:
        print(f"\n=== {r['T']} (seed {r['seed']}) — mapa de classes ('-' = sem vitoria) ===")
        print(mapa_texto(r["dominante"]))


def imprimir_tabela_pesos(r):
    """Pesos finais: 16 neuronios x (w1, w2, w3) + vitorias por classe."""
    print(f"\n=== {r['T']} (seed {r['seed']}) — pesos finais ===")
    print("Neuronio |   w1   |   w2   |   w3   | Vitorias A/B/C | Dominante")
    rede = r["rede"]
    for j in range(rede.n_neuronios):
        w = rede.W[j]
        c = r["contagens"][j]
        vitorias = f"{c.get('A', 0)}/{c.get('B', 0)}/{c.get('C', 0)}"
        print(
            f"{j + 1:>8} | {w[0]:.4f} | {w[1]:.4f} | {w[2]:.4f} | {vitorias:>14} | {r['dominante'][j]:>9}"
        )


def regioes_conexas(dominante, linhas=4, colunas=4):
    """True se as celulas ativas de cada classe formam regiao conexa no grid."""
    for classe in ("A", "B", "C"):
        celulas = {j for j, letra in enumerate(dominante) if letra == classe}
        if not celulas:
            continue
        fila, vistos = [next(iter(celulas))], set()
        while fila:
            j = fila.pop()
            if j in vistos:
                continue
            vistos.add(j)
            lin, col = j // colunas, j % colunas
            for dl, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                l2, c2 = lin + dl, col + dc
                if 0 <= l2 < linhas and 0 <= c2 < colunas:
                    k = l2 * colunas + c2
                    if k in celulas:
                        fila.append(k)
        if vistos != celulas:
            return False
    return True


def imprimir_concordancia(resultados):
    """Item 1: estabilidade dos conjuntos de neuronios por classe.

    A orientacao do mapa no grid depende da inicializacao (propriedade do
    Kohonen), entao a comparacao celula a celula difere entre treinos; a
    ESTRUTURA (numero de neuronios por classe, regioes conexas e pesos) e
    estavel — resumida na tabela abaixo.
    """
    print("\n=== Concordancia entre os 5 mapas (celulas iguais) ===")
    mapas = [tuple(r["dominante"]) for r in resultados]
    for i in range(len(mapas)):
        for k in range(i + 1, len(mapas)):
            iguais = sum(1 for a, b in zip(mapas[i], mapas[k]) if a == b)
            print(f"  {resultados[i]['T']} vs {resultados[k]['T']}: {iguais}/16")

    print("\n=== Estrutura de cada mapa (item 1) ===")
    print("Treino | Classe A        | Classe B        | Classe C        | Sem resposta   | Regioes conexas")
    for r in resultados:
        grupos = {letra: [j + 1 for j, x in enumerate(r["dominante"]) if x == letra]
                  for letra in ("A", "B", "C", "-")}
        celulas = " | ".join(
            f"{letra}: {len(grupos[letra])} ({' '.join(f'n{n}' for n in grupos[letra]) or '-'})"
            for letra in ("A", "B", "C")
        )
        sem = f"{len(grupos['-'])} ({' '.join(f'n{n}' for n in grupos['-']) or '-'})"
        conexa = "sim" if regioes_conexas(r["dominante"]) else "NAO"
        print(f"{r['T']:>6} | {celulas} | {sem:>14} | {conexa}")


if __name__ == "__main__":
    resultados, X, rotulos = executar_treinamentos()
    imprimir_resumo(resultados)
    imprimir_mapas(resultados)
    for r in resultados:
        imprimir_tabela_pesos(r)
    imprimir_concordancia(resultados)
