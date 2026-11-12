"""RBF didatica — k-means na camada escondida + regra Delta na camada de saida.

Implementacao em NumPy puro, sem bibliotecas de redes neurais.

Convencoes usadas no projeto (as mesmas explicadas no relatorio):
  * Gaussiana: phi_j(x) = exp(-||x - c_j||^2 / (2 * sigma_j^2))
  * Variancia do cluster j (populacional):
        sigma_j^2 = (1/n_j) * soma_{i no cluster j} ||x_i - c_j||^2
  * Bias da camada de saida: x0 = -1, isto e:
        y = -W0 + W1*phi_1 + ... + Wk*phi_k
    (W0 e o threshold; no Adaline do curso o bias tambem era x0 = -1)
  * Regra Delta online: w <- w + eta * (d - y) * h,
    com h = [-1, phi_1, ..., phi_k] e erro linear e = d - y.
"""
from __future__ import annotations

import numpy as np


# ======================================================================
# Camada escondida: k-means
# ======================================================================
def distancias_quadraticas(X, centros):
    """Matriz (n_padroes, k) com ||x_i - c_j||^2 (distancia euclidiana ao quadrado)."""
    X = np.asarray(X, dtype=float)
    centros = np.asarray(centros, dtype=float)
    return ((X[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2)


def variancias_dos_clusters(X, centros, rotulos, ddof=0):
    """Variancia de cada cluster em torno do seu centro.

    ddof=0 -> populacional (divide por n_j), usado no relatorio.
    ddof=1 -> amostral (divide por n_j - 1), apenas para comparacao.
    """
    X = np.asarray(X, dtype=float)
    var = np.empty(len(centros), dtype=float)
    for j in range(len(centros)):
        dentro = rotulos == j
        n_j = int(dentro.sum())
        if n_j == 0:
            var[j] = np.nan
        else:
            var[j] = float(((X[dentro] - centros[j]) ** 2).sum() / (n_j - ddof))
    return var


def kmeans(X, k, seed=42, n_init=10, max_iter=300, tol=1e-10):
    """k-means classico (inicializacao Forgy) com reinicios.

    Para cada reinicio:
      1. escolhe k padroes distintos como centros iniciais (RNG com seed+reinicio);
      2. atribui cada padrao ao centro mais proximo (distancia euclidiana);
      3. recalcula cada centro como a media dos padroes atribuidos;
      4. repete ate o maior deslocamento dos centros ficar abaixo de `tol`
         (ou ate `max_iter`).

    Escolhe o reinicio de menor inercia (soma das distancias quadraticas
    intra-cluster) e reordena os clusters por tamanho decrescente
    (cluster 1 = maior). Isso deixa a tabela do exercicio deterministica.

    Retorna um dicionario com:
      centros, variancias, rotulos, n_por_cluster, inercia,
      iteracoes, deslocamentos, inercias_reinicios.
    """
    X = np.asarray(X, dtype=float)
    melhor = None
    inercias = []

    for reinicio in range(n_init):
        rng = np.random.default_rng(seed + reinicio)
        centros = X[rng.choice(len(X), size=k, replace=False)].copy()
        deslocamentos = []

        for it in range(1, max_iter + 1):
            rotulos = distancias_quadraticas(X, centros).argmin(axis=1)
            novos = np.array([
                X[rotulos == j].mean(axis=0) if np.any(rotulos == j) else centros[j]
                for j in range(k)
            ])
            desloc = float(np.max(np.linalg.norm(novos - centros, axis=1)))
            deslocamentos.append(desloc)
            centros = novos
            if desloc < tol:
                break

        d2 = distancias_quadraticas(X, centros)
        rotulos = d2.argmin(axis=1)
        inercia = float(d2[np.arange(len(X)), rotulos].sum())
        inercias.append(inercia)

        if melhor is None or inercia < melhor["inercia"]:
            melhor = {
                "centros": centros.copy(),
                "rotulos": rotulos.copy(),
                "inercia": inercia,
                "iteracoes": it,
                "deslocamentos": list(deslocamentos),
            }

    centros = melhor["centros"]
    rotulos = melhor["rotulos"]
    n_por_cluster = np.array([(rotulos == j).sum() for j in range(k)], dtype=int)

    # Ordena: cluster 0 = maior, cluster 1 = segundo maior, ...
    ordem = np.argsort(-n_por_cluster, kind="stable")
    mapa = {int(antigo): novo for novo, antigo in enumerate(ordem)}
    rotulos = np.array([mapa[int(r)] for r in rotulos], dtype=int)
    centros = centros[ordem]
    n_por_cluster = n_por_cluster[ordem]
    variancias = variancias_dos_clusters(X, centros, rotulos)

    return {
        "centros": centros,
        "variancias": variancias,
        "rotulos": rotulos,
        "n_por_cluster": n_por_cluster,
        "inercia": melhor["inercia"],
        "iteracoes": melhor["iteracoes"],
        "deslocamentos": melhor["deslocamentos"],
        "inercias_reinicios": np.array(inercias),
    }


# ======================================================================
# Camada escondida: ativacao gaussiana e montagem da matriz com bias
# ======================================================================
def ativacao_gaussiana(X, centros, variancias):
    """Ativacao radial: phi_j(x) = exp(-||x - c_j||^2 / (2 * sigma_j^2)).

    X: (n_padroes, n_entradas); centros: (k, n_entradas); variancias: (k,).
    Retorna a matriz Phi com shape (n_padroes, k).
    """
    d2 = distancias_quadraticas(X, centros)
    return np.exp(-d2 / (2.0 * np.asarray(variancias, dtype=float)[None, :]))


def matriz_com_bias(Phi):
    """Acrescenta a coluna do bias x0 = -1 (primeira coluna).

    A convencao x0 = -1 segue o Adaline do curso. Com ela, a saida fica
        y = -W0 + W1*phi_1 + ... + Wk*phi_k
    e W0 e o threshold do neuronio de saida.
    """
    Phi = np.asarray(Phi, dtype=float)
    return np.hstack((-np.ones((Phi.shape[0], 1)), Phi))


# ======================================================================
# Camada de saida: regra Delta (LMS) online
# ======================================================================
def treinar_camada_saida(H, d, eta=0.01, eps=1e-7, max_epocas=200_000, seed=42):
    """Regra Delta (LMS) online para a camada de saida linear.

    H: (n_padroes, k+1), ja com a coluna x0 = -1 na posicao 0.
    d: (n_padroes,) com os valores desejados.

    Para cada padrao, na ordem do arquivo:
        y = w . h
        e = d - y
        w <- w + eta * e * h

    Criterio de parada: |EQM(epoca) - EQM(epoca-1)| < eps.
    Retorna (w, historico_eqm, epocas, w_inicial).
    """
    H = np.asarray(H, dtype=float)
    d = np.asarray(d, dtype=float).ravel()

    rng = np.random.default_rng(seed)
    w = rng.random(H.shape[1])  # pesos iniciais U[0,1]
    w_inicial = w.copy()
    historico = []

    for _epoca in range(1, max_epocas + 1):
        soma_quadrados = 0.0
        for i in range(H.shape[0]):
            h = H[i]
            y = float(w @ h)
            e = float(d[i] - y)  # erro linear (d - y)
            soma_quadrados += e * e
            w = w + eta * e * h

        eqm = soma_quadrados / H.shape[0]
        historico.append(eqm)
        if len(historico) > 1 and abs(historico[-1] - historico[-2]) < eps:
            break

    return w, np.asarray(historico), len(historico), w_inicial


# ======================================================================
# Pos-processamento e metricas
# ======================================================================
def pos_processar(y):
    """Funcao sinal do item 3: +1 se y >= 0; -1 caso contrario.

    Aplicada apenas no conjunto de teste, como pede o enunciado.
    """
    return np.where(np.asarray(y, dtype=float) >= 0.0, 1, -1)


def eqm(d, y):
    """Erro quadratico medio entre vetores."""
    d = np.asarray(d, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return float(np.mean((d - y) ** 2))


def acuracia(y_previsto, d):
    """Taxa de acerto em porcentagem."""
    y_previsto = np.asarray(y_previsto)
    d = np.asarray(d)
    return 100.0 * float((y_previsto == d).mean())


def matriz_confusao(y_previsto, d):
    """Matriz de confusao para classes +1 (positiva) e -1 (negativa).

    Retorna (TP, FN, FP, TN):
      TP = real +1 e previsto +1
      FN = real +1 e previsto -1  (radiacao perdida)
      FP = real -1 e previsto +1  (alarme falso)
      TN = real -1 e previsto -1
    """
    y_previsto = np.asarray(y_previsto)
    d = np.asarray(d)
    tp = int(((y_previsto == 1) & (d == 1)).sum())
    fn = int(((y_previsto == -1) & (d == 1)).sum())
    fp = int(((y_previsto == 1) & (d == -1)).sum())
    tn = int(((y_previsto == -1) & (d == -1)).sum())
    return tp, fn, fp, tn
