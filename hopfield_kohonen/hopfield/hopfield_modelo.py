"""Rede de Hopfield — memoria associativa (trabalho: 45 neuronios, grade 9x5).

Formulacao conforme o enunciado:
- Matriz de pesos pela regra do produto externo, normalizada e com diagonal nula:
      W = (1/N) * soma_k x_k x_k^T,   w_ii = 0
- Funcao de ativacao: Tangente Hiperbolica com beta muito grande.
  Com beta grande, tanh(beta * net) satura em +1 ou -1 e se comporta como a
  funcao sinal — mas mantendo a forma continua pedida no enunciado.
- Recuperacao (fase de lembranca) ASSINCRONA: a cada passo apenas um neuronio
  atualiza o estado, escolhido em ordem aleatoria fixa por seed. Com W simetrica
  e diagonal nula o estado tende a estabilizar; a energia
      E(s) = -1/2 s^T W s
  e registrada apos o estado inicial e apos cada varredura para acompanhar o
  decrescimento (com estados saturados em {±1} a energia nunca aumenta).

Nesta atividade: N = 45 neuronios (grade 9x5 lida linha a linha — row-major),
4 padroes armazenados (branco = -1, escuro = +1).
"""
import numpy as np


class Hopfield:
    """Memoria associativa de Hopfield com atualizacao assincrona."""

    def __init__(self, beta=1000.0, max_varreduras=200):
        self.beta = beta  # beta muito grande -> tanh(beta*net) ~ sinal(net)
        self.max_varreduras = max_varreduras  # teto: para antes se estabilizar
        self.W = None  # matriz de pesos (N x N), simetrica, diagonal nula
        self.n_neuronios = 0
        self.energias = []  # E do estado inicial e E apos cada varredura
        self.n_varreduras = 0  # varreduras completas ate estabilizar
        self.convergiu = False

    # ---------------- armazenamento ----------------
    def treinar(self, X):
        """Armazena os padroes (linhas de X, valores ±1) pelo produto externo."""
        X = np.asarray(X, dtype=float)
        n_padroes, N = X.shape
        W = np.zeros((N, N))
        for x in X:
            W += np.outer(x, x)  # produto externo x x^T
        W /= N  # normalizacao 1/N
        np.fill_diagonal(W, 0.0)  # sem auto-conexao (w_ii = 0)
        self.W = W
        self.n_neuronios = N
        return W

    # ---------------- dinamica ----------------
    def ativacao(self, net):
        """Tangente hiperbolica com beta muito grande (satura em ±1)."""
        return np.tanh(self.beta * np.asarray(net, dtype=float))

    def energia(self, s):
        """E(s) = -1/2 s^T W s."""
        return float(-0.5 * s @ self.W @ s)

    def lembrar(self, s_inicial, seed=10):
        """Recupera o padrao a partir do estado (distorcido) s_inicial.

        Atualizacao assincrona: por varredura, percorre os 45 neuronios em
        ordem aleatoria (gerador isolado default_rng(seed) — nao altera o
        gerador global np.random) e atualiza um de cada vez:
            s_i <- tanh(beta * net_i),   net_i = soma_j w_ij s_j
        Para quando uma varredura inteira nao altera nenhum neuronio.
        """
        if self.W is None:
            raise RuntimeError("Chame treinar(X) antes de lembrar().")
        s = np.asarray(s_inicial, dtype=float).copy()
        rng = np.random.default_rng(seed)
        N = self.n_neuronios

        self.energias = [self.energia(s)]
        self.convergiu = False
        for varredura in range(1, self.max_varreduras + 1):
            mudou = False
            for i in rng.permutation(N):
                net_i = self.W[i] @ s
                novo = self.ativacao(net_i)
                if abs(novo - s[i]) > 1e-12:
                    s[i] = novo
                    mudou = True
            self.energias.append(self.energia(s))
            if not mudou:
                self.convergiu = True
                self.n_varreduras = varredura
                break
        else:
            self.n_varreduras = self.max_varreduras
        return s

    def distorcer(self, x, nivel):
        """Corrompe nivel% dos pixels (branco<->escuro) sorteando indices.

        Usa o gerador global np.random (semeado pelo chamador), igual as
        demais atividades. Ex.: nivel=20 com N=45 inverte 9 pixels.
        """
        x = np.asarray(x, dtype=float).copy()
        k = int(round(x.size * nivel / 100.0))
        idx = np.random.choice(x.size, size=k, replace=False)
        x[idx] *= -1.0
        return x


# ---------------- utilidades de visualizacao ----------------
def binarizar(s):
    """Satura o estado continuo em ±1: >= 0 -> +1 (escuro), < 0 -> -1 (branco)."""
    return np.where(np.asarray(s) >= 0, 1, -1)


def grade_ascii(s, linhas=9, colunas=5):
    """Desenha o padrao 9x5 em texto: '#' = escuro (+1), '.' = branco (-1)."""
    s = np.asarray(s).ravel()
    return "\n".join(
        "".join("#" if v > 0 else "." for v in s[r * colunas:(r + 1) * colunas])
        for r in range(linhas)
    )


def lado_a_lado(*blocos, espaco=3):
    """Junta blocos de texto (mesmo numero de linhas) lado a lado."""
    colunas = [b.split("\n") for b in blocos]
    larguras = [max(len(l) for l in c) for c in colunas]
    return "\n".join(
        (" " * espaco).join(c[r].ljust(larguras[k]) for k, c in enumerate(colunas))
        for r in range(len(colunas[0]))
    )
