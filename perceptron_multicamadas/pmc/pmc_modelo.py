"""PMC com atalho entrada->saida (W1, W2, W3). Implementacao propria, so numpy."""
import numpy as np


def sigmoid(net):
    return 1.0 / (1.0 + np.exp(-np.asarray(net, dtype=float)))


class PMC:
    """Perceptron multicamadas N -> N1 -> 1 com conexao direta N -> 1 (W3)."""

    def __init__(self, n_entradas=3, n_ocultos=5, seed=None):
        self.N = n_entradas
        self.N1 = n_ocultos
        if seed is not None:
            np.random.seed(seed)
        self.W1 = np.random.uniform(0, 1, (self.N, self.N1))
        self.W2 = np.random.uniform(0, 1, self.N1)
        self.W3 = np.random.uniform(0, 1, self.N)
        self.b1 = np.random.uniform(0, 1, self.N1)
        self.b2 = float(np.random.uniform(0, 1))

    def forward(self, x):
        x = np.asarray(x, dtype=float).ravel()
        net_h = x @ self.W1 - self.b1
        y_h = sigmoid(net_h)
        net_o = float(y_h @ self.W2 + x @ self.W3 - self.b2)
        o = float(sigmoid(net_o))
        return {"x": x, "net_h": net_h, "y_h": y_h, "net_o": net_o, "o": o}

    def backward_step(self, x, d, eta=0.01):
        """Um passo estocastico: forward + deltas + atualiza W2,W3 e depois W1."""
        ctx = self.forward(x)
        x, y_h, o = ctx["x"], ctx["y_h"], ctx["o"]
        d = float(d)

        delta_o = (d - o) * o * (1.0 - o)

        # Congela W2 do forward para o retropropagar (correto matematico).
        W2_old = self.W2.copy()

        # Saida primeiro: W2, W3, b2.
        # net = soma(w*x) - b  =>  dE/db = +delta, logo b -= eta*delta.
        self.W2 += eta * delta_o * y_h
        self.W3 += eta * delta_o * x
        self.b2 -= eta * delta_o

        # Escondida depois: delta_h via W2 antigo, entao W1, b1.
        delta_h = y_h * (1.0 - y_h) * delta_o * W2_old
        self.W1 += eta * np.outer(x, delta_h)
        self.b1 -= eta * delta_h

        erro = 0.5 * (d - o) ** 2
        return erro, o

    def prever(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        saidas = []
        for xi in X:
            saidas.append(self.forward(xi)["o"])
        return np.array(saidas)
