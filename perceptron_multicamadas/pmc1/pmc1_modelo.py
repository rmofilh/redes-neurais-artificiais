"""PMC1 classico 3 -> 10 -> 1 (SEM atalho W3). Implementacao propria, so numpy."""
import numpy as np


def sigmoid(net):
    return 1.0 / (1.0 + np.exp(-np.asarray(net, dtype=float)))


class PMC1:
    """MLP N -> N1 -> 1 com logistica nas duas camadas."""

    def __init__(self, n_entradas=3, n_ocultos=10, seed=None):
        self.N = n_entradas
        self.N1 = n_ocultos
        if seed is not None:
            np.random.seed(seed)
        self.W1 = np.random.uniform(0, 1, (self.N, self.N1))
        self.W2 = np.random.uniform(0, 1, self.N1)
        self.b1 = np.random.uniform(0, 1, self.N1)
        self.b2 = float(np.random.uniform(0, 1))

    def forward(self, x):
        x = np.asarray(x, dtype=float).ravel()
        net_h = x @ self.W1 - self.b1
        y_h = sigmoid(net_h)
        net_o = float(y_h @ self.W2 - self.b2)
        o = float(sigmoid(net_o))
        return {"x": x, "net_h": net_h, "y_h": y_h, "net_o": net_o, "o": o}

    def backward_step(self, x, d, eta=0.1):
        ctx = self.forward(x)
        x, y_h, o = ctx["x"], ctx["y_h"], ctx["o"]
        d = float(d)

        delta_o = (d - o) * o * (1.0 - o)
        W2_old = self.W2.copy()

        self.W2 += eta * delta_o * y_h
        self.b2 -= eta * delta_o

        delta_h = y_h * (1.0 - y_h) * delta_o * W2_old
        self.W1 += eta * np.outer(x, delta_h)
        self.b1 -= eta * delta_h

        erro = 0.5 * (d - o) ** 2
        return erro, o

    def prever(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.array([self.forward(xi)["o"] for xi in X])
