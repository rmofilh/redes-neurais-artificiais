"""PMC bem simples (didatico): 1 camada oculta, sigmoid, backprop padrao/momentum.
So usa numpy. Sem libs de redes neurais.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(y):
    # y = sigmoid(x) ja calculado
    return y * (1.0 - y)


class PMC:
    def __init__(self, n_in, n_hidden, n_out, seed=42):
        rng = np.random.default_rng(seed)
        # +1 para o bias (linha extra no inicio das matrizes)
        self.W1 = rng.random((n_in + 1, n_hidden))   # (5 x 15)
        self.W2 = rng.random((n_hidden + 1, n_out))  # (16 x 3)

    def forward(self, x):
        """x: vetor (n_in,). Retorna (y, h_com_bias, h, x_com_bias)."""
        xb = np.concatenate(([1.0], x))          # bias de entrada
        h = sigmoid(xb @ self.W1)                 # oculta
        hb = np.concatenate(([1.0], h))          # bias oculta
        y = sigmoid(hb @ self.W2)                 # saida
        return y, hb, h, xb

    def train(self, X, D, eta=0.1, alpha=0.0, eps=1e-6, max_epocas=5000):
        """Backprop online (padrao por padrao, na ordem do arquivo).
        alpha=0 -> padrao. alpha>0 -> com momentum.
        Parada: |EQM_atual - EQM_ant| < eps. Retorna (eqms, epocas, tempo).
        """
        import time
        P = X.shape[0]
        dW1_prev = np.zeros_like(self.W1)
        dW2_prev = np.zeros_like(self.W2)
        eqms = []
        t0 = time.time()
        eqm_ant = None

        for epoca in range(1, max_epocas + 1):
            sq = 0.0
            for x, d in zip(X, D):
                y, hb, h, xb = self.forward(x)
                e = d - y
                sq += np.sum(e ** 2)

                # deltas (regra da cadeia com derivada da sigmoid)
                delta_out = e * dsigmoid(y)                    # (n_out,)
                delta_hid = (self.W2[1:, :] @ delta_out) * dsigmoid(h)  # (n_hidden,)

                # gradientes (online)
                gW2 = np.outer(hb, delta_out)
                gW1 = np.outer(xb, delta_hid)

                # atualizacao: padrao = eta*g ; momentum soma alpha*atual_anterior
                dW2 = eta * gW2 + alpha * dW2_prev
                dW1 = eta * gW1 + alpha * dW1_prev
                self.W2 += dW2
                self.W1 += dW1
                dW2_prev, dW1_prev = dW2, dW1

            eqm = sq / (P * D.shape[1])
            eqms.append(eqm)
            if eqm_ant is not None and abs(eqm - eqm_ant) < eps:
                break
            eqm_ant = eqm

        tempo = time.time() - t0
        return eqms, len(eqms), tempo

    def predict(self, X):
        return np.array([self.forward(x)[0] for x in X])

    @staticmethod
    def pos_processar(Y, limiar=0.5):
        """Arredondamento simetrico: >=0.5 -> 1, senao 0. So no teste."""
        return (Y >= limiar).astype(int)
