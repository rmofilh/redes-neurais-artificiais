import numpy as np


class Perceptron:
    def __init__(self, taxa_aprendizado=0.01, max_epocas=1000):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = max_epocas  # teto: para antes se zerar erros
        self.pesos = None  # w1, w2, w3
        self.bias = None  # w0 (= theta), usado com x0 = -1
        self.pesos_iniciais = None
        self.bias_inicial = None
        self.n_epocas = 0

    def ativacao(self, valor):
        resultado = np.where(np.asarray(valor) >= 0, 1, -1)
        # preserva retorno escalar como int para uso no treino
        if np.ndim(resultado) == 0:
            return int(resultado)
        return resultado

    def treinar(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n_amostras, n_features = X.shape

        # Inicializacao fora da classe (b) fora): so gera se ainda nao foi setado.
        if self.pesos is None or self.bias is None:
            self.pesos = np.random.uniform(0, 1, n_features)
            self.bias = float(np.random.uniform(0, 1))

        self.pesos_iniciais = self.pesos.copy()
        self.bias_inicial = float(self.bias)

        for epoca in range(1, self.epocas + 1):
            erros = 0
            for xi, d in zip(X, y):
                net = -self.bias + np.dot(xi, self.pesos)
                y_pred = self.ativacao(net)
                if y_pred != d:
                    correcao = self.taxa_aprendizado * (d - y_pred)
                    self.pesos += correcao * xi
                    self.bias += correcao * (-1.0)
                    erros += 1
            if erros == 0:
                self.n_epocas = epoca
                break
        else:
            self.n_epocas = self.epocas

        return self.n_epocas, self.pesos, self.bias

    def prever(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        nets = -self.bias + X @ self.pesos
        return np.where(nets >= 0, 1, -1)