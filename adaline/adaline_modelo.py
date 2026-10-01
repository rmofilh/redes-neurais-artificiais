import numpy as np


class Adaline:
    def __init__(self, taxa_aprendizado=0.0025, max_epocas=1000, precisao=1e-6):
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = max_epocas  # teto: para antes por variacao do EQM
        self.precisao = precisao
        self.pesos = None  # w1, w2, w3, w4
        self.bias = None  # w0 (= theta), usado com x0 = -1
        self.pesos_iniciais = None
        self.bias_inicial = None
        self.n_epocas = 0
        self.historico_eqm = []  # EQM por epoca (item 3: grafico)

    def ativacao(self, valor):
        resultado = np.where(np.asarray(valor) >= 0, 1, -1)
        # preserva retorno escalar como int para uso no teste
        if np.ndim(resultado) == 0:
            return int(resultado)
        return resultado

    def treinar(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_amostras, n_features = X.shape

        # Inicializacao fora da classe: so gera se ainda nao foi setado.
        if self.pesos is None or self.bias is None:
            self.pesos = np.random.uniform(0, 1, n_features)
            self.bias = float(np.random.uniform(0, 1))

        self.pesos_iniciais = self.pesos.copy()
        self.bias_inicial = float(self.bias)
        self.historico_eqm = []

        for epoca in range(1, self.epocas + 1):
            # Regra Delta (LMS) online: atualiza a cada padrao com erro linear (d - u)
            for xi, d in zip(X, y):
                u = -self.bias + np.dot(xi, self.pesos)
                erro = d - u
                self.pesos += self.taxa_aprendizado * erro * xi
                self.bias += self.taxa_aprendizado * erro * (-1.0)

            # EQM da epoca com os pesos atualizados
            u_all = -self.bias + X @ self.pesos
            eqm = float(np.mean((y - u_all) ** 2))
            self.historico_eqm.append(eqm)

            if epoca > 1 and abs(self.historico_eqm[-1] - self.historico_eqm[-2]) < self.precisao:
                self.n_epocas = epoca
                break
        else:
            self.n_epocas = self.epocas

        return self.n_epocas, self.pesos, self.bias

    def prever(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        nets = -self.bias + X @ self.pesos
        return np.where(nets >= 0, 1, -1)
