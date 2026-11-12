"""Rede de Kohonen — mapa auto-organizavel (trabalho: 16 neuronios, grade 4x4).

Formulacao conforme o enunciado:
- Vencedor: neuronio j* com menor norma euclidiana ||W_j - x||.
- Grade topologica 4x4 com neuronios numerados 1..16 linha a linha.
- Vizinhanca de raio 1: distancia euclidiana entre as COORDENADAS do grid
  <= 1, ou seja, o vencedor + os 4 vizinhos ortogonais (norte/sul/leste/oeste).
- Atualizacao (todos os neuronios do conjunto vizinho, h = 1):
      W_j <- W_j + eta * (x - W_j),   eta = 0.001 fixa (sem decaimento)
- A regra acima e o gradiente descendente do erro quadratico
      E(W_j) = 1/2 * soma_i (w_ji - x_i)^2
  do neuronio vencedor (demonstracao no resposta.md, item 3).
- Epoca: as 120 amostras sao apresentadas NA ORDEM do arquivo. A cada epoca a
  rede registra o erro medio de quantizacao
      EQ = media das 1/2 ||W_j* - x||^2
  e a maior variacao de peso ocorrida na epoca (max |Delta w|).
- Parada: max |Delta w| da epoca < tolerancia, ou o teto de epocas. Com eta
  FIXA os pesos oscilam em torno dos centroides das suas regioes com amplitude
  da ordem de eta*EQ; o teto garante o termino (comportamento discutido no
  resposta.md junto da curva de convergencia).
"""
import numpy as np


class Kohonen:
    """Mapa auto-organizavel de Kohonen com vizinhanca de raio 1 no grid."""

    def __init__(self, eta=0.001, linhas=4, colunas=4, tolerancia=1e-6, max_epocas=20000):
        self.eta = eta  # taxa de aprendizado FIXA (o enunciado nao pede decaimento)
        self.linhas = linhas
        self.colunas = colunas
        self.n_neuronios = linhas * colunas
        self.tolerancia = tolerancia  # parada: max |Delta w| da epoca
        self.max_epocas = max_epocas  # teto de seguranca
        self.W = None  # matriz de pesos (16, 3)
        self.epocas = 0  # epocas executadas ate parar
        self.criterio_atingido = False  # True se parou por max |Delta w| < tolerancia
        self.eq_por_epoca = []  # EQ (erro medio de quantizacao) por epoca
        self.maxdw_por_epoca = []  # maior |Delta w| de cada epoca
        self.mudancas_por_epoca = []  # amostras que trocaram de vencedor vs epoca anterior
        self.ultima_mudanca = 0  # ultima epoca em que alguma amostra trocou de vencedor

    # ---------------- geometria do grid ----------------
    def coordenadas(self, j):
        """Coordenada (linha, coluna) do neuronio j (0-indexado) no grid."""
        return j // self.colunas, j % self.colunas

    def vizinhanca(self, j):
        """Indices (0..15) dos neuronios a distancia euclidiana <= 1 de j no grid.

        Inclui o proprio vencedor (distancia 0): vencedor + 4 ortogonais.
        """
        lin, col = self.coordenadas(j)
        conjunto = []
        for k in range(self.n_neuronios):
            l2, c2 = self.coordenadas(k)
            if (lin - l2) ** 2 + (col - c2) ** 2 <= 1:
                conjunto.append(k)
        return conjunto

    # ---------------- dinamica ----------------
    def vencedor(self, x):
        """Indice (0..15) do neuronio com menor norma euclidiana ||W_j - x||."""
        distancias = np.linalg.norm(self.W - np.asarray(x, dtype=float), axis=1)
        return int(np.argmin(distancias))

    def treinar(self, X, W0):
        """Treina apresentando as amostras de X (em ordem) por epocas completas.

        W0: pesos iniciais (16, 3) — sorteados FORA da classe, no script de
        treino (U[0,1] com np.random.seed(seed)), padrao da casa.
        Para quando max |Delta w| da epoca < tolerancia ou ao atingir o teto.
        """
        X = np.asarray(X, dtype=float)
        self.W = np.asarray(W0, dtype=float).copy()

        # conjuntos de vizinhanca pre-computados (fixos: raio constante)
        conjuntos = [np.array(self.vizinhanca(j)) for j in range(self.n_neuronios)]

        self.epocas = 0
        self.criterio_atingido = False
        self.eq_por_epoca = []
        self.maxdw_por_epoca = []
        self.mudancas_por_epoca = []
        self.ultima_mudanca = 0
        atribuicao_anterior = None

        for epoca in range(1, self.max_epocas + 1):
            max_dw = 0.0  # maior variacao de peso desta epoca
            soma_eq = 0.0  # soma do erro de quantizacao das amostras
            atribuicao = []
            for x in X:
                dist = np.linalg.norm(self.W - x, axis=1)  # ||W_j - x|| dos 16
                j = int(np.argmin(dist))  # vencedor: menor norma euclidiana
                soma_eq += 0.5 * float(dist[j] * dist[j])  # 1/2 ||W_j* - x||^2
                atribuicao.append(j)
                conjunto = conjuntos[j]
                # regra da norma euclidiana (h = 1): W <- W + eta*(x - W)
                # o proprio delta e a variacao de peso usada no criterio de parada
                delta = self.eta * (x - self.W[conjunto])
                self.W[conjunto] += delta
                dw = np.abs(delta).max()
                if dw > max_dw:
                    max_dw = float(dw)

            self.eq_por_epoca.append(soma_eq / X.shape[0])
            self.maxdw_por_epoca.append(max_dw)
            self.epocas = epoca
            atribuicao = np.array(atribuicao)
            if atribuicao_anterior is not None:
                n_mudou = int((atribuicao != atribuicao_anterior).sum())
                if n_mudou:
                    self.ultima_mudanca = epoca
            else:
                n_mudou = 0
            self.mudancas_por_epoca.append(n_mudou)
            atribuicao_anterior = atribuicao

            if max_dw < self.tolerancia:
                self.criterio_atingido = True
                break
        return self.W

    # ---------------- item 1: mapeamento das classes ----------------
    def mapear_rotulos(self, X, rotulos):
        """Conta, para cada neuronio, quantas vezes vence amostras de cada classe.

        Devolve (dominante, contagens):
          dominante[j]  -> classe com mais vitorias no neuronio j ('-' se o
                           neuronio nunca venceu; empate resolvido em ordem
                           alfabetica, escolha documentada no resposta.md)
          contagens[j]  -> dicionario {classe: numero de vitorias}
        """
        X = np.asarray(X, dtype=float)
        contagens = [{} for _ in range(self.n_neuronios)]
        for x, rotulo in zip(X, rotulos):
            j = self.vencedor(x)
            contagens[j][rotulo] = contagens[j].get(rotulo, 0) + 1
        dominante = []
        for cont in contagens:
            if cont:
                dominante.append(max(sorted(cont), key=cont.get))
            else:
                dominante.append("-")
        return dominante, contagens


# ---------------- utilidades de visualizacao ----------------
def mapa_texto(dominante, linhas=4, colunas=4):
    """Mapa de classes do grid em texto (4x4), uma letra por neuronio."""
    celulas = [
        f"{dominante[r * colunas + c]:>2}" for r in range(linhas) for c in range(colunas)
    ]
    return "\n".join(
        " ".join(celulas[r * colunas:(r + 1) * colunas]) for r in range(linhas)
    )
