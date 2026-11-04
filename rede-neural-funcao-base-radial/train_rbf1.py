"""RBF1 — classificacao de radiacao (2 entradas -> 2 gaussianas -> 1 saida).

Pipeline do exercicio RBF1:
  item 1: k-means apenas nos padroes com d = +1  -> centros e variancias
  item 2: regra Delta na camada de saida         -> pesos W21,0..2
  item 3: pos-processamento pela funcao sinal
  item 4: validacao no conjunto de teste         -> taxa de acerto
  item 5: material de analise para as estrategias de melhoria

Uso:
    python3 train_rbf1.py

Saidas (em docs/answers/rbf1/):
    rbf1_centros.csv, rbf1_pesos.csv, rbf1_treino_eqm.csv,
    rbf1_clusters.csv, rbf1_teste.csv,
    rbf1_clusters.png, rbf1_eqm.png, rbf1_fronteira.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Patch

from rbf import (
    acuracia,
    ativacao_gaussiana,
    eqm,
    kmeans,
    matriz_com_bias,
    matriz_confusao,
    pos_processar,
    treinar_camada_saida,
)

RAIZ = Path(__file__).resolve().parent
PASTA_DADOS = RAIZ / "data"
PASTA_RESPOSTAS = RAIZ / "docs" / "answers" / "rbf1"

SEED_KMEANS = 42
SEED_DELTA = 42
N_INIT = 10
ETA = 0.01
EPS = 1e-7
MAX_EPOCAS = 200_000

COR_POS = "#2b6cb0"    # azul  -> presenca de radiacao (d = +1)
COR_NEG = "#e07b39"    # laranja -> ausencia (d = -1)
COR_ERRO = "#c53030"   # vermelho -> erro de classificacao


# ----------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------
def carregar_dados():
    """Le os CSVs e devolve arrays prontos para o treino/teste."""
    treino = pd.read_csv(PASTA_DADOS / "treinamento.csv")
    teste = pd.read_csv(PASTA_DADOS / "teste.csv")

    X = treino[["x1", "x2"]].to_numpy(dtype=float)
    d = treino["d"].to_numpy(dtype=float)
    X_teste = teste[["x1", "x2"]].to_numpy(dtype=float)
    d_teste = teste["d"].to_numpy(dtype=float)

    # Checagens didaticas: se algo estiver errado, o erro aparece cedo.
    assert X.shape == (40, 2), f"treinamento inesperado: {X.shape}"
    assert X_teste.shape == (10, 2), f"teste inesperado: {X_teste.shape}"
    assert set(np.unique(d)) == {-1.0, 1.0}
    assert set(np.unique(d_teste)) == {-1.0, 1.0}
    assert not treino.isna().any().any()
    assert not teste.isna().any().any()

    return treino, teste, X, d, X_teste, d_teste


# ----------------------------------------------------------------------
# Treino
# ----------------------------------------------------------------------
def treinar(X, d):
    """Executa as duas fases do treinamento e guarda tudo para o relatorio."""
    # ---------- item 1: camada escondida (k-means so nos positivos) ----------
    X_pos = X[d == 1.0]
    km = kmeans(X_pos, k=2, seed=SEED_KMEANS, n_init=N_INIT, max_iter=300)
    centros = km["centros"]
    variancias = km["variancias"]

    # ---------- item 2: camada de saida (regra Delta) ----------
    Phi = ativacao_gaussiana(X, centros, variancias)  # ativacoes dos 40 padroes
    H = matriz_com_bias(Phi)                          # x0 = -1 na 1a coluna
    w, hist_eqm, epocas, w_inicial = treinar_camada_saida(
        H, d, eta=ETA, eps=EPS, max_epocas=MAX_EPOCAS, seed=SEED_DELTA
    )

    # Referencia teorica: minimos quadrados com a mesma base e a mesma saida linear.
    # Como a saida e linear nos pesos, o problema e convexo; o Delta deve chegar perto.
    w_ls, *_ = np.linalg.lstsq(H, d, rcond=None)
    eqm_ls = eqm(d, H @ w_ls)

    return {
        "km": km,
        "centros": centros,
        "variancias": variancias,
        "Phi": Phi,
        "H": H,
        "w": w,
        "w_inicial": w_inicial,
        "hist_eqm": hist_eqm,
        "epocas": epocas,
        "w_ls": w_ls,
        "eqm_ls": eqm_ls,
    }


# ----------------------------------------------------------------------
# Avaliacao
# ----------------------------------------------------------------------
def avaliar(tr, X, d, X_teste, d_teste):
    """Calcula saidas, pos-processamento, acuracia e matriz de confusao."""
    # ----- treino (diagnostico, nao pedido no item 4) -----
    y_treino = tr["H"] @ tr["w"]
    y_treino_pos = pos_processar(y_treino)

    # ----- teste (item 4) -----
    Phi_teste = ativacao_gaussiana(X_teste, tr["centros"], tr["variancias"])
    H_teste = matriz_com_bias(Phi_teste)
    y_teste = H_teste @ tr["w"]
    y_teste_pos = pos_processar(y_teste)

    tp, fn, fp, tn = matriz_confusao(y_teste_pos, d_teste)

    return {
        "y_treino": y_treino,
        "y_treino_pos": y_treino_pos,
        "acertos_treino": int((y_treino_pos == d).sum()),
        "acuracia_treino": acuracia(y_treino_pos, d),
        "Phi_teste": Phi_teste,
        "H_teste": H_teste,
        "y_teste": y_teste,
        "y_teste_pos": y_teste_pos,
        "acertos_teste": int((y_teste_pos == d_teste).sum()),
        "acuracia_teste": acuracia(y_teste_pos, d_teste),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def exemplo_primeira_atualizacao(tr, d):
    """Reproduz a primeira atualizacao da regra Delta (para o relatorio).

    Usa o primeiro padrao do arquivo, os pesos iniciais e a mesma equacao
    w <- w + eta * (d - y) * h. Serve para mostrar o mecanismo passo a passo.
    """
    h = tr["H"][0]
    w0 = tr["w_inicial"]
    y = float(w0 @ h)
    e = float(d[0] - y)
    w_novo = w0 + ETA * e * h
    return {"h": h, "y": y, "e": e, "w_novo": w_novo}


# ----------------------------------------------------------------------
# Tabelas
# ----------------------------------------------------------------------
def salvar_tabelas(tr, av, treino_df, teste_df, X, d, X_teste, d_teste):
    """Salva os CSVs que sustentam as tabelas do relatorio."""
    PASTA_RESPOSTAS.mkdir(parents=True, exist_ok=True)

    # ----- item 1: centros e variancias -----
    cent = pd.DataFrame({
        "cluster": np.arange(1, len(tr["centros"]) + 1),
        "n_padroes": tr["km"]["n_por_cluster"],
        "centro_x1": tr["centros"][:, 0],
        "centro_x2": tr["centros"][:, 1],
        "variancia": tr["variancias"],
        "desvio_padrao": np.sqrt(tr["variancias"]),
    })
    cent["inercia_parcial"] = cent["n_padroes"] * cent["variancia"]
    cent.to_csv(PASTA_RESPOSTAS / "rbf1_centros.csv", index=False)

    # ----- item 2: pesos iniciais e finais -----
    nomes = ["W21,0", "W21,1", "W21,2"]
    pesos = pd.DataFrame({
        "peso": nomes,
        "w_inicial": tr["w_inicial"],
        "w_final": tr["w"],
    })
    pesos.to_csv(PASTA_RESPOSTAS / "rbf1_pesos.csv", index=False)

    # ----- historico de EQM (para o grafico) -----
    pd.DataFrame({
        "epoca": np.arange(1, len(tr["hist_eqm"]) + 1),
        "eqm": tr["hist_eqm"],
    }).to_csv(PASTA_RESPOSTAS / "rbf1_treino_eqm.csv", index=False)

    # ----- membros dos clusters (apenas os positivos) -----
    pos_df = treino_df[treino_df["d"] == 1].copy()
    pos_df["cluster"] = tr["km"]["rotulos"] + 1
    pos_df[["amostra", "x1", "x2", "cluster"]].to_csv(
        PASTA_RESPOSTAS / "rbf1_clusters.csv", index=False
    )

    # ----- item 4: tabela de teste completa -----
    teste_out = pd.DataFrame({
        "amostra": teste_df["amostra"],
        "x1": X_teste[:, 0],
        "x2": X_teste[:, 1],
        "d": d_teste,
        "phi1": av["Phi_teste"][:, 0],
        "phi2": av["Phi_teste"][:, 1],
        "y": av["y_teste"],
        "y_pos": av["y_teste_pos"].astype(int),
        "acerto": (av["y_teste_pos"] == d_teste).astype(int),
    })
    teste_out.to_csv(PASTA_RESPOSTAS / "rbf1_teste.csv", index=False)

    return cent, pesos, teste_out


# ----------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------
def figura_clusters(X, d, centros, variancias, caminho):
    """Dispersao dos padroes de treino + centros + circulos de sigma e 2 sigma."""
    fig, ax = plt.subplots(figsize=(8.4, 6.2))

    ax.scatter(X[d == -1, 0], X[d == -1, 1], s=44, c=COR_NEG,
               edgecolors="white", linewidths=0.6,
               label="Treino, d = -1 (ausência)", zorder=3)
    ax.scatter(X[d == 1, 0], X[d == 1, 1], s=44, c=COR_POS,
               edgecolors="white", linewidths=0.6,
               label="Treino, d = +1 (presença)", zorder=3)

    for j, (c, v) in enumerate(zip(centros, variancias), start=1):
        ax.scatter(c[0], c[1], marker="*", s=330, c="black",
                   edgecolors="white", linewidths=0.8, zorder=5,
                   label="Centros" if j == 1 else None)
        ax.annotate(f"C{j}", xy=c, xytext=(9, 8), textcoords="offset points",
                    fontsize=12, fontweight="bold")
        ax.add_patch(Circle(c, np.sqrt(v), fill=False, ls="--", lw=1.3,
                            color="black", alpha=0.6))
        ax.add_patch(Circle(c, 2.0 * np.sqrt(v), fill=False, ls=":", lw=1.1,
                            color="black", alpha=0.45))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("RBF1 — padrões de treino, centros e larguras\n"
                 "(tracejado: σ; pontilhado: 2σ)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=9, framealpha=0.92, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figura_eqm(hist, epocas, eqm_final, caminho):
    """Curva de EQM por epoca da regra Delta."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(np.arange(1, len(hist) + 1), hist, color=COR_POS, lw=1.8)
    ax.axhline(eqm_final, color="gray", ls=":", lw=1.1)
    ax.annotate(f"parada: {epocas} épocas\nEQM = {eqm_final:.6f}",
                xy=(epocas, eqm_final), xytext=(-125, 30),
                textcoords="offset points", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.0))
    ax.set_xlabel("Época")
    ax.set_ylabel("EQM (erro quadrático médio)")
    ax.set_title("RBF1 — convergência da regra Delta na camada de saída")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def figura_fronteira(X, d, X_teste, d_teste, y_teste_pos,
                     centros, variancias, w, caminho):
    """Fronteira de decisao y = 0, regioes previstas e pontos de teste."""
    xs = np.linspace(-0.02, 1.02, 260)
    ys = np.linspace(-0.02, 1.02, 260)
    XX, YY = np.meshgrid(xs, ys)
    grade = np.column_stack((XX.ravel(), YY.ravel()))
    Phi = ativacao_gaussiana(grade, centros, variancias)
    Z = (matriz_com_bias(Phi) @ w).reshape(XX.shape)

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    ax.contourf(XX, YY, Z, levels=[Z.min() - 1e-6, 0.0, Z.max() + 1e-6],
                colors=["#fde4d0", "#dbe9f7"], alpha=0.9)
    ax.contour(XX, YY, Z, levels=[0.0], colors="black",
               linewidths=1.7, linestyles="--")

    ax.scatter(X[d == -1, 0], X[d == -1, 1], s=26, c=COR_NEG, alpha=0.75,
               edgecolors="white", linewidths=0.4,
               label="Treino d = -1", zorder=3)
    ax.scatter(X[d == 1, 0], X[d == 1, 1], s=26, c=COR_POS, alpha=0.75,
               edgecolors="white", linewidths=0.4,
               label="Treino d = +1", zorder=3)

    for classe, cor in [(-1, COR_NEG), (1, COR_POS)]:
        mascara = d_teste == classe
        ax.scatter(X_teste[mascara, 0], X_teste[mascara, 1], marker="s",
                   s=82, facecolors="white", edgecolors=cor, linewidths=1.8,
                   label=f"Teste d = {classe:+d}", zorder=5)
    for i in range(len(X_teste)):
        ax.annotate(str(i + 1), xy=X_teste[i], xytext=(6, 5),
                    textcoords="offset points", fontsize=8,
                    color="#333333", zorder=6)

    erros = np.where(y_teste_pos != d_teste)[0]
    for k, i in enumerate(erros):
        ax.scatter(X_teste[i, 0], X_teste[i, 1], s=235, facecolors="none",
                   edgecolors=COR_ERRO, linewidths=2.2, zorder=7,
                   label="Erro de teste" if k == 0 else None)

    ax.scatter(centros[:, 0], centros[:, 1], marker="*", s=270, c="black",
               edgecolors="white", linewidths=0.8, zorder=8, label="Centros")

    handles = [
        Patch(facecolor="#dbe9f7", edgecolor="none", label="Região prevista +1"),
        Patch(facecolor="#fde4d0", edgecolor="none", label="Região prevista -1"),
        plt.Line2D([], [], color="black", ls="--", lw=1.5, label="Fronteira y = 0"),
    ]
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + h, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=8.5, framealpha=0.95, ncol=1, borderaxespad=0)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("RBF1 — fronteira de decisão e classificação do teste")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Execucao principal
# ----------------------------------------------------------------------
def main():
    PASTA_RESPOSTAS.mkdir(parents=True, exist_ok=True)

    treino_df, teste_df, X, d, X_teste, d_teste = carregar_dados()
    tr = treinar(X, d)
    av = avaliar(tr, X, d, X_teste, d_teste)
    cent, pesos, teste_out = salvar_tabelas(
        tr, av, treino_df, teste_df, X, d, X_teste, d_teste
    )

    figura_clusters(X, d, tr["centros"], tr["variancias"],
                    PASTA_RESPOSTAS / "rbf1_clusters.png")
    figura_eqm(tr["hist_eqm"], tr["epocas"], tr["hist_eqm"][-1],
               PASTA_RESPOSTAS / "rbf1_eqm.png")
    figura_fronteira(X, d, X_teste, d_teste, av["y_teste_pos"],
                     tr["centros"], tr["variancias"], tr["w"],
                     PASTA_RESPOSTAS / "rbf1_fronteira.png")

    # ---------------- resumo no terminal ----------------
    print("=" * 74)
    print("RBF1 — classificacao de radiacao (2 -> 2 -> 1)")
    print("=" * 74)
    print(f"Treino: {len(X)} padroes "
          f"({int((d == 1).sum())} positivos, {int((d == -1).sum())} negativos)")
    print(f"Teste:  {len(X_teste)} padroes "
          f"({int((d_teste == 1).sum())} positivos, {int((d_teste == -1).sum())} negativos)")
    print()

    print("--- Item 1: k-means apenas nos positivos (k=2) ---")
    for j, (c, v, n) in enumerate(zip(tr["centros"], tr["variancias"],
                                      tr["km"]["n_por_cluster"]), start=1):
        print(f"Cluster {j}: n={n:2d}  centro=({c[0]:.6f}, {c[1]:.6f})  "
              f"variancia={v:.6f}  sigma={np.sqrt(v):.6f}")
    print(f"Inercia total: {tr['km']['inercia']:.6f}")
    print(f"Iteracoes do melhor reinicio: {tr['km']['iteracoes']}")
    print(f"Inercias dos {N_INIT} reinicios: "
          f"{np.array2string(tr['km']['inercias_reinicios'], precision=6)}")
    print()

    print("--- Item 2: regra Delta na camada de saida ---")
    print(f"Pesos iniciais (U[0,1], seed={SEED_DELTA}): "
          f"{np.array2string(tr['w_inicial'], precision=6)}")
    print(f"Epocas: {tr['epocas']}   EQM final: {tr['hist_eqm'][-1]:.8f}")
    print(f"Pesos finais (x0=-1): W21,0={tr['w'][0]:.6f}  "
          f"W21,1={tr['w'][1]:.6f}  W21,2={tr['w'][2]:.6f}")
    print(f"Referencia LS:      W21,0={tr['w_ls'][0]:.6f}  "
          f"W21,1={tr['w_ls'][1]:.6f}  W21,2={tr['w_ls'][2]:.6f}  "
          f"(EQM={tr['eqm_ls']:.8f})")
    print()

    ex = exemplo_primeira_atualizacao(tr, d)
    print("Exemplo: primeira atualizacao (amostra 1, d=-1)")
    print(f"  h = [-1, phi1, phi2] = [-1, {ex['h'][1]:.6f}, {ex['h'][2]:.6f}]")
    print(f"  y = {ex['y']:.6f}   e = d - y = {ex['e']:.6f}")
    print(f"  w novo = {np.array2string(ex['w_novo'], precision=6)}")
    print()

    print("--- Item 4: teste ---")
    print(teste_out.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print()
    print(f"Acertos: {av['acertos_teste']}/{len(d_teste)}  "
          f"Taxa de acerto: {av['acuracia_teste']:.2f}%")
    print(f"Matriz de confusao: TP={av['tp']}  FN={av['fn']}  "
          f"FP={av['fp']}  TN={av['tn']}")
    print()

    print("--- Diagnostico no treino (nao pedido, mas util) ---")
    print(f"Acertos: {av['acertos_treino']}/{len(d)}  "
          f"Taxa: {av['acuracia_treino']:.2f}%")
    erros_treino = np.where(av["y_treino_pos"] != d)[0]
    for i in erros_treino:
        print(f"  erro: amostra {i + 1:2d}  x=({X[i, 0]:.4f}, {X[i, 1]:.4f})  "
              f"d={d[i]:+.0f}  y={av['y_treino'][i]:+.6f}")
    print()

    print("Arquivos salvos em:", PASTA_RESPOSTAS)


if __name__ == "__main__":
    main()
