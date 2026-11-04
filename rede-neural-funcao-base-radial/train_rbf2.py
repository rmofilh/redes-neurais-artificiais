"""RBF2 — aproximação da quantidade de gasolina (3 -> N1 -> 1).

Pipeline do exercicio RBF2:
  item 1: 3 treinamentos para cada topologia (N1 = 5, 10, 15)
  item 2: EQM final e numero de epocas de cada treinamento
  item 3: validacao no conjunto de teste -> erro relativo medio e variancia
  item 4: graficos de EQM por epoca (melhor treino de cada topologia)
  item 5: analise para escolher a topologia e o treinamento final

Uso:
    python3 train_rbf2.py

Saidas (em docs/answers/rbf2/):
    rbf2_treinamentos.csv, rbf2_centros.csv, rbf2_pesos.csv,
    rbf2_eqm_historico.csv, rbf2_teste_predicoes.csv, rbf2_erros_teste.csv,
    rbf2_eqm.png, rbf2_erro_relativo.png, rbf2_previsto_desejado.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rbf import (
    ativacao_gaussiana,
    eqm,
    kmeans,
    matriz_com_bias,
    treinar_camada_saida,
)

RAIZ = Path(__file__).resolve().parent
PASTA_DADOS = RAIZ / "data"
PASTA_RESPOSTAS = RAIZ / "docs" / "answers" / "rbf2"

TOPOLOGIAS = [
    {"codigo": "R1", "rede": "Rede 1", "n1": 5},
    {"codigo": "R2", "rede": "Rede 2", "n1": 10},
    {"codigo": "R3", "rede": "Rede 3", "n1": 15},
]
TREINOS = ["T1", "T2", "T3"]
SEEDS = [42, 43, 44]

SEED_KMEANS = 42
N_INIT = 10
MAX_ITER_KMEANS = 300
TOL_KMEANS = 1e-10

ETA = 0.01
EPS = 1e-7
MAX_EPOCAS = 200_000

CORES_REDE = {5: "#2b6cb0", 10: "#e07b39", 15: "#2f855a"}
COR_MELHOR = "#c53030"


# ----------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------
def carregar_dados():
    """Le os CSVs e devolve arrays prontos para o treino/teste."""
    treino = pd.read_csv(PASTA_DADOS / "treinamento2.csv")
    teste = pd.read_csv(PASTA_DADOS / "teste2.csv")

    X = treino[["x1", "x2", "x3"]].to_numpy(dtype=float)
    d = treino["d"].to_numpy(dtype=float)
    X_teste = teste[["x1", "x2", "x3"]].to_numpy(dtype=float)
    d_teste = teste["d"].to_numpy(dtype=float)

    # Checagens didaticas: se algo estiver errado, o erro aparece cedo.
    assert X.shape == (150, 3), f"treinamento inesperado: {X.shape}"
    assert X_teste.shape == (15, 3), f"teste inesperado: {X_teste.shape}"
    assert not treino.isna().any().any()
    assert not teste.isna().any().any()
    assert not treino.duplicated().any()
    assert not teste.duplicated().any()
    assert np.isfinite(X).all() and np.isfinite(d).all()
    assert np.isfinite(X_teste).all() and np.isfinite(d_teste).all()
    assert (d > 0).all() and (d_teste > 0).all()
    assert ((X >= 0) & (X <= 1)).all()
    assert ((X_teste >= 0) & (X_teste <= 1)).all()

    return treino, teste, X, d, X_teste, d_teste


# ----------------------------------------------------------------------
# Metricas de aproximacao
# ----------------------------------------------------------------------
def erro_relativo_absoluto_percentual(d, y):
    """|d - y| / d * 100, amostra por amostra.

    Todos os valores desejados deste exercicio sao positivos, portanto o
    erro relativo esta bem definido e tem unidade percentual.
    """
    d = np.asarray(d, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return np.abs((d - y) / d) * 100.0


def erro_relativo_assinado_percentual(d, y):
    """(d - y) / d * 100, usado apenas como diagnostico de vies.

    Valor positivo -> a rede subestima d; valor negativo -> superestima.
    """
    d = np.asarray(d, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return ((d - y) / d) * 100.0


def resumo_erro_relativo(d, y):
    """ERM, variancia populacional, desvio padrao e erro maximo."""
    erros = erro_relativo_absoluto_percentual(d, y)
    assinados = erro_relativo_assinado_percentual(d, y)
    return {
        "erros_relativos": erros,
        "erros_assinados": assinados,
        "erro_relativo_medio_pct": float(erros.mean()),
        "variancia_pct2": float(erros.var(ddof=0)),
        "desvio_padrao_pct": float(erros.std(ddof=0)),
        "erro_maximo_pct": float(erros.max()),
        "erro_medio_assinado_pct": float(assinados.mean()),
    }


# ----------------------------------------------------------------------
# Treino
# ----------------------------------------------------------------------
def treinar_topologia(X, d, X_teste, d_teste, topologia, verbose=True):
    """Executa k-means + os 3 treinos da camada de saida para uma topologia."""
    n1 = topologia["n1"]
    if verbose:
        print(f"  k-means com k={n1:2d} ...", end=" ", flush=True)

    km = kmeans(
        X, k=n1, seed=SEED_KMEANS, n_init=N_INIT,
        max_iter=MAX_ITER_KMEANS, tol=TOL_KMEANS,
    )
    Phi = ativacao_gaussiana(X, km["centros"], km["variancias"])
    H = matriz_com_bias(Phi)
    Phi_teste = ativacao_gaussiana(X_teste, km["centros"], km["variancias"])
    H_teste = matriz_com_bias(Phi_teste)

    # Referencia teorica: minimos quadrados com a mesma base e a mesma saida
    # linear. Como a saida e linear nos pesos, o problema e convexo.
    w_ls, *_ = np.linalg.lstsq(H, d, rcond=None)
    eqm_ls = eqm(d, H @ w_ls)

    treinos = []
    for treino_nome, seed in zip(TREINOS, SEEDS):
        w, hist, epocas, w_inicial = treinar_camada_saida(
            H, d, eta=ETA, eps=EPS, max_epocas=MAX_EPOCAS, seed=seed
        )
        y_treino = H @ w
        y_teste = H_teste @ w
        resumo = resumo_erro_relativo(d_teste, y_teste)

        # O treino deve parar pelo criterio de precisao, nao pelo teto.
        assert epocas < MAX_EPOCAS, (
            f"{topologia['rede']} {treino_nome} atingiu o teto de epocas"
        )
        assert abs(hist[-1] - hist[-2]) < EPS
        # O historico guarda o EQM calculado durante a ultima epoca; a
        # diferenca para o EQM dos pesos finais e pequena (~1e-5) e nao
        # altera a comparacao entre topologias.
        assert abs(eqm(d, y_treino) - hist[-1]) < 1e-3

        treinos.append({
            "treino": treino_nome,
            "seed": seed,
            "w": w,
            "w_inicial": w_inicial,
            "hist_eqm": hist,
            "epocas": epocas,
            "eqm_final": float(hist[-1]),
            "eqm_ls": float(eqm_ls),
            "y_treino": y_treino,
            "y_teste": y_teste,
            **resumo,
        })

    # Os tres treinos devem ter inicializacoes diferentes.
    for i in range(len(treinos)):
        for j in range(i + 1, len(treinos)):
            assert not np.allclose(
                treinos[i]["w_inicial"], treinos[j]["w_inicial"]
            ), "os pesos iniciais nao podem ser iguais"

    if verbose:
        inercia = km["inercia"]
        print(f"ok (inercia={inercia:.6f}, "
              f"epocas={[t['epocas'] for t in treinos]})")

    return {
        "codigo": topologia["codigo"],
        "rede": topologia["rede"],
        "n1": n1,
        "km": km,
        "H": H,
        "H_teste": H_teste,
        "w_ls": w_ls,
        "eqm_ls": float(eqm_ls),
        "treinos": treinos,
    }


def melhor_treino(topologia):
    """Menor ERM; desempate por variancia e EQM."""
    return min(
        topologia["treinos"],
        key=lambda t: (
            t["erro_relativo_medio_pct"],
            t["variancia_pct2"],
            t["eqm_final"],
        ),
    )


def melhor_global(resultados):
    """Melhor par (topologia, treino) entre os 9 treinos."""
    candidatos = [
        (topologia, treino)
        for topologia in resultados
        for treino in topologia["treinos"]
    ]
    return min(
        candidatos,
        key=lambda par: (
            par[1]["erro_relativo_medio_pct"],
            par[1]["variancia_pct2"],
            par[1]["eqm_final"],
        ),
    )


# ----------------------------------------------------------------------
# Tabelas
# ----------------------------------------------------------------------
def salvar_centros(resultados):
    linhas = []
    for topologia in resultados:
        km = topologia["km"]
        for j in range(topologia["n1"]):
            c = km["centros"][j]
            var = km["variancias"][j]
            n_j = int(km["n_por_cluster"][j])
            linhas.append({
                "rede": topologia["rede"],
                "n1": topologia["n1"],
                "cluster": j + 1,
                "n_padroes": n_j,
                "centro_x1": c[0],
                "centro_x2": c[1],
                "centro_x3": c[2],
                "variancia": var,
                "desvio_padrao": np.sqrt(var),
                "inercia_parcial": n_j * var,
                "inercia_total": km["inercia"],
                "iteracoes_kmeans": km["iteracoes"],
            })
    df = pd.DataFrame(linhas)
    df.to_csv(PASTA_RESPOSTAS / "rbf2_centros.csv", index=False)
    return df


def salvar_treinamentos(resultados):
    linhas = []
    for topologia in resultados:
        for t in topologia["treinos"]:
            linhas.append({
                "rede": topologia["rede"],
                "codigo": topologia["codigo"],
                "n1": topologia["n1"],
                "treino": t["treino"],
                "seed": t["seed"],
                "epocas": t["epocas"],
                "eqm_final": t["eqm_final"],
                "eqm_ls": t["eqm_ls"],
                "diferenca_eqm_pct": 100.0 * (t["eqm_final"] - t["eqm_ls"]) / t["eqm_ls"],
                "erro_relativo_medio_pct": t["erro_relativo_medio_pct"],
                "variancia_pct2": t["variancia_pct2"],
                "desvio_padrao_pct": t["desvio_padrao_pct"],
                "erro_maximo_pct": t["erro_maximo_pct"],
                "erro_medio_assinado_pct": t["erro_medio_assinado_pct"],
                "convergiu": bool(t["epocas"] < MAX_EPOCAS),
            })
    df = pd.DataFrame(linhas)
    df.to_csv(PASTA_RESPOSTAS / "rbf2_treinamentos.csv", index=False)
    return df


def salvar_pesos(resultados):
    linhas = []
    for topologia in resultados:
        for t in topologia["treinos"]:
            for j, (wi, wf) in enumerate(zip(t["w_inicial"], t["w"])):
                linhas.append({
                    "rede": topologia["rede"],
                    "n1": topologia["n1"],
                    "treino": t["treino"],
                    "seed": t["seed"],
                    "peso": f"W21,{j}",
                    "w_inicial": wi,
                    "w_final": wf,
                })
    df = pd.DataFrame(linhas)
    df.to_csv(PASTA_RESPOSTAS / "rbf2_pesos.csv", index=False)
    return df


def salvar_eqm_historico(resultados):
    linhas = []
    for topologia in resultados:
        for t in topologia["treinos"]:
            for epoca, valor in enumerate(t["hist_eqm"], start=1):
                linhas.append({
                    "rede": topologia["rede"],
                    "codigo": topologia["codigo"],
                    "n1": topologia["n1"],
                    "treino": t["treino"],
                    "seed": t["seed"],
                    "epoca": epoca,
                    "eqm": valor,
                })
    df = pd.DataFrame(linhas)
    df.to_csv(PASTA_RESPOSTAS / "rbf2_eqm_historico.csv", index=False)
    return df


def salvar_teste(resultados, teste_df, X_teste, d_teste):
    dados = {
        "amostra": teste_df["amostra"].to_numpy(),
        "x1": X_teste[:, 0],
        "x2": X_teste[:, 1],
        "x3": X_teste[:, 2],
        "d": d_teste,
    }
    for topologia in resultados:
        for t in topologia["treinos"]:
            dados[f"y_{topologia['codigo']}_{t['treino']}"] = t["y_teste"]
    predicoes = pd.DataFrame(dados)
    predicoes.to_csv(PASTA_RESPOSTAS / "rbf2_teste_predicoes.csv", index=False)

    linhas = []
    for topologia in resultados:
        for t in topologia["treinos"]:
            for i in range(len(d_teste)):
                linhas.append({
                    "rede": topologia["rede"],
                    "codigo": topologia["codigo"],
                    "n1": topologia["n1"],
                    "treino": t["treino"],
                    "seed": t["seed"],
                    "amostra": int(teste_df["amostra"].iloc[i]),
                    "x1": X_teste[i, 0],
                    "x2": X_teste[i, 1],
                    "x3": X_teste[i, 2],
                    "d": d_teste[i],
                    "y": t["y_teste"][i],
                    "erro_relativo_pct": t["erros_relativos"][i],
                    "erro_assinado_pct": t["erros_assinados"][i],
                })
    erros = pd.DataFrame(linhas)
    erros.to_csv(PASTA_RESPOSTAS / "rbf2_erros_teste.csv", index=False)
    return predicoes, erros


# ----------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------
def figura_eqm(resultados, caminho):
    """Tres graficos de EQM por epoca, um por topologia (item 4)."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharey=True)

    for ax, topologia in zip(axes, resultados):
        t = melhor_treino(topologia)
        epocas = np.arange(1, len(t["hist_eqm"]) + 1)
        cor = CORES_REDE[topologia["n1"]]

        ax.plot(epocas, t["hist_eqm"], color=cor, lw=1.7)
        ax.axhline(t["eqm_final"], color="gray", ls=":", lw=1.0)
        ax.annotate(
            f"{t['treino']}: {t['epocas']} épocas\nEQM = {t['eqm_final']:.6f}",
            xy=(t["epocas"], t["eqm_final"]),
            xytext=(-112, 28), textcoords="offset points",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.9),
        )
        ax.set_title(f"{topologia['rede']} (N1={topologia['n1']})", fontsize=11)
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, t["epocas"] * 1.04)

    axes[0].set_ylabel("EQM (erro quadrático médio)")
    fig.suptitle(
        "RBF2 — EQM por época (melhor treino de cada topologia)",
        fontsize=12.5,
    )
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figura_erro_relativo(resultados, caminho):
    """Barras do ERM por treino/topologia, com hastes de desvio padrao."""
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    largura = 0.25
    centros_x = np.arange(len(resultados))
    deslocamentos = [-largura, 0.0, largura]
    tons = ["#1a4971", "#2b6cb0", "#7fb3e0"]  # T1, T2, T3

    for k, treino_nome in enumerate(TREINOS):
        valores = []
        desvios = []
        for topologia in resultados:
            t = next(t for t in topologia["treinos"] if t["treino"] == treino_nome)
            valores.append(t["erro_relativo_medio_pct"])
            desvios.append(t["desvio_padrao_pct"])
        x = centros_x + deslocamentos[k]
        barras = ax.bar(
            x, valores, width=largura, color=tons[k],
            edgecolor="white", linewidth=0.6,
            yerr=[np.zeros_like(desvios), desvios], capsize=3.5,
            error_kw=dict(ecolor="#444444", lw=1.0),
            label=treino_nome,
        )
        for barra, valor in zip(barras, valores):
            ax.text(
                barra.get_x() + barra.get_width() / 2.0,
                barra.get_height() + 0.45,
                f"{valor:.2f}",
                ha="center", va="bottom", fontsize=7.8,
            )

    ax.set_xticks(centros_x)
    ax.set_xticklabels([f"{t['rede']}\n(N1={t['n1']})" for t in resultados])
    ax.set_ylabel("Erro relativo médio (%)")
    ax.set_title(
        "RBF2 — erro relativo médio no teste\n"
        "(barras: média; hastes: desvio padrão dos 15 erros)"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Treinamento", framealpha=0.95)
    ax.set_ylim(0, max(
        t["erro_relativo_medio_pct"] + t["desvio_padrao_pct"]
        for topologia in resultados for t in topologia["treinos"]
    ) * 1.22)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figura_previsto_desejado(resultados, teste_df, d_teste, caminho):
    """Dispersao y x d do melhor treino global, com a reta identidade."""
    topologia, t = melhor_global(resultados)
    y = t["y_teste"]

    fig, ax = plt.subplots(figsize=(6.9, 6.2))
    limite_min = min(d_teste.min(), y.min()) - 0.035
    limite_max = max(d_teste.max(), y.max()) + 0.035

    ax.plot([limite_min, limite_max], [limite_min, limite_max],
            color="#777777", ls="--", lw=1.2, label="y = d (ideal)")
    dispersao = ax.scatter(
        d_teste, y, c=t["erros_relativos"], cmap="viridis",
        s=70, edgecolors="white", linewidths=0.7, zorder=3,
    )
    for i, amostra in enumerate(teste_df["amostra"]):
        ax.annotate(
            str(int(amostra)), xy=(d_teste[i], y[i]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=8, color="#333333",
        )
    barra = fig.colorbar(dispersao, ax=ax, pad=0.02)
    barra.set_label("Erro relativo absoluto (%)")

    ax.set_xlim(limite_min, limite_max)
    ax.set_ylim(limite_min, limite_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("d (valor desejado)")
    ax.set_ylabel("y (valor previsto pela rede)")
    ax.set_title(
        f"RBF2 — previsto × desejado\n"
        f"{topologia['rede']} (N1={topologia['n1']}), {t['treino']}: "
        f"ERM = {t['erro_relativo_medio_pct']:.2f}%"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return topologia, t


# ----------------------------------------------------------------------
# Execucao principal
# ----------------------------------------------------------------------
def main():
    PASTA_RESPOSTAS.mkdir(parents=True, exist_ok=True)

    treino_df, teste_df, X, d, X_teste, d_teste = carregar_dados()
    print("=" * 78)
    print("RBF2 — aproximacao da quantidade de gasolina (3 -> N1 -> 1)")
    print("=" * 78)
    print(f"Treino: {len(X)} padroes | Teste: {len(X_teste)} padroes")
    print(f"Entradas: x1, x2, x3 | Saida: y | Topologias: N1 = 5, 10, 15")
    print()

    print("--- Camada escondida (k-means em todos os padroes de treino) ---")
    resultados = []
    for topologia in TOPOLOGIAS:
        resultados.append(
            treinar_topologia(X, d, X_teste, d_teste, topologia)
        )
    print()

    print("--- Salvando tabelas ---")
    centros_df = salvar_centros(resultados)
    treinos_df = salvar_treinamentos(resultados)
    pesos_df = salvar_pesos(resultados)
    eqm_df = salvar_eqm_historico(resultados)
    predicoes_df, erros_df = salvar_teste(
        resultados, teste_df, X_teste, d_teste
    )
    print(f"  rbf2_centros.csv          ({len(centros_df)} linhas)")
    print(f"  rbf2_treinamentos.csv     ({len(treinos_df)} linhas)")
    print(f"  rbf2_pesos.csv            ({len(pesos_df)} linhas)")
    print(f"  rbf2_eqm_historico.csv    ({len(eqm_df)} linhas)")
    print(f"  rbf2_teste_predicoes.csv  ({len(predicoes_df)} linhas)")
    print(f"  rbf2_erros_teste.csv      ({len(erros_df)} linhas)")
    print()

    print("--- Gerando figuras ---")
    figura_eqm(resultados, PASTA_RESPOSTAS / "rbf2_eqm.png")
    figura_erro_relativo(resultados, PASTA_RESPOSTAS / "rbf2_erro_relativo.png")
    topologia_best, treino_best = figura_previsto_desejado(
        resultados, teste_df, d_teste,
        PASTA_RESPOSTAS / "rbf2_previsto_desejado.png",
    )
    print("  rbf2_eqm.png")
    print("  rbf2_erro_relativo.png")
    print("  rbf2_previsto_desejado.png")
    print()

    # ---------------- resumo no terminal ----------------
    print("--- Item 2: EQM final e epocas ---")
    resumo_treinos = treinos_df[[
        "rede", "treino", "seed", "epocas", "eqm_final",
    ]].copy()
    print(resumo_treinos.to_string(
        index=False,
        float_format=lambda v: f"{v:.8f}",
    ))
    print()

    print("--- Item 3: erro relativo medio e variancia no teste ---")
    resumo_teste = treinos_df[[
        "rede", "treino", "erro_relativo_medio_pct", "variancia_pct2",
        "desvio_padrao_pct", "erro_maximo_pct",
    ]].copy()
    print(resumo_teste.to_string(
        index=False,
        float_format=lambda v: f"{v:.4f}",
    ))
    print()

    print("--- k-means: resumo dos clusters ---")
    for topologia in resultados:
        km = topologia["km"]
        sigmas = np.sqrt(km["variancias"])
        print(f"  {topologia['rede']} (N1={topologia['n1']}): "
              f"inercia={km['inercia']:.6f}, "
              f"tamanhos={km['n_por_cluster'].tolist()}, "
              f"sigma_min={sigmas.min():.4f}, "
              f"sigma_med={np.median(sigmas):.4f}, "
              f"sigma_max={sigmas.max():.4f}")
    print()

    print("--- Melhor treino de cada topologia (item 4) ---")
    for topologia in resultados:
        t = melhor_treino(topologia)
        print(f"  {topologia['rede']} (N1={topologia['n1']}): "
              f"{t['treino']} (seed {t['seed']}), "
              f"{t['epocas']} epocas, EQM={t['eqm_final']:.8f}, "
              f"ERM={t['erro_relativo_medio_pct']:.4f}%")
    print()

    print("--- Melhor global (item 5) ---")
    print(f"  {topologia_best['rede']} (N1={topologia_best['n1']}), "
          f"{treino_best['treino']} (seed {treino_best['seed']})")
    print(f"  Epocas: {treino_best['epocas']} | "
          f"EQM: {treino_best['eqm_final']:.8f}")
    print(f"  ERM: {treino_best['erro_relativo_medio_pct']:.4f}% | "
          f"Variancia: {treino_best['variancia_pct2']:.4f} (%^2) | "
          f"DP: {treino_best['desvio_padrao_pct']:.4f}% | "
          f"Erro maximo: {treino_best['erro_maximo_pct']:.4f}%")
    print()

    print("--- Maiores erros do melhor treino ---")
    ordem = np.argsort(-treino_best["erros_relativos"])
    for i in ordem[:5]:
        print(f"  amostra {int(teste_df['amostra'].iloc[i]):2d}: "
              f"d={d_teste[i]:.4f}, y={treino_best['y_teste'][i]:.4f}, "
              f"erro={treino_best['erros_assinados'][i]:+.2f}% "
              f"(|erro|={treino_best['erros_relativos'][i]:.2f}%)")
    print()
    print("Arquivos salvos em:", PASTA_RESPOSTAS)


if __name__ == "__main__":
    main()
