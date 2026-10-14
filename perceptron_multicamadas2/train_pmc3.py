"""PMC3 (TDNN serie temporal): 9 treinos com backprop + momentum.
Redes: (p=5,N1=10), (p=10,N1=15), (p=15,N1=25). eta=0.1, alfa=0.8, eps=0.5e-6.
Seeds 42/43/44 = T1/T2/T3 (pesos U[0,1] diferentes por treino).
Reutiliza pmc.py (so numpy). Sem lib de rede neural.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pmc import PMC

ETA = 0.1
ALPHA = 0.8
EPS = 0.5e-6
MAX_EP = 5000
SEEDS = [42, 43, 44]
REDES = [(5, 10), (10, 15), (15, 25)]  # (p, N1)
OUT = "docs/answers/pmc3"

# --- serie ---
tr = pd.read_csv("data/treinamento(1).csv")["f(t)"].to_numpy()  # f(1..100)
te = pd.read_csv("data/teste(1).csv")["f(t)"].to_numpy()         # f(101..120)
full = np.concatenate([tr, te])  # indice 0 = t1 ... indice 119 = t120


def janelas(serie, p, idx_alvo):
    """Monta X (lags) e D para os indices-alvo (0-based) dados."""
    X = np.array([serie[t - p:t] for t in idx_alvo])
    D = np.array([serie[t] for t in idx_alvo]).reshape(-1, 1)
    return X, D


idx_tr = {p: list(range(p, 100)) for p, _ in REDES}  # treino: t=p+1..100 (1-based)
idx_te = list(range(100, 120))                        # teste: t=101..120

resultados = []   # uma linha por treino
prevs = {}        # (ri,ti) -> Y previsto no teste
eqms_hist = {}    # (ri,ti) -> lista eqm

for ri, (p, n1) in enumerate(REDES, start=1):
    Xtr, Dtr = janelas(full, p, idx_tr[p])
    Xte, Dte = janelas(full, p, idx_te)
    print(f"Rede {ri}: p={p} N1={n1} padroes_treino={len(Xtr)}")
    for ti, seed in enumerate(SEEDS, start=1):
        rede = PMC(p, n1, 1, seed=seed)
        eqms, ep, tempo = rede.train(Xtr, Dtr, eta=ETA, alpha=ALPHA,
                                    eps=EPS, max_epocas=MAX_EP)
        Y = rede.predict(Xte).ravel()
        d = Dte.ravel()
        rel = np.abs(d - Y) / np.abs(d)  # literal do enunciado
        resultados.append(dict(rede=ri, treino=ti, p=p, n1=n1, seed=seed,
                               epocas=ep, eqm=eqms[-1], tempo=tempo,
                               err_medio=rel.mean(), var=rel.var()))
        prevs[(ri, ti)] = Y
        eqms_hist[(ri, ti)] = eqms
        print(f"  T{ti} (seed {seed}): ep={ep} EQM={eqms[-1]:.8f} "
              f"tempo={tempo:.1f}s errMed={rel.mean():.4f} var={rel.var():.4f}")

res = pd.DataFrame(resultados)

# --- melhor treino por rede = menor erro relativo medio no teste ---
melhores = {ri: int(res[res.rede == ri].sort_values("err_medio").iloc[0]["treino"])
            for ri in (1, 2, 3)}
print("Melhores:", melhores)

# --- CSVs ---
res.to_csv(f"{OUT}/pmc3_tabela_treinos.csv", index=False)
prev_df = pd.DataFrame({"t": np.arange(101, 121), "desejado": te})
for ri in (1, 2, 3):
    for ti in (1, 2, 3):
        prev_df[f"r{ri}t{ti}"] = prevs[(ri, ti)]
prev_df.to_csv(f"{OUT}/pmc3_previsoes.csv", index=False, float_format="%.6f")
print(f"CSVs em {OUT}/")

# --- Fig item 4: EQM dos melhores (3 paineis, grade + log, nao superpostos) ---
fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=False)
for i, ri in enumerate((1, 2, 3)):
    ti = melhores[ri]
    e = np.array(eqms_hist[(ri, ti)])
    p, n1 = REDES[ri - 1]
    ax[i].plot(e, linewidth=1.4, label=f"EQM final {e[-1]:.6f}")
    ax[i].axhline(e[-1], linestyle="--", linewidth=1, alpha=0.7)
    if e.max() / max(e.min(), 1e-12) > 1.5:
        ax[i].set_yscale("log")
        ax[i].set_ylabel("EQM (log)")
        ax[i].grid(True, which="major", alpha=0.6, linewidth=0.8)
        ax[i].grid(True, which="minor", alpha=0.3, linewidth=0.5)
    else:  # curva travada (ex. rede saturada): log achata os ticks; usa linear
        ax[i].set_ylabel("EQM")
        ax[i].ticklabel_format(useOffset=False, style="plain")
        ax[i].grid(True, alpha=0.6, linewidth=0.8)
    ax[i].set_title(f"Rede {ri} (p={p}, N1={n1}) — melhor T{ti}: "
                    f"{len(e)} epocas, EQM={e[-1]:.6f}")
    ax[i].set_xlabel("Epoca")
    ax[i].legend(fontsize=8, loc="upper right")
fig.suptitle("PMC3 — EQM por epoca (melhor treino de cada rede)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/eqm_pmc3_melhores.png", dpi=150)

# --- Fig item 5: desejado x estimado t=101..120 (3 paineis) ---
t = np.arange(101, 121)
fig2, ax2 = plt.subplots(3, 1, figsize=(8, 9), sharex=True, sharey=True)
for i, ri in enumerate((1, 2, 3)):
    ti = melhores[ri]
    p, n1 = REDES[ri - 1]
    ax2[i].plot(t, te, marker="o", markersize=4, linewidth=1.4, label="desejado")
    ax2[i].plot(t, prevs[(ri, ti)], marker="s", markersize=4,
                linewidth=1.4, label=f"rede {ri} T{ti}")
    ax2[i].grid(True, alpha=0.5)
    ax2[i].set_title(f"Rede {ri} (p={p}, N1={n1}) T{ti} — "
                     f"errMed={res[(res.rede==ri)&(res.treino==ti)]['err_medio'].values[0]:.4f}")
    ax2[i].set_xlabel("t")
    ax2[i].set_ylabel("f(t)")
    ax2[i].legend(fontsize=9)
fig2.suptitle("PMC3 — desejado x estimado (t=101..120)", fontsize=12)
fig2.tight_layout()
fig2.savefig(f"{OUT}/prev_pmc3_melhores.png", dpi=150)
print(f"Figuras em {OUT}/")
