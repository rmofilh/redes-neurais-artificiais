"""Treino PMC2: 4-15-3, sigmoid, eta=0.1, eps=1e-6.
Item 1: backprop padrao. Item 2: mesmo W0 + momentum alpha=0.9.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pmc import PMC

SEED = 42
ETA = 0.1
ALPHA = 0.9
EPS = 1e-6
MAX_EP = 5000
N_HIDDEN = 15  # figura do PDF: camada oculta vai ate 15

# --- dados ---
train = pd.read_csv("data/treinamento.csv")
test = pd.read_csv("data/teste.csv")
Xtr = train[["x1", "x2", "x3", "x4"]].to_numpy()
Dtr = train[["d1", "d2", "d3"]].to_numpy()
Xte = test[["x1", "x2", "x3", "x4"]].to_numpy()
Dte = test[["d1", "d2", "d3"]].to_numpy()

# --- mesmo W0 nos dois treinos: gera uma vez e copia ---
base = PMC(4, N_HIDDEN, 3, seed=SEED)
W10, W20 = base.W1.copy(), base.W2.copy()

# Item 1: padrao
rede1 = PMC(4, N_HIDDEN, 3, seed=SEED)
rede1.W1, rede1.W2 = W10.copy(), W20.copy()
eqm1, ep1, t1 = rede1.train(Xtr, Dtr, eta=ETA, alpha=0.0, eps=EPS, max_epocas=MAX_EP)

# Item 2: momentum (mesmo W0)
rede2 = PMC(4, N_HIDDEN, 3, seed=SEED)
rede2.W1, rede2.W2 = W10.copy(), W20.copy()
eqm2, ep2, t2 = rede2.train(Xtr, Dtr, eta=ETA, alpha=ALPHA, eps=EPS, max_epocas=MAX_EP)

print(f"Padrao:   epocas={ep1} EQM_final={eqm1[-1]:.8f} tempo={t1:.2f}s")
print(f"Momentum: epocas={ep2} EQM_final={eqm2[-1]:.8f} tempo={t2:.2f}s")

# --- grafico lado a lado (nao superpostos), mesma folha ---
# Refinado: mesma escala Y + grade + escala log para evidenciar a cauda
# (em escala linear 0-0.66 a diferenca 0.020 vs 0.015 some; no log ela aparece).
OUT = "docs/answers/pmc2"
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for a in ax:
    a.set_yscale("log")
    a.grid(True, which="major", alpha=0.6, linewidth=0.8)
    a.grid(True, which="minor", alpha=0.3, linewidth=0.5)
    a.set_axisbelow(True)
ax[0].plot(eqm1, color="tab:blue", linewidth=1.4, label=f"EQM final {eqm1[-1]:.5f}")
ax[0].axhline(eqm1[-1], color="tab:blue", linestyle="--", linewidth=1, alpha=0.7)
ax[0].set_title(f"Padrao (eta=0.1)\n{ep1} epocas, EQM={eqm1[-1]:.6f}, {t1:.1f}s")
ax[0].set_xlabel("Epoca"); ax[0].set_ylabel("EQM (escala log)")
ax[0].legend(fontsize=8, loc="upper right")
ax[1].plot(eqm2, color="tab:orange", linewidth=1.4, label=f"EQM final {eqm2[-1]:.5f}")
ax[1].axhline(eqm2[-1], color="tab:orange", linestyle="--", linewidth=1, alpha=0.7)
ax[1].set_title(f"Momentum (eta=0.1, alfa=0.9)\n{ep2} epocas, EQM={eqm2[-1]:.6f}, {t2:.1f}s")
ax[1].set_xlabel("Epoca"); ax[1].set_ylabel("EQM (escala log)")
ax[1].legend(fontsize=8, loc="upper right")
fig.suptitle("PMC2 4-15-3 — EQM por epoca (mesmo W0, grade + eixo log)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/eqm_pmc2.png", dpi=150)
print(f"Grafico: {OUT}/eqm_pmc2.png")

# --- validacao (itens 3 e 4): pos-processamento + taxa de acerto ---
def avalia(rede, nome):
    Y = rede.predict(Xte)
    Ypos = PMC.pos_processar(Y)
    # acerto = padrao correto apenas se as 3 saidas baterem (A=100, B=010, C=001)
    ok = (Ypos == Dte).all(axis=1).sum()
    acc = 100 * ok / len(Dte)
    print(f"\n[{nome}] acertos {ok}/{len(Dte)} = {acc:.2f}%")
    for i in range(len(Dte)):
        print(f"  {i+1:2d} y=[{Y[i,0]:.3f} {Y[i,1]:.3f} {Y[i,2]:.3f}] "
              f"pos=[{Ypos[i,0]} {Ypos[i,1]} {Ypos[i,2]}] des=[{Dte[i,0]} {Dte[i,1]} {Dte[i,2]}] "
              f"{'OK' if (Ypos[i]==Dte[i]).all() else 'ERRO'}")
    return Y, Ypos, acc

Y1, Yp1, acc1 = avalia(rede1, "Padrao")
Y2, Yp2, acc2 = avalia(rede2, "Momentum")

# salva numeros para o relatorio md
np.savetxt("docs/answers/pmc2/pmc2_teste_padrao.csv",
           np.hstack([Xte, Dte, Y1, Yp1]), delimiter=",",
           header="x1,x2,x3,x4,d1,d2,d3,y1,y2,y3,yp1,yp2,yp3", comments="", fmt="%.4f")
np.savetxt("docs/answers/pmc2/pmc2_teste_momentum.csv",
           np.hstack([Xte, Dte, Y2, Yp2]), delimiter=",",
           header="x1,x2,x3,x4,d1,d2,d3,y1,y2,y3,yp1,yp2,yp3", comments="", fmt="%.4f")
print(f"\nTaxa de acerto padrao={acc1:.2f}% momentum={acc2:.2f}%")
