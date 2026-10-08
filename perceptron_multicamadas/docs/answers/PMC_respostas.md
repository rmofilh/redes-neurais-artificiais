# PMC — Backpropagation com atalho Entrada → Saída (W3)

> Convenção do enunciado (`docs/PMC.pdf`):
> `W1` = 1ª → 2ª camada, `W2` = 2ª → 3ª camada, `W3` = 1ª → 3ª camada (atalho).
> Parâmetros adotados: `N1 = 5`, `eta = 0.01`, ativação **logística** nas duas camadas neurais.
> Dados: `N = 3` (`x1,x2,x3`), `P = 200` treino + `20` validação, todos em `0..1`.

## 1. Topologia e dimensões

```
Entrada (N)      Escondida (N1)     Saída (1)
X[1..N]  --W1--> h[1..N1] --W2-->  o
X[1..N]  ----------W3------------> o   (atalho, curvas grossas da figura)
```

| Matriz | De → Para | Dimensão | Significado |
|---|---|---|---|
| `W1` | entrada → escondida | `N x N1` | `W1[i][j]`: entrada `i` → oculto `j` |
| `W2` | escondida → saída | `N1` (vetor) | `W2[j]`: oculto `j` → saída |
| `W3` | entrada → saída | `N` (vetor) | `W3[i]`: entrada `i` → saída (atalho) |
| `b1` | bias escondida | `N1` | um threshold por neurônio oculto |
| `b2` | bias saída | escalar | um threshold na saída |

Hipótese explícita: o PDF não cita bias, mas sem bias a sigmoide nunca se desloca.
Usamos a mesma convenção do trabalho anterior (`x0 = -1`): `net = soma(w*x) - b`.

Com `N = 3, N1 = 5`: `W1[3x5]`, `W2[5]`, `W3[3]`, `b1[5]`, `b2[1]`.

## 2. Forward (propagação direta) — por padrão p

Dado padrão `X = [x1..xN]`, desejado `d`:

**Passo F1 — escondida:**
```
net_h[j] = soma_{i=1..N}( x[i] * W1[i][j] ) - b1[j]      , j = 1..N1
y_h[j]   = sigmoid(net_h[j]) = 1 / (1 + exp(-net_h[j]))
```

**Passo F2 — saída (soma das DUAS vias):**
```
net_o = soma_{j=1..N1}( y_h[j] * W2[j] ) + soma_{i=1..N}( x[i] * W3[i] ) - b2
o     = sigmoid(net_o)
```

**Passo F3 — erro instantâneo:**
```
E = 1/2 * (d - o)^2
```

Repare: `net_o` tem parcela via escondida (`W2`) + parcela direta (`W3`).
É isso que torna este PMC diferente do MLP clássico.

## 3. Backward (retropropagação) — dedução

Função logística: `f(net) = 1/(1+e^-net)`, derivada em função da saída: `f'(net) = y*(1-y)`.

Queremos gradiente descendente: `w_novo = w_velho - eta * dE/dw`.

### 3.1 Delta da saída

```
dE/do = -(d - o)
do/dnet_o = o*(1-o)
delta_o = -(dE/dnet_o) com sinal já embutido p/ subida = (d - o) * o*(1-o)
```

### 3.2 Gradientes que chegam direto na saída (W2, W3, b2)

```
dE/dW2[j] = dE/dnet_o * dnet_o/dW2[j] = -(delta_o) * y_h[j]
=> W2[j] += eta * delta_o * y_h[j]

dE/dW3[i] = -(delta_o) * x[i]
=> W3[i] += eta * delta_o * x[i]

dnet_o/db2 = -1  (pois net_o = ... - b2)
dE/db2 = (-delta_o)*(-1) = +delta_o
=> b2 -= eta * delta_o
```

Mesma lógica para `b1[j] -= eta * delta_h[j]` (seção 3.4).
Como `net = soma(w*x) - b`, o gradiente do bias tem sinal `+delta`,
logo a descida do gradiente subtrai.

### 3.3 Delta da escondida (o erro volta por W2)

O neurônio oculto `j` só influencia `E` através de `net_o`, via `W2[j]`:

```
dE/dnet_h[j] = dE/dnet_o * dnet_o/dy_h[j] * dy_h[j]/dnet_h[j]
             = -(delta_o) * W2[j] * y_h[j]*(1-y_h[j])

delta_h[j] = y_h[j]*(1-y_h[j]) * delta_o * W2[j]
```

**Ponto crítico didático:** usar o `W2[j]` **antigo** (antes da atualização do passo 3.2)
para calcular `delta_h`. Na prática a diferença é pequena com `eta = 0.01`,
mas o correto matemático é congelar os pesos do forward para o backward inteiro
e só depois aplicar todas as correções — ou ao menos calcular todos os deltas
antes de sobrescrever `W2`.

### 3.4 Gradientes da primeira camada (W1, b1)

```
W1[i][j] += eta * delta_h[j] * x[i]
b1[j]    += eta * delta_h[j]        (mesma convenção de sinal de b2)
```

`W3` **não** gera delta escondido: o atalho vai direto à saída, não passa por `h`.

## 4. Sequência e ordem de ajuste (resposta direta ao enunciado)

Para cada época, para cada padrão `p = 1..P` (ordem embaralhada a cada época):

```
1. F1: calcula net_h, y_h          (usa W1, b1)
2. F2: calcula net_o, o            (usa y_h, W2 + X, W3, b2)
3. B1: delta_o = (d - o)*o*(1-o)
4. B2: atualiza W2[j] += eta*delta_o*y_h[j]   (j=1..N1)
       atualiza W3[i] += eta*delta_o*x[i]     (i=1..N)
       atualiza b2
5. B3: delta_h[j] = y_h[j]*(1-y_h[j])*delta_o*W2_antigo[j]
6. B4: atualiza W1[i][j] += eta*delta_h[j]*x[i]
       atualiza b1[j]
7. Acumula E para o MSE da época.
```

Ordem de ajuste exigida: **W2 e W3 primeiro (saída), W1 por último (retropropagado).**
Nunca atualizar `W1` antes de calcular `delta_o`.

Por época: modo **estocástico (padrão-a-padrão)** — mais didático e converge melhor
que batch para `P = 200` com `eta = 0.01` pequeno.

## 5. Por que logística e por que eta = 0.01?

* Dados já em `0..1` → logística devolve `o` em `0..1`, compara direto com `d`.
  Tanh devolveria `-1..1` e exigiria normalizar `d_norm = 2*d - 1` e desnormalizar
  na hora de testar — trabalho extra sem ganho didático aqui.
* `eta = 0.01` pequeno + sigmoide (derivada máxima `0.25`) = passo estável.
  Com atalho `W3`, o gradiente da saída recebe duas parcelas; `eta` pequeno evita
  oscilação entre `W2` e `W3` disputando a mesma correção.

## 6. Mapeamento para o código

* `pmc/pmc_modelo.py` → `forward()` = F1+F2, `backward_step()` = B1..B4, `prever()` = só F1+F2.
* `pmc/pmc_treino.py` (Fase 2) → loop de épocas, MSE = média de `E*2` sobre `P`.
* `pmc/pmc_teste.py` (Fase 3) → só forward sobre `conjunto_validacao.csv`.
