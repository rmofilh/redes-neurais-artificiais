# Perceptron — Respostas (Óleo C1/C2)

## Item 1 — 5 treinamentos

- Modelo: `perceptron/perceptron_modelo.py` — 3 entradas + bias `x0 = -1`, `net = -w0 + x1*w1 + x2*w2 + x3*w3`, ativação bipolar `y = +1 se net >= 0 senão -1`.
- Regra supervisionada com `(d - y)`: se `y != d`, `w += eta*(d-y)*x` e `w0 += eta*(d-y)*(-1)`, com `eta = 0.01`.
- Inicialização fora da classe, `U[0,1]`, gerador reiniciado por seed: T1=10, T2=20, T3=30, T4=40, T5=50.
- Dados: `data/oleo_treinamento.csv` (30 padrões do anexo).
- Parada: erro zero em uma época completa (teto `max_epocas = 1000`).
- Execução: `python3 -m perceptron.perceptron_treino` (`executar_treinamentos()`).

## Item 2 — Vetores de pesos e épocas

Ordem da tabela: `w0 (= θ, bias com x0 = -1), w1, w2, w3`. 4 casas decimais.

| Treinamento | w0i | w1i | w2i | w3i | w0f | w1f | w2f | w3f | Épocas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 (seed 10) | 0.7488 | 0.7713 | 0.0208 | 0.6336 | -3.1312 | 1.6227 | 2.5323 | -0.7483 | 452 |
| T2 (seed 20) | 0.8158 | 0.5881 | 0.8977 | 0.8915 | -3.1442 | 1.5916 | 2.5328 | -0.7482 | 465 |
| T3 (seed 30) | 0.1637 | 0.6441 | 0.3807 | 0.6630 | -3.0363 | 1.4840 | 2.4774 | -0.7261 | 376 |
| T4 (seed 40) | 0.2873 | 0.4077 | 0.0554 | 0.7885 | -3.0927 | 1.5396 | 2.5221 | -0.7107 | 405 |
| T5 (seed 50) | 0.3963 | 0.4946 | 0.2281 | 0.2555 | -3.1237 | 1.5773 | 2.5180 | -0.7428 | 416 |

Todos os 5 treinamentos convergiram (acurácia 1.000 no treino).

## Item 3 — Classificação das 10 amostras

Legenda: `-1 = C1`, `+1 = C2`. Execução: `python3 -m perceptron.perceptron_teste` (retreina T1–T5 com os mesmos seeds do item 2 e aplica `prever()` em `data/oleo_teste.csv`).

| Amostra | x1 | x2 | x3 | y(T1) | y(T2) | y(T3) | y(T4) | y(T5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.3565 | 0.0620 | 5.9891 | -1 | -1 | -1 | -1 | -1 |
| 2 | -0.7842 | 1.1267 | 5.5912 | 1 | 1 | 1 | 1 | 1 |
| 3 | 0.3012 | 0.5611 | 5.8234 | 1 | 1 | 1 | 1 | 1 |
| 4 | 0.7757 | 1.0648 | 8.0677 | 1 | 1 | 1 | 1 | 1 |
| 5 | 0.1570 | 0.8028 | 6.3040 | 1 | 1 | 1 | 1 | 1 |
| 6 | -0.7014 | 1.0316 | 3.6005 | 1 | 1 | 1 | 1 | 1 |
| 7 | 0.3748 | 0.1536 | 6.1537 | -1 | -1 | -1 | -1 | -1 |
| 8 | -0.6920 | 0.9404 | 4.4058 | 1 | 1 | 1 | 1 | 1 |
| 9 | -1.3970 | 0.7141 | 4.9263 | -1 | -1 | -1 | -1 | -1 |
| 10 | -1.8842 | -0.2805 | 1.2548 | -1 | -1 | -1 | -1 | -1 |

Os 5 modelos concordaram em todas as amostras (4 em C1, 6 em C2).

## Item 4 — Por que as épocas variam

O número de épocas variou (T1=452, T2=465, T3=376, T4=405, T5=416) porque cada treinamento partiu de um vetor inicial `w0..w3` diferente (`U[0,1]` com seeds 10, 20, 30, 40, 50 — ex: T3 começou em `[0.1637, 0.6441, 0.3807, 0.6630]` e convergiu em 376 épocas, T2 começou em `[0.8158, 0.5881, 0.8977, 0.8915]` e precisou de 465). Como a correção `eta*(d-y)*x` só ocorre quando há erro, pontos de partida distintos geram sequências de erros e trajetórias distintas até o hiperplano solução, mesmo com `eta = 0.01` e dados idênticos.

## Item 5 — Principal limitação do perceptron

Só resolve problemas linearmente separáveis, pois implementa um único hiperplano `-w0 + x1*w1 + x2*w2 + x3*w3 = 0`. Neste exercício os 30 padrões eram separáveis — prova: acurácia 1.000 nos 5 treinos e consenso total nas 10 amostras de teste. Em casos não-lineares (ex: XOR) nenhum vetor `w` zera o erro, o treino não converge e seria necessária uma rede multicamadas (MLP).
