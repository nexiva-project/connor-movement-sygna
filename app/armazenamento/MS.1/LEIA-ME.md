# MS.1

Modelo aprovado pelo usuário em 19/10/2026, 21:50.

## O que tem nesta pasta

| Arquivo | O que é |
|---|---|
| `modelo_sinais.pkl` | IA dos sinais (RandomForest, sem máscara): carro, casa, neutro |
| `modelo_sinais_pre.pkl` | IA dos pré-sinais: S antes de carro, B antes de casa |
| `modelo_sinais.json` | Resumo do treino |
| `dados_sinais.csv` | As gravações usadas: carro ×8, casa ×9, neutro ×1, S ×3, B ×3 |

## Configuração do app que acompanha o MS.1

Código: branch `lab/movimento`, commit `e3bdb5d` (tag local `MS.1`).

- `REALCE_PELE = False`, `MEMORIA_MAO_S = 0.8`, `CONFIANCA_DETECCAO = 0.3`
- `QUADROS_PARA_LIBERAR_SEM_VOTO = 60` (mão sumida libera o sinal em 2 s)
- `REVEZAMENTO_DETECTORES = 3` (3 detectores de mãos, ~30 qps)
- `MASCARA_FRACOES = ()` (sem máscara)

## Resultados

- Validação no treino: 99,9% (17 rodadas).
- Nos 11 vídeos de teste: carro certo 87%, casa certa 96%, carro contínuo escrito
  7 vezes em 7 vídeos (o ideal), 4 escritas falsas no vídeo de mão relaxada.
- Ao vivo (47 s): 29,9 qps, certeza média 94%, incerto 10%, mão sumida 4–5%.

## Como voltar a usar

Copie os 4 arquivos para `app/armazenamento/` (substituindo os de lá). Use
com o código da tag `MS.1`.
