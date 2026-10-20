# Por que o app ao vivo roda a ~15 qps (19/10/2026)

## Duas causas

1. **Câmera com pouca luz:** à noite a webcam entregou só 17 qps, mesmo sem
   nenhum processamento (exposição automática longa). A leitura espera o
   próximo quadro: 37 a 58 ms.
2. **Detector de mãos pesado:** dentro do app, `maos.detectar` leva ~49 ms por
   quadro; todo o resto soma ~13 ms (rosto 4,7; vetor 2,5; tela 2,6; desenho 1,2).

## Laço do app com câmera falsa de 30 qps (`medir_qps_app.py`)

| Versão | qps | volta (ms) | espera da câmera (ms) | tela (ms) |
|---|---|---|---|---|
| A: como está (pausa 15 ms) | 13,0 | 76,8 | 12,4 | 2,1 |
| B: pausa 1 ms | 12,5 | 77,1 | 9,3 | 2,7 |
| C: pausa 1 ms + câmera em thread | 14,5 | 62,0 | 0,0 | 2,8 |

## O detector de mãos não depende de Pose, detecção nem realce

~46 ms (mediana) em todas as 8 combinações: Pose em paralelo ou em sequência,
detecção 0,5 ou 0,3, realce ligado ou desligado.

## Modelo de mãos leve (model_complexity 0) × completo (1, o do app)

| Vídeo | Modelo | ms | Sem mão | 2 mãos |
|---|---|---|---|---|
| carro rápido | completo | 43,7 | 26% | 34% |
| carro rápido | leve | 31,2 | 32% | 32% |
| carro muito rápido | completo | 46,3 | 59% | 5% |
| carro muito rápido | leve | 31,0 | 72% | 3% |
| casa | completo | 42,6 | 0% | 100% |
| casa | leve | 20,5 | 0% | 98% |

O leve é ~30% mais rápido, mas perde mais mãos justo no movimento rápido.
Por isso não foi adotado.
