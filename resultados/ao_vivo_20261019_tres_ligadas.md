# Teste ao vivo: as três ligadas (19/10/2026, 18:07)

App da cópia (`app/`), modelo sem máscara, liberação sem voto 60 quadros, e
agora também: `REALCE_PELE = False`, `MEMORIA_MAO_S = 0.8`,
`CONFIANCA_DETECCAO = 0.3`. Diagnóstico: `ao_vivo_20261019_180712_diagnostico.csv`.

## O que foi feito

Trocas de propósito entre carro e casa de 0 a 10 s; depois carro.

## Resultado

- 536 quadros em 34,6 s (~15 qps).
- Mão sumida: esquerda 3% (no teste ao vivo anterior, 29%), direita 16% (antes 15%).

| Tempo | Escrito |
|---|---|
| 1,9 s | carro |
| 4,3 s | casa |
| 5,5 s | carro |
| 7,0 s | casa |
| 8,1 s | carro |
| 9,7 s | casa |
| 16,3 s | carro |

## Conclusão

Cada troca feita de propósito foi escrita, e o carro ficou firme de 16 s até o
fim, sem repetir. A mão esquerda, que mais sumia, quase não sumiu mais.
O usuário percebeu a melhora ao vivo.
