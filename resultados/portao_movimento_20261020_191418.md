# Portão de movimento nos vídeos de teste (MS.1)

Velocidade do pulso (Pose) em larguras de ombro/s; portão aberto se passou do
limite nos últimos H segundos, ou sem corpo para medir.

## Os 11 vídeos

| Variante | sem mão | carro: voto certo | carro: escritas (ideal 1/vídeo) | casa: voto certo | casa: escritas | neutro: escritas (ideal 0) | carro→casa→carro |
|---|---|---|---|---|---|---|---|
| sem portão (MS.1 hoje) | 8% | 86% | 7/7 | 96% | luz_escura: casa×1, luz_normal: casa×1 | 4 | casa×9, carro×8 |
| limite 0,6 · janela 0,5 s | 8% | 85% | 7/7 | 96% | luz_escura: casa×1, luz_normal: casa×1 | 4 | casa×9, carro×8 |
| limite 0,6 · janela 1,0 s | 8% | 85% | 7/7 | 96% | luz_escura: casa×1, luz_normal: casa×1 | 4 | casa×9, carro×8 |
| limite 0,8 · janela 0,5 s | 8% | 85% | 7/7 | 96% | luz_escura: casa×1, luz_normal: casa×1 | 3 | casa×9, carro×8 |
| limite 0,8 · janela 1,0 s | 8% | 85% | 7/7 | 96% | luz_escura: casa×1, luz_normal: casa×1 | 3 | casa×9, carro×8 |

## Pré-sinal parado e S/B + sinal

| Variante | S sozinho escreve | S + carro: 1º carro após | B sozinho escreve | B + casa: 1º casa após | etapas S→carro→S→carro |
|---|---|---|---|---|---|
| sem portão (MS.1 hoje) | carro×2 | - | nada | 0.44 s | carro 5.9 s |
| limite 0,6 · janela 0,5 s | nada | 0.15 s | nada | 0.44 s | carro 5.9 s |
| limite 0,6 · janela 1,0 s | carro×1 | 0.15 s | nada | 0.44 s | carro 5.9 s |
| limite 0,8 · janela 0,5 s | nada | 0.15 s | nada | 0.44 s | carro 5.9 s |
| limite 0,8 · janela 1,0 s | carro×1 | 0.15 s | nada | 0.44 s | carro 5.9 s |

Tempo total: 209 s