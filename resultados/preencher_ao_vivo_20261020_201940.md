# Preencher os buracos da mão ao vivo (vídeos de teste)

Modelo do app hoje (S como nada), com portão de movimento e região das mãos.
Com espera de K quadros, ao vivo o sinal sairia K/30 s mais tarde.

## Preencher ao vivo

| Variante | sem mão | carro: voto certo | carro: escritas (ideal 1/vídeo) | casa: voto certo | casa: escritas | neutro: escritas (ideal 0) | carro→casa→carro | S sozinho escreve | S + carro escreve |
|---|---|---|---|---|---|---|---|---|---|
| app hoje (congela 0,8 s) | 8% | 85% | 9/7 | 96% | luz_escura: casa×1, luz_normal: casa×1 | 2 | casa×9, carro×8 | nada | carro×1 |
| sem memória (buraco vazio) | 41% | 50% | 6/7 | 94% | luz_escura: casa×1, luz_normal: casa×1 | 2 | carro×9, casa×9 | nada | carro×1 |
| sem memória + congelar 0,8 s aqui | 3% | 85% | 10/7 (+1 erradas) | 96% | luz_escura: casa×1, luz_normal: casa×1 | 1 | casa×9, carro×8 | nada | carro×1 |
| preencher até 3 quadros (espera 0.10 s) + congelar | 3% | 86% | 10/7 (+1 erradas) | 96% | luz_escura: casa×1, luz_normal: casa×1 | 1 | carro×9, casa×9 | nada | carro×1 |
| preencher até 5 quadros (espera 0.17 s) + congelar | 3% | 86% | 10/7 (+1 erradas) | 96% | luz_escura: casa×1, luz_normal: casa×1 | 1 | carro×9, casa×9 | nada | carro×1 |
| preencher até 15 quadros (espera 0.50 s) + congelar | 3% | 87% | 10/7 | 96% | luz_escura: casa×1, luz_normal: casa×1 | 1 | carro×9, casa×9 | nada | carro×1 |

Tempo total: 602 s