# Teste ao vivo: 3 detectores de mãos em revezamento (19/10/2026, 19:18)

Cópia do app com `REVEZAMENTO_DETECTORES = 3`, realce desligado, memória 0,8 s,
detecção 0,3 e liberação sem voto de 60 quadros.
Diagnóstico: `ao_vivo_20261019_191826_diagnostico.csv`.

## Resultado

- **806 quadros em 27,1 s = 29,7 qps**: o limite da câmera (com 2 detectores,
  28,4; antes do revezamento, ~15).
- Mão sumida: esquerda 4%, direita 5%.
- Escritos: carro ×12, casa ×10, trocas rápidas de 6,7 s a 25,2 s (uma troca a
  cada 0,4 a 2,9 s). O app fechou sem erros, com os 3 detectores encerrados.

## Ponto de atenção

Entre 22,1 s e 23,7 s o carro foi escrito duas vezes seguidas. Nesse trecho
houve 8 votos de casa: poucos para escrever a casa, mas o bastante para
liberar o carro de novo (outro sinal libera em `QUADROS_PARA_LIBERAR` = 8
quadros). Provável casa rápida demais, que não chegou a ser confirmada.
