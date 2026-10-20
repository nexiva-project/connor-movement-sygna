# Testes de 20/10/2026 com o MS.1

## Pré-sinal (presinal_20261020_184604.md)

- O S sozinho (10 s) já é escrito como "carro" (1 a 2 vezes): o carro é feito
  com as mãos em S e a IA confunde os dois.
- Com o S antes, o carro de verdade não é escrito de novo (já estava escrito
  pelo S). No vídeo em etapas S → carro → S → carro, o carro saiu só 1 vez.
- A IA de pré-sinais vê o S DENTRO do próprio carro (S aos 0,1 s do carro
  sozinho), então a regra 70%/78% quase não muda nada.
- B → casa: o B sozinho não vira casa; com ou sem B a casa sai igual (0,44 s).

## Ajustes da visão (ajustes_ms1_20261020_185342.md)

Memória 1,0 s e detecção 0,2, sozinhas ou juntas: mão sumida 8% → 7%, carro e
casa iguais ou 1 ponto piores. Não vale mudar.

## Votos de outro sinal para liberar

8, 12 e 15 votos deram resultados idênticos nos vídeos, inclusive as ~16 trocas
rápidas do carro→casa→carro. Os vídeos não têm o caso da "casa rápida demais"
visto ao vivo, então não dá para medir o ganho; subir para 12 não custou nada.

## S parado como "nada" (s_como_nada_20261020_190112.md)

As 3 gravações de pré-sinal S do treino entram também na IA dos sinais como
"nada" (rótulo que o app nunca escreve).

| Modelo | S sozinho escreve | S + carro: 1º carro após | 11 vídeos: carro certo | carro escrito (ideal 7) | casa certa |
|---|---|---|---|---|---|
| MS.1 | carro ×2 | não sai (2 de 3 vídeos) | 86% | 7 | 96% |
| MS.1 + S como nada | nada | 0,15 a 1,15 s | 84% | 8 | 96% |
| MS.1 + S e B como nada | nada | 0,15 a 1,35 s | 81% | 9 | 96% |

S como "nada" resolve o S virando carro e faz o carro sair MAIS RÁPIDO depois do
S (0,15 s, contra 0,74 s sem o S), custando 2 pontos no carro dos 11 vídeos.
Colocar também o B piora o carro e não ajuda a casa. No vídeo em etapas
(S → carro → S → carro) o carro continua saindo só uma vez.

## Portão de movimento (portao_movimento_20261020_191418.md)

Ideia do satyaidk/sign-language-recognition: a IA só vota se o pulso (Pose) se
mexeu acima de um limite nos últimos H segundos. Sem trocar o modelo MS.1.

| Variante | S sozinho escreve | S + carro: 1º carro após | carro certo | carro escrito (ideal 7) | casa certa | neutro falso |
|---|---|---|---|---|---|---|
| sem portão (MS.1 hoje) | carro ×2 | não sai | 86% | 7 | 96% | 4 |
| limite 0,8 · janela 0,5 s | nada | 0,15 s | 85% | 7 | 96% | 3 |

O melhor foi limite 0,8 larguras de ombro/s e janela 0,5 s: resolve o S parado
virando carro sem treinar de novo, e ainda tira uma escrita falsa do neutro.
Custo: 1 ponto no voto certo do carro. Janela de 1,0 s deixa o S virar carro 1 vez.
No vídeo em etapas (S → carro → S → carro) o carro continua saindo só uma vez.
