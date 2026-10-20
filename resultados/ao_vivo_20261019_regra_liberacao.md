# Teste ao vivo: regra nova de liberação (19/10/2026, 16:30)

App da cópia (`app/`), modelo **sem máscara**, `QUADROS_PARA_LIBERAR_SEM_VOTO = 60`
(mão sumida só libera o sinal depois de 2 s; outro sinal libera em 8 quadros).
Diagnóstico completo: `ao_vivo_20261019_163008_diagnostico.csv`.

## O que foi feito

Carro contínuo e, de propósito, trocas para casa aos 28–33 s e aos 54–65 s.

## Resultado

- 994 quadros em 66,7 s (~15 qps; a câmera estava mais lenta, pouca luz).
- Mão sumida: esquerda 29% dos quadros, direita 15%.
- Votos: carro 525, casa 83, nenhum 386.

| Tempo | Escrito |
|---|---|
| 1,7 s | carro |
| 28,2 s | casa |
| 30,1 s | carro |
| 31,8 s | casa |
| 32,5 s | carro |
| 55,8 s | casa |
| 59,1 s | carro |
| 64,0 s | casa |
| 64,7 s | carro |

## Conclusão

- Carro contínuo de 1,7 s a 28 s (26 s) escrito **1 vez**, mesmo com a mão
  sumindo. Com a regra antiga (8 quadros), os vídeos de teste mostravam até 10.
- As trocas carro ↔ casa feitas de propósito foram todas escritas.
- Nos vídeos (`liberacao_20261019_162831.md`): carro muito rápido 10 → 1,
  mãos embaixo 4 → 1, perto 3 → 1, luz escura 10 → 2, longe 7 → 4.
