# Teste ao vivo: detectores de mãos em revezamento (19/10/2026, 18:58)

Cópia do app com `REVEZAMENTO_DETECTORES = 2` (2 detectores de mãos em
threads; rosto e braços numa thread só), mais realce desligado, memória 0,8 s,
detecção 0,3 e liberação sem voto de 60 quadros.
Diagnóstico: `ao_vivo_20261019_185848_diagnostico.csv`.

## Resultado

- **1053 quadros em 37,1 s = 28,4 qps** (nos testes ao vivo anteriores, ~15 qps).
- Mão sumida: esquerda 12%, direita 11%.
- Escritos: 1,8 s carro · 8,5 s casa · 10,4 s carro · 11,8 s casa · 12,6 s carro ·
  18,1 s casa · 23,9 s carro · 35,2 s casa · 35,5 s carro (carro ×5, casa ×4).
- Pré-sinais vistos: S ×12, B ×9. O app fechou sem erros.

## Velocidade com câmera falsa de 30 qps (`medir_qps_app.py`, mesma hora)

| Versão | qps |
|---|---|
| Antes (1 quadro por vez, pausa 15 ms) | 9,8 |
| 2 detectores de mãos em revezamento | 25,2 |
| 3 detectores de mãos em revezamento | 30,0 (qualidade não conferida nos vídeos) |

## Por que só as mãos se revezam

Com equipes completas (um Pose por thread), a casa caiu de 90% para 41–49% nos
vídeos: cada Pose suaviza o pulso do seu jeito e o pulso "tremia" entre
quadros. Revezando só as mãos, a qualidade ficou igual à de um detector vendo
todos os quadros (`revezamento_20261019_185429.md`).
