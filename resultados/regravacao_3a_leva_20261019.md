# Regravação: 3ª leva de carro e casa (19/10/2026, 21:40)

Depois do modelo das 21:18 (antigas + novas sem a gravação com aviso), foram
gravadas mais 3 rodadas de carro (com S) e 3 de casa (com B), todas "Boa
gravação". Treino com tudo (carro ×8, casa ×9, neutro ×1, S ×3, B ×3):
validação 99,9% em 17 rodadas.

Nos 11 vídeos (visão da cópia, mesma IA de pré-sinais):

| Modelo | Carro certo | Casa certa | Carro escrito (ideal 7) | Neutro: escritas falsas |
|---|---|---|---|---|
| Antigo (backup, de manhã) | 80% | 93% | 8 | 4 |
| 21:18 (carro ×5, casa ×6) | 86% | 93% | 8 | 6 |
| Agora (carro ×8, casa ×9) | 87% | 96% | 7 | 4 |

Backups: `app/armazenamento/backup_antes_regravar/`, `backup_so_gravacoes_novas/`,
`backup_modelo_2118/`.
