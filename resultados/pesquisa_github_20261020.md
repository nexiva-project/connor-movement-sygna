# Pesquisa: projetos do GitHub que ainda não usamos (20/10/2026)

| Projeto | O que faz | O que serve para o MS.1 | Licença |
|---|---|---|---|
| [satyaidk/sign-language-recognition](https://github.com/satyaidk/sign-language-recognition) | 12 sinais, ~11 gravações cada, BiGRU 0,5M parâmetros, 91% (1 pessoa) | "Portão de movimento": só reconhece quando a mão se move e fecha o sinal após 0,5 s parada; filtros finais (mão em >15% dos quadros, 55% de certeza, margem 15%, mesmo sinal ignorado por 4 s); velocidade + máscara de presença como entrada; filtro One-Euro | não informada |
| [ceydaakin/asl-realtime](https://github.com/ceydaakin/asl-realtime) | ASL, 250 sinais, transformer 1,26M parâmetros, 74,7% em pessoas novas | Conv1D venceu GRU em todos os testes; sem espelhamento de propósito; normalização pelo ombro; aumento: escala, rotação, corte, pontos apagados, velocidade | MIT |
| [danielelvs/islr-subset](https://github.com/danielelvs/islr-subset) | Libras (MINDS-Libras, LIBRAS-UFOP) com MediaPipe | Preencher buracos com spline (até 5 quadros) subiu 68% → 77%; o melhor subconjunto é pose + duas mãos, sem rosto; datasets de Libras prontos | MIT |
| [nielsschluesener/Fast-Learning-Hand-Gesture-Recognition](https://github.com/nielsschluesener/Fast-Learning-Hand-Gesture-Recognition) | Gestos dinâmicos com 1, 2 ou 5 exemplos (Relation Network) | Ensinar sinal novo com 5 gravações: 81% em 10 gestos | não informada |
| [AdrianoCLeao/talking-hands](https://github.com/AdrianoCLeao/talking-hands) | Libras em tempo real, ~18 sinais, Keras, interface PyQt5, fala | Exemplo de app Libras completo; sem números de acerto | MIT |

Também vistos: gabguerin/Sign-Language-Recognition--MediaPipe-DTW (DTW, como o nosso MS
de alinhamento), Monika974/Sign-language-recognition (GRU/LSTM/CNN/Transformer em WLASL).
