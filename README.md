# Laboratório MS × RandomForest

Compara, nas gravações reais do **kara-sygna-py**, o RandomForest que o app usa
hoje com o **MS (movimento de sinal)**: alinha a gravação nova com cada gravação
de ensino (por DTW, *Dynamic Time Warping*) e fica com a mais parecida.

## Pastas

* `app/`: **cópia** do kara-sygna-py (branch `experiment/kun`, copiada em
  19/10/2026, sem a `.venv`). É nela que as ideias novas entram. O
  kara-sygna-py original **não é alterado**.
* O laboratório importa o código e lê as gravações da cópia `app/` (mesmo vetor
  de 223 números, mesmas janelas de 10 quadros, mesmo RandomForest com espelho,
  velocidades e aumento de dados).

## O que mudou na cópia (`app/`)

| Mudança | Onde | Situação |
|---|---|---|
| Máscara no treino (cópias com a mão apagada) | `ia/mascara.py`, `ia/sequencia.py`, `config.MASCARA_FRACOES` | **Não usada**: piorou a validação (99,0% → 85,8%) e não ajudou nos vídeos. Modelos guardados em `armazenamento/modelo_sinais_mascara*.pkl` |
| Validação ignora as cópias com máscara | `ia/treinamento.py` | Ativa (sem efeito sem máscara) |
| Mão sumida libera o sinal só depois de 2 s | `ia/reconhecimento.py`, `config.QUADROS_PARA_LIBERAR_SEM_VOTO = 60` | **Ativa**. Carro contínuo: até 10 escritas → 1 (vídeos e ao vivo) |

O kara-sygna-py original **não foi alterado**.

## Vídeos de teste

* `gravar_video.py`: janela com a lista de 16 vídeos; clique para gravar
  (3 s + 20 s). Vão para `videos/` (só teste, não entram no treino).
* `estudar_videos.py`: passa cada vídeo pela visão do app (uma vez, com cache
  em `videos/cache/`) e reconhece com cada modelo.
* `estudar_liberacao.py`: mesmos vídeos, testando o tempo de liberação.

## Como rodar

Use o Python da `.venv` do kara-sygna-py original. Ele só serve para rodar,
porque já tem numpy, pandas, scikit-learn e mediapipe:

```bash
C:/Users/chris/Desktop/novo/kara-sygna-py/.venv/Scripts/python.exe comparar.py
```

Para usar outra cópia do app, defina `KARA_SYGNA_PY` com o caminho.
Opções: `--csv`, `--peso-forma`, `--peso-movimento`, `--sem-espelho`.
O relatório fica em `resultados/`.

Testes: `python -m pytest`.

## Como a comparação é feita

* **Uma rodada de fora:** cada gravação de um sinal com 2 ou mais rodadas é
  testada com os métodos treinados nas outras. Sinais com 1 rodada (ex.:
  `neutro`) só entram no treino.
* **Mesmas janelas:** os dois métodos classificam as mesmas janelas de 10
  quadros que o app usaria ao vivo. Também se mede a **gravação inteira**
  (RandomForest: voto das janelas; MS: alinhamento da gravação inteira).
* **Dois cenários:** treino com todas as outras rodadas (como o app hoje) e
  com só **1 rodada por sinal** (poucos dados).
* **Ritmo:** com 3 rodadas por sinal, a ordem dá devagar, médio e rápido.
* **ms/janela:** tempo para classificar uma janela (o app tem ~33 ms por quadro).

## O MS

* Por quadro: a **forma** das duas mãos (126 valores) e a **velocidade** do
  pulso de cada braço, pelo Pose (4). Velocidade, não posição, para valer em
  qualquer lugar, como `USAR_LUGAR = False` no app.
* Mão que sumiu repete a forma válida mais próxima.
* Cada gravação também é guardada espelhada (vale para a outra mão).
* Janela ao vivo: DTW de **subsequência** (acha o trecho do modelo que mais
  combina). Gravação inteira: DTW completo com faixa de Sakoe-Chiba.

## Mão sumindo por borrão (simulação)

`--buracos 0.3` apaga trechos de 2 a 15 quadros de cada mão nas gravações de
**teste**, até 30% dos quadros, e passa pela memória de 0,3 s do app. É como
o "carro" ao vivo. O pulso do braço (Pose) fica, como na vida real.
`--mascara-treino` também testa um RandomForest treinado com cópias das
gravações com 15%, 30% e 50% das mãos apagadas.

```bash
C:/Users/chris/Desktop/novo/kara-sygna-py/.venv/Scripts/python.exe comparar.py --buracos 0.3 --mascara-treino
```

## Teste de câmera ao vivo

`camera_teste.py` abre a webcam com o detector de mãos do app e mostra quantos
quadros ficam sem mão. Pelo teclado você muda a exposição (`e`/`E`, `a` volta
ao automático), o ganho (`g`/`G`), a confiança de rastreio (`t`), o realce de
pele (`r`) e abre os ajustes do driver (`s`). **ESPAÇO** mede 10 s enquanto
você faz o sinal; cada medida vai para `resultados/camera_<data>.csv`.

```bash
C:/Users/chris/Desktop/novo/kara-sygna-py/.venv/Scripts/python.exe camera_teste.py
```

* `--backend dshow` (padrão): aceita exposição manual, mas roda a ~17 qps.
* `--backend msmf`: o do app (até 30 qps), mas ignora a exposição manual.
* Ao sair, a exposição volta ao automático (a webcam guarda o valor manual).

Na webcam de teste: automático ≈ 31 ms (-5); -6 = 15,6 ms ainda dá imagem;
-7 = 7,8 ms fica preto sem luz extra; menos que -7 a câmera não aceita.

## Arquivos

```
comparar.py            roda tudo e escreve o relatório
gravar_video.py        grava os vídeos de teste
estudar_videos.py      estuda os vídeos com cada modelo
estudar_liberacao.py   estuda a regra de liberação nos vídeos
app/                   cópia do kara-sygna-py (onde as ideias entram)
videos/                vídeos de teste (.avi + .json) e cache da visão
camera_teste.py        teste de câmera ao vivo (exposição, rastreio, realce)
lab_ms/buracos.py      simula a mão sumindo e a memória de 0,3 s
lab_ms/kara.py         acha e importa o kara-sygna-py
lab_ms/alinhamento.py  alinhamento inteiro e de trecho (técnica DTW, numpy)
lab_ms/ms_knn.py       vetor por quadro e o classificador MS (1-NN)
lab_ms/experimento.py  protocolo de teste e os dois métodos
testes/                testes sem câmera
```
