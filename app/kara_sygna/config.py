"""Configurações centrais do Kara Sygna.

Todo número "mágico", caminho ou limiar fica aqui, em um só lugar. Os demais
módulos importam estas constantes em vez de repetir valores espalhados.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
# Ancorados na raiz do projeto: não dependem da pasta de onde o programa foi
# executado (o que antes fazia o CSV "cair" dentro de src/).
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
PASTA_ARMAZENAMENTO = RAIZ_PROJETO / "armazenamento"
ARQUIVO_CSV = PASTA_ARMAZENAMENTO / "dados_sinais.csv"
ARQUIVO_MODELO = PASTA_ARMAZENAMENTO / "modelo_sinais.pkl"
# Um CSV por sessão de reconhecimento, com o que a IA viu e decidiu em cada
# quadro (para estudar erros). DIAGNOSTICO_ATIVO = False desliga.
PASTA_DIAGNOSTICO = PASTA_ARMAZENAMENTO / "diagnostico"
DIAGNOSTICO_ATIVO = True

# ---------------------------------------------------------------------------
# Câmera
# ---------------------------------------------------------------------------
# Índices testados em ordem. Em notebooks Linux é comum /dev/video0 ser a
# imagem e /dev/video1 apenas metadados; por isso cada índice é validado lendo
# um quadro de verdade.
INDICES_CAMERA = (0, 1, 2, 3, 4)
# Resolução pedida à câmera. As características das mãos usam coordenadas
# normalizadas pela imagem, então manter a mesma proporção (4:3) em todos os
# computadores evita que o modelo "veja" mãos esticadas ou achatadas.
# Use None para não forçar nenhuma resolução.
LARGURA_CAPTURA = 640
ALTURA_CAPTURA = 480
LEITURAS_DE_TESTE = 10
PAUSA_ENTRE_LEITURAS_S = 0.05

# ---------------------------------------------------------------------------
# MediaPipe (detecção das mãos)
# ---------------------------------------------------------------------------
MAX_MAOS = 2
# Qual detector acha as mãos: "hands" (MediaPipe Hands, o de sempre) ou
# "landmarker" (Hand Landmarker, do MediaPipe Tasks, em teste desde 24/10/2026;
# no vídeo do dedo escondido ficou parecido com o Hands). Escolhido na janela
# que abre antes da tela de carregamento.
DETECTOR_MAOS = "hands"
# Modelo do Hand Landmarker (hand_landmarker.task, do MediaPipe/Google), na
# pasta "modelos" do laboratório, ao lado da pasta do app.
MODELO_HAND_LANDMARKER = RAIZ_PROJETO.parent / "modelos" / "hand_landmarker.task"
# Núcleos do processador por chamada dos detectores do rtmlib (RTMPose...).
# Com o padrão do ONNX (todos os 28), o RTMPose levava 128 ms por quadro;
# com 4, 33 ms; com 8, 26 ms. Com o rtmlib, o app usa um detector só (sem
# revezamento). Com o RTMW-x (24/10/2026): 8 → 41 ms, 12 → 36 ms, 16 → 33 ms;
# 12 ainda deixa núcleos para o rosto, o corpo e o treino da IA.
RTMLIB_NUCLEOS = 12
# Rodar o rtmlib na placa de vídeo (DirectML: pip install onnxruntime-directml),
# se houver. No PC do lab (RTX 4070 Ti SUPER, 24/10/2026), o RTMW-x caiu de
# 37 ms para 9 ms, com os mesmos pontos; então roda em todo quadro. Sem placa,
# fica na CPU (RTMLIB_NUCLEOS e RTMLIB_A_CADA).
RTMLIB_PLACA_VIDEO = True
# O RTMPose procura a mão do zero a cada tantos quadros (PoseTracker); no meio,
# segue a mão pela posição anterior. E filtro One-Euro nos pontos (corte em
# Hz, beta). No vídeo do defeito (24/10/2026): quadro a quadro, 26 pulos de
# dedo; a cada 7 + One-Euro (1,0; 0,01), 3 (o MediaPipe Hands teve 9). A cada
# 3: os mesmos poucos pulos, e uma mão que entra aparece em até 0,1 s (com 7,
# até 0,23 s).
RTMLIB_DETECCAO_A_CADA = 6
# O modelo do rtmlib roda a cada tantos quadros; no meio, repete as últimas
# mãos (na CPU, o RTMW-x leva ~36 ms: a cada 2, a tela não fica esperando).
RTMLIB_A_CADA = 2
# Filtro mais leve (3,0; 0,05): com (1,0; 0,01) a malha do RTMW-x, a ~9
# quadros/s, ficava atrasada (em 28% dos quadros, a mais de meia palma da mão).
RTMLIB_EURO_CORTE = 3.0
RTMLIB_EURO_BETA = 0.05
# Na placa de vídeo o RTMW-x roda em todo quadro, então dá para filtrar mais
# forte, mas separado: o pulso leve (a mão acompanha o movimento rápido) e a
# forma dos dedos (cada dedo menos o pulso) forte. Estudo de 24/10/2026 no
# vídeo do defeito, contra o filtro único (3,0; 0,05): pulos de dedo de 15
# para 3, tremida (aceleração dos dedos) 2,1 vezes menor e o pulso até mais
# perto da mão (no pior quadro, 0,07 palma atrás, antes 0,10). Um filtro único
# forte (1,0; 0,005) tirava todos os pulos, mas a malha ficava atrasada no
# movimento rápido (0,46 palma no pior quadro; você viu ao vivo).
RTMLIB_EURO_PLACA_PULSO = (5.0, 0.1)
RTMLIB_EURO_PLACA_DEDOS = (1.0, 0.01)
# 0.5: com 0.7, mãos borradas pelo movimento eram descartadas (no teste com
# o sinal "carro", 42% dos quadros ficavam sem nenhuma mão; com 0.5, 24%).
# 0.3 (laboratório, 19/10/2026): nos 11 vídeos de teste, junto com o realce
# desligado e a memória de 0.8 s, a mão sumida caiu de 37% para 8%.
CONFIANCA_DETECCAO = 0.3
# 0.3: no vídeo do sinal "carro", foi o valor que mais manteve as duas mãos.
CONFIANCA_RASTREIO = 0.3
# Anti-pulo (kara_sygna.visao.anti_pulo): um dedo que pula sozinho (mais que
# este tanto, em tamanhos de palma) com o resto da mão parado volta para onde
# estava, por até ANTI_PULO_MAX_QUADROS quadros seguidos. 0 = desligado.
ANTI_PULO_LIMIAR = 0.35
# TEMPORÁRIO (24/10/2026, teste do atraso da malha do RTMW-x): era 3.
ANTI_PULO_MAX_QUADROS = 0
# TEMPORÁRIO (24/10/2026, teste do atraso da malha do RTMW-x): False desliga o
# FaceMesh (rosto) e o Pose (braços); os blocos deles ficam em zero, então as
# IAs treinadas com rosto e braços podem errar mais. Para voltar: True.
ROSTO_E_CORPO = True
# TEMPORÁRIO (mesmo teste): False desenha as mãos sem a "Suavização das mãos"
# das Configurações (só o One-Euro do detector). Para voltar: True.
SUAVIZAR_DESENHO_MAOS = False
# O detector de mãos recebe uma cópia da imagem com a pele clara e o resto
# escuro (a cor da pele é medida no rosto), para as mãos se destacarem da
# roupa. Só o detector vê essa cópia; a tela mostra a imagem normal.
# False (laboratório, 19/10/2026): nos vídeos de teste, sem realce a mão
# sumida caiu de 37% para 16% e a casa no escuro passou a ser reconhecida.
REALCE_PELE = False
# Quanto sobra do que não é pele (0 = preto, 1 = sem realce).
REALCE_PELE_ESCURO = 0.1
# Se uma mão some por até estes segundos (quadro borrado), continua valendo a
# última posição dela, no desenho e na IA. 0 = desliga.
# 0.8 (laboratório, 19/10/2026): nos vídeos de teste, a mão sumida caiu de
# 37% para 25% sozinha, sem piorar nenhum sinal.
MEMORIA_MAO_S = 0.8
# Com o rtmlib (RTMW...), que segue a mão sozinho, a memória ao vivo é curta:
# a malha some logo quando a mão sai da tela (com 0,8 s ela ficava um pouco
# atrasada). Só ao vivo: a máscara do treino continua com MEMORIA_MAO_S, igual
# para todos os detectores.
MEMORIA_MAO_RTMLIB_S = 0.3

# MediaPipe Pose (braços): 0 = modelo leve (~13 ms por quadro), 1 = completo.
COMPLEXIDADE_POSE = 0
# Conferência das mãos pelos braços (tira as "mãos fantasmas"), em larguras
# de ombro. Só é descartada a mão mais longe que isto dos dois pulsos do Pose
# (o Pose erra o pulso com as mãos embaixo; com 0,6 ele descartava a mão
# esquerda de verdade no sinal "carro").
DISTANCIA_MAO_PULSO = 1.5
# Duas mãos mais perto que isto uma da outra são a mesma mão (fantasma).
DISTANCIA_MAOS_DUPLICADAS = 0.35

# O frame é espelhado (como um espelho) antes de ir para o MediaPipe, e o
# MediaPipe já espera a imagem espelhada: o rótulo "Right" é a mão direita de
# verdade. False = sem inverter (a mão direita é "dir"). Até a versão com
# True, a mão direita aparecia como "esq". Mudar este valor troca o
# significado dos dados gravados: grave os sinais de novo.
INVERTER_LATERALIDADE = False

# Cores (BGR, como o OpenCV espera) usadas para desenhar cada mão.
CORES_MAOS_BGR = {"esq": (70, 190, 255), "dir": (150, 105, 255)}

# ---------------------------------------------------------------------------
# Reconhecimento
# ---------------------------------------------------------------------------
CONFIANCA_MINIMA = 0.70  # probabilidade mínima do melhor sinal
MARGEM_MINIMA = 0.15  # melhor - segundo melhor (evita empates)
JANELA_ESTABILIZADOR = 10  # quadros considerados na votação
VOTOS_MINIMOS = 6  # votos necessários para confirmar um sinal
QUADROS_PARA_LIBERAR = 8  # quadros com OUTRO sinal para poder repetir o sinal
# Quadros sem o sinal no total (mão sumida, IA incerta) para poder repeti-lo.
# 60 (2 s): nos vídeos de teste, o carro contínuo era escrito até 10 vezes com
# 8 (como era antes) e 1 vez com 60. Para repetir o mesmo sinal, abaixe as mãos
# por 2 s ou passe pelo neutro (outro sinal libera em QUADROS_PARA_LIBERAR).
QUADROS_PARA_LIBERAR_SEM_VOTO = 60
# Portão de movimento (kara_sygna.ia.portao): a IA só vota se o pulso se mexeu
# mais que LIMITE (larguras de ombro por segundo) nos últimos JANELA_S. Assim a
# mão parada em S não vira "carro". 0 = desliga. Laboratório, 20/10/2026: 0.8
# e 0.5 s foram os melhores nos vídeos de teste.
PORTAO_MOVIMENTO_LIMITE = 0.8
PORTAO_MOVIMENTO_JANELA_S = 0.5
# O movimento dos primeiros segundos depois que as mãos aparecem não abre o
# portão (é a subida das mãos até a posição). 26/10/2026, nos vídeos de teste:
# 0,3 manteve o carro e a casa; 0,5 já perdia a casa no escuro (80% → 57%).
PORTAO_SUBIDA_S = 0.3
# Região entre as mãos (kara_sygna.ia.regiao): num sinal de duas mãos, o voto
# não vale se as mãos estiverem bem mais longe uma da outra do que nas
# gravações (percentil 95 + esta folga, em larguras de ombro). O lugar do sinal
# continua livre. Negativo = desliga.
REGIAO_MAOS_FOLGA = 0.4
# Quadros recentes que a região olha: se as mãos ficaram longe em qualquer um
# deles, o voto não vale (ao vivo, a distância oscila em volta do limite).
REGIAO_MAOS_MEMORIA = 15
# Rótulos aprendidos pela IA que NÃO viram texto (ex.: mão relaxada).
ROTULOS_IGNORADOS = frozenset({"neutro", "nada"})

# ---------------------------------------------------------------------------
# Pré-sinal (forma de mão, como o "S", que vem antes de alguns sinais)
# ---------------------------------------------------------------------------
# Probabilidade mínima para a IA dizer que uma mão está num pré-sinal. 0.50:
# num teste ao vivo, o "S" mostrado ficou entre 33% e 60% e nunca chegava a 70%.
CONFIANCA_PRE_SINAL = 0.50
# Um sinal que tem pré-sinal também é escrito SEM o pré-sinal ter sido visto,
# se a IA tiver pelo menos esta certeza. O pré-sinal vira uma ajuda (quando
# aparece, só os sinais dele valem), não uma trava. 1.0 = exige o pré-sinal.
# 0.78: num teste ao vivo, os quadros certos que esperavam o pré-sinal tinham
# 78% em média (com 0.85 eles não eram escritos).
CONFIANCA_SEM_PRE_SINAL = 0.78
# Votação do pré-sinal: VOTOS_PRE_SINAL entre os últimos JANELA_PRE_SINAL
# quadros (menos que nos sinais: a forma da mão é fácil de ver e precisa sair
# logo, para o sinal não atrasar).
JANELA_PRE_SINAL = 5
VOTOS_PRE_SINAL = 3
# Árvores da IA de pré-sinais (é menor que a dos sinais: só olha uma mão).
ARVORES_PRE_SINAL = 100
# No ensino com várias etapas e o campo de segundos vazio, cada etapa dura
# isto (para a sequência andar sozinha).
SEGUNDOS_PRE_SINAL_PADRAO = 8
# Um Enter grava o sinal uma vez em cada ritmo (cada uma vira uma rodada):
# "carro 1/3: devagar", "carro 2/3: médio", "carro 3/3: rápido".
# () = uma vez só, como antes.
RITMOS_ENSINO = ("devagar", "médio", "rápido")
# No reconhecimento, por quantos segundos o pré-sinal visto ajuda os sinais
# dele (eles são aceitos com a confiança normal).
SEGUNDOS_ESPERA_SINAL_PADRAO = 3

# ---------------------------------------------------------------------------
# Quais quadros da câmera são usados (ensino E reconhecimento)
# ---------------------------------------------------------------------------
# "mudanca": só usa um quadro quando a mão muda (forma ou lugar) ou quando uma
#            mão aparece/some. Mão parada não enche o CSV. (Filtro original.)
# "fixo":    usa 1 a cada USAR_1_A_CADA_QUADROS quadros, mude ou não.
# Se trocar o modo, grave os sinais de novo e treine a IA.
MODO_QUADROS = "mudanca"

# Modo "mudanca":
# Mudança mínima na forma das mãos (média da diferença dos 126 valores).
LIMIAR_MUDANCA_FORMA = 0.015
# Mudança mínima no lugar das mãos, em larguras de rosto (0.05 = 5% do rosto).
LIMIAR_MUDANCA_POSICAO = 0.05
# Com a mão parada, ainda usa 1 quadro a cada tantos segundos, para a IA
# aprender o sinal parado. 0 = nunca (exatamente como o filtro original).
PARADO_USAR_A_CADA_S = 0.5

# Modo "fixo":
USAR_1_A_CADA_QUADROS = 2

# False: a IA NÃO usa o lugar exato das mãos (distância dos dedos até o rosto,
# posição das mãos em relação ao nariz e dos braços em relação aos ombros);
# olha a forma das mãos e o movimento. Num teste com o sinal "casa" feito em
# 7 lugares, com o lugar ela só reconhecia no lugar de sempre; sem o lugar,
# reconheceu em 6 dos 7, com a mesma acurácia (99%). Ligue (True) para sinais
# que só mudam pelo lugar (ex.: mesma mão na testa ou no peito). Ao mudar,
# treine a IA de novo (não precisa regravar).
USAR_LUGAR = False

# Quantos quadros usados a IA olha para reconhecer um sinal (inclusive o
# movimento).
QUADROS_POR_JANELA = 10

# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------
SEMENTE_ALEATORIA = 42
ARVORES = 300
MIN_AMOSTRAS_FOLHA = 2
MAX_DOBRAS_VALIDACAO = 5
MIN_VARIANTES_AUMENTO = 2
MAX_VARIANTES_AUMENTO = 8
ORCAMENTO_LINHAS_AUMENTO = 40_000
# No treino, cada gravação é cortada em janelas que começam de N em N quadros.
PASSO_JANELA_TREINO = 2
# Velocidades extras de cada gravação no treino (0.75 = mais devagar, 1.33 =
# mais rápido), para o sinal valer feito em ritmos diferentes. () = desliga.
VELOCIDADES_TREINO = (0.75, 1.33)
# Sinais de direção livre (ex.: "avião", marcados no ensino ou na lista): no
# treino, o movimento também entra girado nestes ângulos (graus), para valer
# para frente, para a esquerda, para cima... () = só como foi gravado.
ANGULOS_DIRECAO_LIVRE = (45, 90, 135, 180, 225, 270, 315)
# Redes neurais (GRU, LSTM, Conv1D; kara_sygna.ia.redes): quantas vezes o treino
# passa por todas as janelas, e quantas janelas por passo.
EPOCAS_REDE = 25
LOTE_REDE = 256
# Máscara no treino (kara_sygna.ia.mascara): cada gravação também entra com a
# mão sumindo nestas partes dos quadros, como acontece ao vivo com a mão
# borrada. () = desliga. Os buracos têm de MIN a MAX quadros (15 ≈ 0,5 s).
# Desligada (laboratório, 19/10/2026): com (0.15, 0.3, 0.5) a validação caiu
# de 99,0% para 85,8%, e nos 11 vídeos de teste o modelo sem máscara foi igual
# ou melhor em 9.
MASCARA_FRACOES = ()
MASCARA_BURACO_MIN = 2
MASCARA_BURACO_MAX = 15
# Durante o ensino, os quadros ficam na memória e vão para o disco ao parar
# (ou a cada tantos quadros, em gravações longas). Antes de gravar, as mãos
# que sumiram por pouco tempo são preenchidas entre antes e depois.
QUADROS_POR_LOTE_GRAVACAO = 900
# Maior buraco de uma mão preenchido na gravação, em segundos (0 = nenhum).
PREENCHER_BURACO_S = 0.5
# Ao começar a ensinar, espera estes segundos antes de gravar (tempo para se
# posicionar e já começar o sinal).
SEGUNDOS_CONTAGEM_ENSINO = 3
# "Gravação rápida" (botão ao lado dos segundos): a espera entre uma gravação
# e a próxima (0 = sem parar). A primeira tem a contagem normal.
PAUSA_GRAVACAO_RAPIDA_S = 0.0
# Quantas vezes a "Gravação rápida" grava, se o campo estiver vazio ou errado.
VEZES_GRAVACAO_RAPIDA = 4

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
INTERVALO_QUADRO_MS = 15
# Quantos detectores de MÃOS analisam quadros ao mesmo tempo, em revezamento
# (kara_sygna.visao.revezamento); rosto e braços ficam numa thread só. 1 = um
# quadro por vez, como antes (~15 qps ao vivo). Laboratório, 19/10/2026, com
# câmera falsa de 30 qps: 2 detectores = 25 qps, 3 = 30 qps (o limite da
# câmera); nos vídeos de teste, 2, 3 e 4 tiveram a mesma qualidade de um
# detector vendo todos os quadros, e as trocas rápidas carro→casa passaram a
# ser todas reconhecidas.
REVEZAMENTO_DETECTORES = 3
# Tamanho da janela principal ao abrir (menor, se a tela for menor).
LARGURA_JANELA_PADRAO = 1024
ALTURA_JANELA_PADRAO = 768
LARGURA_AREA_CAMERA = 900
ALTURA_AREA_CAMERA = 510
