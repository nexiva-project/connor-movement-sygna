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
# 0.5: com 0.7, mãos borradas pelo movimento eram descartadas (no teste com
# o sinal "carro", 42% dos quadros ficavam sem nenhuma mão; com 0.5, 24%).
CONFIANCA_DETECCAO = 0.5
CONFIANCA_RASTREIO = 0.5
# Se uma mão some por até estes segundos (quadro borrado), continua valendo a
# última posição dela, no desenho e na IA. 0 = desliga.
MEMORIA_MAO_S = 0.3

# O frame é espelhado antes de ser enviado ao MediaPipe; com isso os rótulos
# "Left"/"Right" saem trocados nesta configuração. Mudar este valor altera o
# significado dos dados já gravados: se mudar, grave os sinais de novo.
INVERTER_LATERALIDADE = True

# Cores (BGR, como o OpenCV espera) usadas para desenhar cada mão.
CORES_MAOS_BGR = {"esq": (70, 190, 255), "dir": (150, 105, 255)}

# ---------------------------------------------------------------------------
# Reconhecimento
# ---------------------------------------------------------------------------
CONFIANCA_MINIMA = 0.70  # probabilidade mínima do melhor sinal
MARGEM_MINIMA = 0.15  # melhor - segundo melhor (evita empates)
JANELA_ESTABILIZADOR = 10  # quadros considerados na votação
VOTOS_MINIMOS = 6  # votos necessários para confirmar um sinal
QUADROS_PARA_LIBERAR = 8  # quadros sem o sinal para poder repeti-lo
# Rótulos aprendidos pela IA que NÃO viram texto (ex.: mão relaxada).
ROTULOS_IGNORADOS = frozenset({"neutro", "nada"})

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
# Durante o ensino, os quadros são gravados no disco em lotes deste tamanho.
QUADROS_POR_LOTE_GRAVACAO = 30
# Ao começar a ensinar, espera estes segundos antes de gravar (tempo para se
# posicionar e já começar o sinal).
SEGUNDOS_CONTAGEM_ENSINO = 3

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
INTERVALO_QUADRO_MS = 15
LARGURA_AREA_CAMERA = 900
ALTURA_AREA_CAMERA = 510
