# Changelog

## [0.5.0] - 2026-10-11

### Corrigido
- Webcam no Linux/Ubuntu: o código abria a câmera sempre com `cv2.CAP_DSHOW`, que só existe no Windows. Agora cada sistema usa os seus backends (Windows: DirectShow/MediaFoundation; Linux: V4L2; macOS: AVFoundation), com o automático do OpenCV como último recurso.
- Em notebooks Linux, `/dev/video1` costuma ser só metadados. Cada câmera só é aceita se entregar um quadro de verdade (com algumas tentativas de aquecimento).
- Digitar a letra "q" no campo da palavra fechava o programa. O atalho foi removido.
- Um sinal repetido ("um um") não podia ser escrito duas vezes; agora basta tirar a mão (ou trocar de sinal) entre as repetições.
- O CSV antigo (sem a coluna `rodada`) agora é convertido automaticamente, com backup, em vez de ser movido para o lado.
- A pré-visualização do sinal abria uma janela do OpenCV (instável junto com o Tk no Linux); agora é desenhada dentro de uma janela do Tk.
- O treinamento rodava na thread da interface e congelava a câmera; agora roda em segundo plano.
- Mensagens de erro da câmera agora listam cada tentativa e dão dicas (`ls /dev/video*`, grupo `video`).

### Refatorado
- O `interface_sinais.py` (≈1600 linhas, uma classe fazendo tudo) foi dividido em pacotes por responsabilidade: `camera`, `visao`, `dados`, `ia`, `aplicacao`, `interface` e `cli`.
- Regras do programa (`aplicacao/controlador.py`) separadas da interface Tk, permitindo testes sem câmera nem tela.
- Caminhos ancorados na raiz do projeto; CSV e modelo ficam em `armazenamento/`.
- Log feito com o módulo `logging` do Python (mesmo formato no terminal e na janela de log) em vez de um módulo global.
- Constantes e limiares reunidos em `config.py`.
- Todas as funções, classes e métodos têm docstring (estilo Google, em português).

### Adicionado
- Suíte de testes automatizados (`pytest`) e testes manuais de câmera em `testes/manuais/`.
- Validação do modelo por rodada de gravação, com aumento de dados só no treino.
- Votação em janela de quadros para confirmar sinais, com margem mínima entre o 1º e o 2º mais provável.
- Botão "ABRIR LOG" (a janela de log começa oculta).
- Scripts de terminal: `python -m kara_sygna.cli.coletar_dados | treinar_modelo | reconhecer_sinais`.

### Removido
- Código morto da mão 3D ao vivo e atalhos de teclado que não faziam nada.
- Scripts antigos com imports quebrados (`dados.py`, `aumentar_dados.py` etc. foram reescritos dentro dos pacotes).

### Dependências
- `opencv-python` substituído por `opencv-contrib-python` (não instale os dois juntos).
- Adicionado `pillow`; `numpy<2` fixado.

## [0.4.1] - 2026-10-02

### Corrigido
- Bug no Linux/Ubuntu em que `cv2.imshow` abria múltiplas janelas de câmera, causado por conflito entre os backends gráficos do OpenCV (GTK) e do Tkinter.

### Refatorado
- Vídeo da câmera agora é renderizado dentro de um widget `Label` do Tkinter, embutido na própria janela do painel, em vez de uma janela separada do OpenCV.

### Removido
- Uso de `cv2.imshow` e `cv2.waitKey` para exibir a câmera (substituídos pela renderização via Tkinter).

### Dependências
- Adicionado `pillow` ao `requirements.txt` (necessário para converter frames da câmera em imagens exibíveis pelo Tkinter).

## [0.4.0] - 2026-09-18

### Adicionado
- Painel de controle único para a aplicação.
- Modo "Ensinar" para coletar novos sinais diretamente pelo painel.
- Treinamento da IA diretamente pelo painel.
- Modo "Reconhecimento" para identificar sinais em tempo real.
- Lista de sinais aprendidos.
- Controle para iniciar e interromper o ensino e o reconhecimento.

### Refatorado
- Centralização do fluxo da aplicação no `app.py`.
- Integração entre coleta de dados, treinamento e reconhecimento.
- Atualização da câmera pelo loop do Tkinter, sem utilização de threads.
- Remove o módulo `pipeline.py`

### Interface
- Adicionados controles para:
  - Começar a ensinar.
  - Parar.
  - Treinar IA.
  - Iniciar reconhecimento.

## [0.3.0] - 2026-09-18

### Adicionado
- Painel de controle único desenvolvido com Tkinter.
- Botões para executar as principais funcionalidades do projeto.
- A câmera agora é iniciada pelo painel de controle.
- Interface gráfica separada da lógica de processamento.

### Refatorado
- Criação do módulo `pipeline.py` para centralizar a lógica do projeto.
- Criação do módulo `app.py` para controlar a interface gráfica.
- Reaproveitamento da lógica existente no novo fluxo.

### Estrutura
- `app.py`: interface gráfica e controles da aplicação.
- `pipeline.py`: lógica de processamento e execução do pipeline.

## [0.2.5] - 2026-09-11

### Adicionado
- Suporte à detecção de até duas mãos.
- Suavização temporal das previsões.

### Corrigido
- Reconhecimento entre mão esquerda e direita.

## [0.1.0] - 2026-09-08

### Adicionado
- Primeiro modelo de reconhecimento.
- Coleta de landmarks.
- Treinamento do modelo.
