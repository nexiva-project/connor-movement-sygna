# Guia de Padrões Git

Este documento define o fluxo de trabalho, a convenção de nomenclatura de *branches* e as boas práticas para o versionamento de código do projeto.

---

## 🌿 1. Convenção de Nomenclatura de Branches

Toda *branch* deve ser nomeada seguindo a estrutura:

```text
<tipo>/<escopo>-<descricao-curta>
```

- **`<tipo>`**: Categoria da alteração realizada (veja a tabela abaixo).
- **`<escopo>`** *(Opcional, mas recomendado)*: Módulo, componente ou área afetada (ex: `auth`, `checkout`, `ui`, `api`).
- **`<descricao-curta>`**: Resumo claro do objetivo da branch em formato *kebab-case* (letras minúsculas separadas por hífen, sem acentos ou caracteres especiais).

---

### Tabela de Tipos de Branches

| Tipo | Descrição | Exemplo de Nome da Branch |
| :--- | :--- | :--- |
| **`feat`** | Desenvolvimento de uma **nova funcionalidade** ou recurso para o usuário. | `feat/auth-login-social` |
| **`fix`** | Correção de um **bug** ou erro reportado no sistema. | `fix/checkout-calculo-frete` |
| **`test`** | Criação, ajuste ou manutenção de **testes** (unitários, integração, E2E). | `test/api-fluxo-pagamento` |
| **`refactor`** | Reformulação de código que **não** altera o comportamento externo (melhoria de arquitetura, clareza ou legibilidade). | `refactor/utils-formatadores` |
| **`docs`** | Alterações exclusivas na **documentação** (README, guias, tutoriais). | `docs/atualiza-guia-instalacao` |
| **`chore`** | Tarefas de **manutenção geral**, build, instalação/atualização de dependências ou configurações de ferramentas. | `chore/atualiza-dependencias` |
| **`perf`** | Mudanças voltadas para a melhoria de **performance** ou otimização de recursos. | `perf/query-listagem-usuarios` |
| **`style`** | Modificações puramente estéticas ou de **formatação** (espaçamentos, ponto e vírgula, linters) que não afetam a lógica. | `style/padroniza-linter-tabelas` |
| **`experiment`** | Provas de conceito (PoC), protótipos ou testes de novas ideias/tecnologias que podem ser descartadas. | `experiment/novo-layout-dashboard` |
| **`hotfix`** | Correção **urgente** aplicada diretamente em ambiente de produção. | `hotfix/vulnerabilidade-sessao` |

---

## 🚀 2. Estrutura de Branches Principais

* **`main`** (ou `master`): Contém o código estável atualmente em **produção**.
  * *Regra:* Nenhum commit direto deve ser feito na `main`. Todo código entra via Pull Request (PR).
* **`develop`**: Branch de **integração** das novas funcionalidades e melhorias.
  * *Regra:* É a branch base a partir da qual novas branches (`feat`, `fix`, etc.) devem ser criadas.

---

## 📝 3. Padrão de Commits (Conventional Commits)

Os commits devem ser mantidos pequenos, atômicos e com mensagens no formato:

```text
<tipo>(<escopo>): <descrição no imperativo e em minúsculas>
```

### Exemplos:
* `feat(auth): adiciona autenticação via Google OAuth`
* `fix(cart): corrige estouro de memória ao limpar o carrinho`
* `test(user): adiciona cenários de teste para cadastro de usuário`
* `docs(readme): atualiza instruções para rodar o projeto localmente`

---

## 🔄 4. Fluxo de Trabalho Passo a Passo

1. **Sincronize a branch principal local:**
   ```bash
   git checkout develop
   git pull origin develop
   ```

2. **Crie sua nova branch de trabalho:**
   ```bash
   git checkout -b feat/painel-notificacoes
   ```

3. **Realize os commits com frequência:**
   ```bash
   git add .
   git commit -m "feat(notifications): cria componente visual de alertas"
   ```

4. **Suba a branch para o repositório remoto:**
   ```bash
   git push origin feat/painel-notificacoes
   ```

5. **Abra um Pull Request (PR):**
   - Direcione o PR da sua branch para a branch `develop`.
   - Preencha o resumo com o contexto da alteração e solicite a revisão (Code Review) da equipe.