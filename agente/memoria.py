"""Memória conversacional do agente.

Guarda as mensagens como HumanMessage/AIMessage do LangChain e injeta
no prompt apenas as últimas N trocas (janela deslizante), para que o contexto
não cresça sem limite nem estoure a janela do modelo.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

HISTORICO_VAZIO = "(início da conversa, sem histórico)"


class MemoriaConversacional:
    def __init__(self, janela_trocas: int = 8):
        if janela_trocas < 1:
            raise ValueError("janela_trocas deve ser >= 1")
        self.janela_trocas = janela_trocas
        self.mensagens: list[BaseMessage] = []

    def registrar(self, pergunta: str, resposta: str) -> None:
        """Salva uma troca completa (tutor -> agente)."""
        self.mensagens.extend([HumanMessage(content=pergunta), AIMessage(content=resposta)])

    def mensagens_recentes(self) -> list[BaseMessage]:
        """Últimas N trocas (cada troca = 2 mensagens)."""
        return self.mensagens[-2 * self.janela_trocas :]

    def como_texto(self) -> str:
        """Histórico formatado para o placeholder {chat_history} do prompt ReAct."""
        mensagens = self.mensagens_recentes()
        if not mensagens:
            return HISTORICO_VAZIO
        linhas = []
        for mensagem in mensagens:
            autor = "Tutor" if isinstance(mensagem, HumanMessage) else "PetVerde"
            linhas.append(f"{autor}: {mensagem.content}")
        return "\n".join(linhas)

    def limpar(self) -> None:
        self.mensagens.clear()

    def __len__(self) -> int:
        """Quantidade de trocas registradas."""
        return len(self.mensagens) // 2
