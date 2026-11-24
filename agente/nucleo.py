"""Lógica do agente: integra LLM + ferramentas + ReAct + memória.

Fluxo de uma pergunta:
    1. o histórico recente (memória) e a pergunta entram no prompt ReAct;
    2. o AgentExecutor alterna Thought -> Action -> Observation até o modelo
       emitir "Final Answer" (ou atingir o limite de iterações);
    3. a troca é salva na memória e os passos intermediários são devolvidos
       para que a interface mostre o raciocínio.
"""

from dataclasses import dataclass, field

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseLanguageModel

from agente.config import Configuracao
from agente.ferramentas import criar_ferramentas
from agente.llm import criar_llm
from agente.memoria import MemoriaConversacional
from agente.prompts import MENSAGEM_ERRO_FORMATO, PROMPT_REACT

RESPOSTA_LIMITE_ITERACOES = "Agent stopped due to iteration limit or time limit."


@dataclass
class PassoReAct:
    """Um ciclo Thought -> Action -> Observation."""

    pensamento: str
    ferramenta: str
    entrada: str
    observacao: str


@dataclass
class RespostaAgente:
    texto: str
    passos: list[PassoReAct] = field(default_factory=list)
    erro: str | None = None

    @property
    def ferramentas_usadas(self) -> list[str]:
        return [p.ferramenta for p in self.passos if p.ferramenta != "_Exception"]


def _extrair_pensamento(log: str) -> str:
    """Isola o texto do 'Thought' do log bruto da ação."""
    return log.split("Action:")[0].replace("Thought:", "").strip()


class AgentePetVerde:
    def __init__(self, config: Configuracao | None = None, llm: BaseLanguageModel | None = None):
        """`llm` pode ser injetado (ex.: modelo falso nos testes); senão vem da configuração."""
        self.config = config or Configuracao.carregar()
        self.llm = llm or criar_llm(self.config)
        self.ferramentas = criar_ferramentas()
        self.memoria = MemoriaConversacional(self.config.janela_memoria)

        agente_react = create_react_agent(self.llm, self.ferramentas, PROMPT_REACT)
        self.executor = AgentExecutor(
            agent=agente_react,
            tools=self.ferramentas,
            max_iterations=self.config.max_iteracoes,
            handle_parsing_errors=MENSAGEM_ERRO_FORMATO,
            return_intermediate_steps=True,
            verbose=False,
        )

    def perguntar(self, pergunta: str, callbacks: list | None = None) -> RespostaAgente:
        """Executa o ciclo ReAct para uma pergunta e atualiza a memória."""
        pergunta = (pergunta or "").strip()
        if not pergunta:
            return RespostaAgente(texto="Digite uma pergunta sobre o seu pet.", erro="pergunta vazia")

        try:
            resultado = self.executor.invoke(
                {"input": pergunta, "chat_history": self.memoria.como_texto()},
                config={"callbacks": callbacks or []},
            )
        except Exception as erro:  # falha do provedor (chave inválida, cota, rede...)
            return RespostaAgente(
                texto=(
                    "Não consegui falar com o modelo de linguagem agora. "
                    "Verifique a chave de API/conexão e tente de novo."
                ),
                erro=f"{type(erro).__name__}: {erro}",
            )

        passos = [
            PassoReAct(
                pensamento=_extrair_pensamento(acao.log),
                ferramenta=acao.tool,
                entrada=str(acao.tool_input),
                observacao=str(observacao),
            )
            for acao, observacao in resultado.get("intermediate_steps", [])
        ]
        texto = str(resultado.get("output", "")).strip()
        if texto == RESPOSTA_LIMITE_ITERACOES:
            texto = (
                "Não cheguei a uma conclusão dentro do limite de passos. "
                "Tente reformular a pergunta com mais detalhes (espécie, peso, ração)."
            )

        self.memoria.registrar(pergunta, texto)
        return RespostaAgente(texto=texto, passos=passos)

    def reiniciar_conversa(self) -> None:
        self.memoria.limpar()
