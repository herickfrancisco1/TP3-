"""Interface Streamlit do PetVerde Agente.

Execução:  streamlit run app.py
"""

import streamlit as st

from agente import AgentePetVerde
from agente.config import ErroConfiguracao

EXEMPLOS = [
    "Tenho uma cadela SRD castrada de 12 kg. Quantos gramas de ração seca ela deve comer por dia?",
    "E qual a pegada de carbono mensal dessa alimentação? Compare com ração úmida.",
    "Procure rações da marca Golden e diga quais ingredientes aparecem.",
    "O que é a raça Border Collie e quanto um adulto ativo de 18 kg come por dia?",
]

st.set_page_config(page_title="PetVerde Agente", page_icon="🐾", layout="centered")


@st.cache_resource(show_spinner=False)
def _verificar_configuracao() -> str | None:
    """Tenta criar um agente uma vez só para validar .env e provedor."""
    try:
        AgentePetVerde()
    except ErroConfiguracao as erro:
        return str(erro)
    return None


def obter_agente() -> AgentePetVerde:
    """Um agente (e portanto uma memória) por sessão do navegador."""
    if "agente" not in st.session_state:
        st.session_state.agente = AgentePetVerde()
        st.session_state.mensagens = []
    return st.session_state.agente


def mostrar_raciocinio(passos) -> None:
    """Exibe os ciclos Thought/Action/Observation de uma resposta."""
    if not passos:
        st.caption("Respondido sem usar ferramentas.")
        return
    with st.expander(f"Raciocínio ReAct ({len(passos)} passo(s))"):
        for numero, passo in enumerate(passos, start=1):
            st.markdown(f"**Passo {numero}**")
            if passo.pensamento:
                st.markdown(f"💭 *Pensamento:* {passo.pensamento}")
            st.markdown(f"🔧 *Ação:* `{passo.ferramenta}` — entrada: `{passo.entrada}`")
            st.code(passo.observacao, language=None)


def processar(pergunta: str) -> None:
    agente = obter_agente()
    st.session_state.mensagens.append({"papel": "user", "texto": pergunta, "passos": []})
    with st.spinner("Pensando, consultando ferramentas..."):
        resposta = agente.perguntar(pergunta)
    st.session_state.mensagens.append(
        {"papel": "assistant", "texto": resposta.texto, "passos": resposta.passos, "erro": resposta.erro}
    )


# ---------------------------------------------------------------------------
st.title("🐾 PetVerde Agente")
st.write(
    "Assistente de consumo consciente para tutores de cães e gatos: calcula porções, "
    "busca rações reais na Open Pet Food Facts, estima a pegada de carbono da alimentação "
    "e consulta a Wikipédia — lembrando o que você já contou sobre o seu pet."
)

erro_config = _verificar_configuracao()
if erro_config:
    st.error(f"Configuração incompleta: {erro_config}")
    st.stop()

agente = obter_agente()

with st.sidebar:
    st.subheader("Configuração")
    st.write(f"Provedor: **{agente.config.provedor}**")
    st.write(f"Modelo: `{agente.config.modelo}`")
    st.write(f"Memória: últimas {agente.config.janela_memoria} trocas ({len(agente.memoria)} salvas)")
    st.subheader("Ferramentas")
    for ferramenta in agente.ferramentas:
        st.markdown(f"- `{ferramenta.name}`")
    if st.button("Nova conversa (limpar memória)"):
        agente.reiniciar_conversa()
        st.session_state.mensagens = []
        st.rerun()
    st.subheader("Exemplos")
    for exemplo in EXEMPLOS:
        if st.button(exemplo, use_container_width=True):
            processar(exemplo)
            st.rerun()

for mensagem in st.session_state.mensagens:
    with st.chat_message(mensagem["papel"]):
        st.markdown(mensagem["texto"])
        if mensagem["papel"] == "assistant":
            if mensagem.get("erro"):
                st.caption(f"Detalhe técnico: {mensagem['erro']}")
            mostrar_raciocinio(mensagem["passos"])

if pergunta := st.chat_input("Pergunte sobre a alimentação do seu pet..."):
    processar(pergunta)
    st.rerun()
