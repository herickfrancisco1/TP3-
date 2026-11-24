"""Fábrica do modelo de linguagem usado pelo agente.

Suporta Gemini (nuvem, padrão) e Ollama (local, sem custo), escolhidos pela
variável LLM_PROVIDER. Os imports ficam dentro das funções para que só o
pacote do provedor escolhido precise estar funcional.
"""

from langchain_core.language_models import BaseChatModel

from agente.config import Configuracao, ErroConfiguracao


def criar_llm(config: Configuracao) -> BaseChatModel:
    """Instancia o chat model do provedor configurado."""
    if config.provedor == "gemini":
        if not config.google_api_key:
            raise ErroConfiguracao(
                "GOOGLE_API_KEY não definida. Crie uma chave gratuita em "
                "https://aistudio.google.com/apikey e coloque-a no arquivo .env "
                "(ou use LLM_PROVIDER=ollama para rodar localmente)."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.modelo,
            temperature=config.temperatura,
            google_api_key=config.google_api_key,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.modelo,
        temperature=config.temperatura,
        base_url=config.ollama_url,
    )
