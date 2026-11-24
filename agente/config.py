"""Configuração central lida de variáveis de ambiente (arquivo .env).

Mantém em um único lugar o provedor de LLM, o modelo e os limites do agente,
para que interface, CLI e testes usem exatamente os mesmos parâmetros.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

PROVEDORES_SUPORTADOS = ("gemini", "ollama")


class ErroConfiguracao(Exception):
    """Configuração ausente ou inválida (ex.: chave de API não definida)."""


@dataclass(frozen=True)
class Configuracao:
    provedor: str
    modelo: str
    temperatura: float
    max_iteracoes: int
    janela_memoria: int
    ollama_url: str
    google_api_key: str | None

    @classmethod
    def carregar(cls) -> "Configuracao":
        provedor = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        if provedor not in PROVEDORES_SUPORTADOS:
            raise ErroConfiguracao(
                f"LLM_PROVIDER='{provedor}' inválido. Use um de: {', '.join(PROVEDORES_SUPORTADOS)}."
            )

        modelo_padrao = "gemini-2.5-flash" if provedor == "gemini" else "qwen2.5:7b"
        try:
            return cls(
                provedor=provedor,
                modelo=os.getenv("LLM_MODEL", modelo_padrao),
                temperatura=float(os.getenv("LLM_TEMPERATURE", "0")),
                max_iteracoes=int(os.getenv("AGENT_MAX_ITERATIONS", "8")),
                janela_memoria=int(os.getenv("MEMORY_WINDOW", "8")),
                ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
        except ValueError as erro:
            raise ErroConfiguracao(f"Valor numérico inválido no .env: {erro}") from erro
