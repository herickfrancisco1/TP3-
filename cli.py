"""Interface de linha de comando do PetVerde Agente.

Execução:  python cli.py           (mostra o raciocínio ReAct)
           python cli.py --silencioso
Comandos:  /limpar  apaga a memória | /sair  encerra
"""

import argparse
import sys

from agente import AgentePetVerde
from agente.config import ErroConfiguracao


def imprimir_passos(resposta) -> None:
    for numero, passo in enumerate(resposta.passos, start=1):
        print(f"  [{numero}] Pensamento: {passo.pensamento}")
        print(f"      Ação: {passo.ferramenta}({passo.entrada})")
        observacao = passo.observacao.replace("\n", "\n      ")
        print(f"      Observação: {observacao}")


def main() -> int:
    parser = argparse.ArgumentParser(description="PetVerde Agente (LangChain + ReAct)")
    parser.add_argument("--silencioso", action="store_true", help="oculta os passos de raciocínio")
    args = parser.parse_args()

    try:
        agente = AgentePetVerde()
    except ErroConfiguracao as erro:
        print(f"Erro de configuração: {erro}", file=sys.stderr)
        return 1

    print(f"PetVerde Agente — {agente.config.provedor}/{agente.config.modelo}")
    print("Digite sua pergunta (/limpar para nova conversa, /sair para encerrar).\n")
    while True:
        try:
            pergunta = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not pergunta:
            continue
        if pergunta.lower() in {"/sair", "sair", "exit"}:
            break
        if pergunta.lower() == "/limpar":
            agente.reiniciar_conversa()
            print("Memória apagada.\n")
            continue

        resposta = agente.perguntar(pergunta)
        if not args.silencioso:
            imprimir_passos(resposta)
        print(f"\nPetVerde: {resposta.texto}\n")
        if resposta.erro:
            print(f"(detalhe técnico: {resposta.erro})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
