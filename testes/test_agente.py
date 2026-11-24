"""Testes automatizados (offline): ferramentas, memória e ciclo ReAct.

O ciclo ReAct é testado com um LLM falso (FakeListLLM) que devolve respostas
roteirizadas — assim validamos o encadeamento Thought/Action/Observation e o
uso da memória sem gastar chamadas de API.

Execução:  python -m pytest -q
"""

from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.language_models.fake import FakeListLLM

from agente.config import Configuracao
from agente.ferramentas import (
    calcular_porcao_diaria,
    buscar_racao,
    estimar_pegada_carbono,
    interpretar_entrada,
    pesquisar_wikipedia,
)
from agente.memoria import HISTORICO_VAZIO, MemoriaConversacional
from agente.nucleo import AgentePetVerde

CONFIG_TESTE = Configuracao(
    provedor="ollama",
    modelo="falso",
    temperatura=0,
    max_iteracoes=5,
    janela_memoria=2,
    ollama_url="http://localhost:11434",
    google_api_key=None,
)


# --- interpretação de entradas ------------------------------------------------
def test_interpretar_entrada_json_e_chave_valor():
    assert interpretar_entrada('{"especie": "cao", "peso_kg": 10}') == {"especie": "cao", "peso_kg": "10"}
    assert interpretar_entrada("especie=gato, peso_kg=4,5")["especie"] == "gato"
    assert interpretar_entrada("```json\n{\"a\": 1}\n```") == {"a": "1"}


# --- calcular_porcao_diaria -----------------------------------------------------
def test_porcao_cao_adulto_castrado_10kg():
    # RER = 70 * 10^0.75 = 393.6 ; MER = 1.6 * RER = 629.8 ; 629.8 / 3.7 = 170 g
    saida = calcular_porcao_diaria('{"especie": "cão", "peso_kg": "10 kg", "fase": "adulto castrado"}')
    assert "394 kcal/dia" in saida
    assert "630 kcal/dia" in saida
    assert "170 g/dia" in saida


def test_porcao_usa_kcal_do_rotulo():
    saida = calcular_porcao_diaria("especie=gato, peso_kg=4, fase=adulto_castrado, kcal_por_kg=4000")
    assert "informada no rótulo" in saida


@pytest.mark.parametrize(
    "entrada, trecho_erro",
    [
        ('{"especie": "papagaio", "peso_kg": 1}', "'especie'"),
        ('{"especie": "cao", "peso_kg": 500}', "entre 0,3 e 100"),
        ('{"especie": "gato", "peso_kg": 4, "fase": "ativo"}', "'fase'"),
        ("qualquer coisa sem formato", "chave=valor"),
    ],
)
def test_porcao_entradas_invalidas_viram_erro_legivel(entrada, trecho_erro):
    saida = calcular_porcao_diaria(entrada)
    assert saida.startswith("ERRO")
    assert trecho_erro in saida


# --- estimar_pegada_carbono -----------------------------------------------------
def test_pegada_carbono_racao_seca():
    # 200 g/dia * 30 = 6 kg/mês * 3.0 = 18 kg CO2e
    saida = estimar_pegada_carbono('{"gramas_por_dia": 200, "tipo_alimento": "seca"}')
    assert "6.0 kg/mês" in saida
    assert "18.0 kg CO2e/mês" in saida


def test_pegada_carbono_exige_quantidade():
    assert estimar_pegada_carbono('{"tipo_alimento": "seca"}').startswith("ERRO")


# --- ferramentas HTTP (requests simulado) ---------------------------------------
def _resposta_http(json_dados):
    resposta = MagicMock()
    resposta.json.return_value = json_dados
    resposta.raise_for_status.return_value = None
    return resposta


def test_buscar_racao_formata_produtos():
    dados = {"count": 1, "products": [{"product_name": "Golden Filhote", "brands": "Golden"}]}
    with patch("agente.ferramentas.requests.get", return_value=_resposta_http(dados)):
        saida = buscar_racao("Golden")
    assert "Golden Filhote" in saida and "marca: Golden" in saida


def test_buscar_racao_trata_timeout():
    with patch("agente.ferramentas.requests.get", side_effect=requests.Timeout()):
        assert buscar_racao("Golden").startswith("ERRO: o serviço externo demorou")


def test_wikipedia_sem_resultados():
    with patch("agente.ferramentas.requests.get", return_value=_resposta_http({"query": {"search": []}})):
        assert "Nenhum verbete" in pesquisar_wikipedia("xyzabc")


# --- memória ---------------------------------------------------------------------
def test_memoria_respeita_janela():
    memoria = MemoriaConversacional(janela_trocas=2)
    assert memoria.como_texto() == HISTORICO_VAZIO
    for i in range(3):
        memoria.registrar(f"pergunta {i}", f"resposta {i}")
    texto = memoria.como_texto()
    assert "pergunta 0" not in texto  # caiu fora da janela
    assert "Tutor: pergunta 2" in texto and "PetVerde: resposta 2" in texto
    assert len(memoria) == 3


# --- ciclo ReAct completo com LLM falso ------------------------------------------
def test_ciclo_react_usa_ferramenta_e_memoria():
    respostas_roteirizadas = [
        # 1ª pergunta: o modelo raciocina, chama a tool e depois conclui
        ' Preciso calcular a porção.\nAction: calcular_porcao_diaria\n'
        'Action Input: {"especie": "cao", "peso_kg": 10, "fase": "adulto_castrado"}',
        " Já tenho o resultado.\nFinal Answer: Cerca de 170 g por dia.",
        # 2ª pergunta: responde direto (o dado vem da memória)
        " A informação já está no histórico.\nFinal Answer: Seu cão pesa 10 kg.",
    ]
    llm_falso = FakeListLLM(responses=respostas_roteirizadas)
    agente = AgentePetVerde(config=CONFIG_TESTE, llm=llm_falso)

    primeira = agente.perguntar("Meu cão castrado tem 10 kg. Quanto ele come?")
    assert primeira.erro is None
    assert primeira.ferramentas_usadas == ["calcular_porcao_diaria"]
    assert "170 g/dia" in primeira.passos[0].observacao
    assert primeira.texto == "Cerca de 170 g por dia."

    segunda = agente.perguntar("Quanto ele pesa mesmo?")
    assert segunda.passos == []
    assert len(agente.memoria) == 2
    assert "Meu cão castrado tem 10 kg" in agente.memoria.como_texto()


def test_pergunta_vazia_nao_chama_llm():
    agente = AgentePetVerde(config=CONFIG_TESTE, llm=FakeListLLM(responses=["nunca usado"]))
    resposta = agente.perguntar("   ")
    assert resposta.erro == "pergunta vazia"
    assert len(agente.memoria) == 0
