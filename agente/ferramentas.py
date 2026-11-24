"""Ferramentas (tools) externas disponíveis ao agente ReAct.

Cada ferramenta recebe UMA string (formato exigido pelo ReAct textual: a linha
"Action Input" do modelo) e devolve uma string que o agente lê como
"Observation". Erros nunca são lançados para o agente: viram uma mensagem
começando com "ERRO:" para que o modelo possa raciocinar e tentar de novo.

Ferramentas:
    buscar_racao           -> API pública Open Pet Food Facts (dados reais de produtos)
    calcular_porcao_diaria -> cálculo veterinário de energia (RER/MER) e gramas por dia
    estimar_pegada_carbono -> estimativa de CO2e mensal da alimentação do pet
    pesquisar_wikipedia    -> resumo de verbetes da Wikipédia em português
"""

import json
import re
from typing import Any, Callable

import requests
from langchain_core.tools import Tool

TIMEOUT_HTTP_S = 15
# Identificação genérica pedida pelas APIs abertas (sem dados pessoais).
CABECALHOS_HTTP = {"User-Agent": "PetVerdeAgente/1.0 (projeto academico; LangChain)"}

URL_OPEN_PET_FOOD_FACTS = "https://world.openpetfoodfacts.org/cgi/search.pl"
URL_WIKIPEDIA_API = "https://pt.wikipedia.org/w/api.php"
URL_WIKIPEDIA_RESUMO = "https://pt.wikipedia.org/api/rest_v1/page/summary/{titulo}"


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
class ErroEntrada(ValueError):
    """Entrada da ferramenta em formato inválido."""


def interpretar_entrada(texto: str) -> dict[str, str]:
    """Converte o Action Input em dicionário.

    Aceita JSON ('{"peso_kg": 10}') ou pares chave=valor separados por
    vírgula/ponto e vírgula ('especie=cao, peso_kg=10'). Chaves são
    normalizadas para minúsculas; valores viram string.
    """
    texto = (texto or "").strip().strip("`").strip()
    if texto.lower().startswith("json"):
        texto = texto[4:].strip()

    if texto.startswith("{"):
        try:
            dados = json.loads(texto)
        except json.JSONDecodeError as erro:
            raise ErroEntrada(f"JSON inválido: {erro.msg}") from erro
        return {str(k).strip().lower(): str(v).strip() for k, v in dados.items()}

    pares = {}
    for trecho in re.split(r"[;,\n]", texto):
        if "=" in trecho or ":" in trecho:
            chave, valor = re.split(r"[=:]", trecho, maxsplit=1)
            pares[chave.strip().strip("'\"").lower()] = valor.strip().strip("'\"")
    if not pares:
        raise ErroEntrada("use JSON ou pares chave=valor (ex.: especie=cao, peso_kg=10)")
    return pares


def converter_numero(valor: str, nome_campo: str) -> float:
    """Lê números aceitando vírgula decimal e unidades ('12,5 kg' -> 12.5)."""
    achado = re.search(r"-?\d+(?:[.,]\d+)?", str(valor))
    if not achado:
        raise ErroEntrada(f"'{nome_campo}' deve ser numérico (recebido: '{valor}')")
    return float(achado.group().replace(",", "."))


def normalizar_texto(valor: str) -> str:
    """Minúsculas, sem acentos comuns e com '_' no lugar de espaços."""
    mapa = str.maketrans("áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ", "aaaaeeiooouc" + "aaaaeeiooouc")
    return valor.strip().lower().translate(mapa).replace(" ", "_").replace("-", "_")


def proteger(funcao: Callable[[str], str]) -> Callable[[str], str]:
    """Transforma exceções em mensagens 'ERRO:' legíveis pelo agente."""

    def envoltorio(entrada: str) -> str:
        try:
            return funcao(entrada)
        except ErroEntrada as erro:
            return f"ERRO: entrada inválida — {erro}."
        except requests.Timeout:
            return "ERRO: o serviço externo demorou demais para responder. Tente novamente ou siga sem esse dado."
        except requests.RequestException as erro:
            return f"ERRO: falha de rede ao consultar o serviço externo ({type(erro).__name__})."
        except Exception as erro:  # rede de segurança: o agente nunca deve quebrar por uma tool
            return f"ERRO inesperado na ferramenta: {type(erro).__name__}: {erro}"

    envoltorio.__name__ = funcao.__name__
    envoltorio.__doc__ = funcao.__doc__
    return envoltorio


# ---------------------------------------------------------------------------
# 1) Busca de produtos — Open Pet Food Facts
# ---------------------------------------------------------------------------
def _formatar_produto(produto: dict[str, Any], posicao: int) -> str:
    def campo(chave: str, padrao: str = "não informado") -> str:
        valor = produto.get(chave)
        return str(valor).strip() if valor else padrao

    ingredientes = campo("ingredients_text")
    if len(ingredientes) > 300:
        ingredientes = ingredientes[:300] + "..."
    eco = campo("ecoscore_grade", "sem eco-score").upper()
    return (
        f"{posicao}. {campo('product_name', 'sem nome')} | marca: {campo('brands')} | "
        f"embalagem: {campo('packaging')} | quantidade: {campo('quantity')} | "
        f"eco-score: {eco} | ingredientes: {ingredientes}"
    )


@proteger
def buscar_racao(entrada: str) -> str:
    """Busca rações/petiscos na base aberta Open Pet Food Facts."""
    termo = entrada.strip().strip("\"'")
    if termo.startswith("{"):
        termo = interpretar_entrada(termo).get("termo", "")
    if len(termo) < 2:
        raise ErroEntrada("informe o nome ou a marca do produto (ex.: 'Golden filhote')")

    resposta = requests.get(
        URL_OPEN_PET_FOOD_FACTS,
        params={
            "search_terms": termo,
            "search_simple": 1,
            "json": 1,
            "page_size": 5,
            "fields": "product_name,brands,ingredients_text,ecoscore_grade,packaging,quantity",
        },
        headers=CABECALHOS_HTTP,
        timeout=TIMEOUT_HTTP_S,
    )
    resposta.raise_for_status()
    dados = resposta.json()
    produtos = [p for p in dados.get("products", []) if p.get("product_name")]
    if not produtos:
        return f"Nenhum produto encontrado para '{termo}'. Tente um termo mais curto (só a marca, por exemplo)."

    linhas = [_formatar_produto(p, i) for i, p in enumerate(produtos, start=1)]
    total = dados.get("count", len(produtos))
    return f"{total} resultado(s) na Open Pet Food Facts para '{termo}'. Primeiros:\n" + "\n".join(linhas)


# ---------------------------------------------------------------------------
# 2) Porção diária — fórmula RER/MER (NRC 2006; diretrizes WSAVA)
# ---------------------------------------------------------------------------
# Fatores multiplicadores do RER (energia de repouso) por espécie e fase/condição.
FATORES_ENERGIA = {
    "cao": {
        "filhote_ate_4_meses": 3.0,
        "filhote": 2.0,
        "adulto_castrado": 1.6,
        "adulto": 1.8,
        "sedentario": 1.2,
        "perda_de_peso": 1.0,
        "ativo": 2.0,
        "idoso": 1.4,
    },
    "gato": {
        "filhote": 2.5,
        "adulto_castrado": 1.2,
        "adulto": 1.4,
        "sedentario": 1.0,
        "perda_de_peso": 0.8,
        "idoso": 1.1,
    },
}
SINONIMOS_ESPECIE = {"cao": "cao", "cachorro": "cao", "dog": "cao", "gato": "gato", "felino": "gato", "cat": "gato"}
# Densidade energética típica (kcal por kg do alimento) quando o rótulo não é informado.
ENERGIA_PADRAO_KCAL_KG = {"seca": 3700.0, "umida": 1000.0}


@proteger
def calcular_porcao_diaria(entrada: str) -> str:
    """Calcula kcal/dia e gramas/dia de alimento para cães e gatos."""
    dados = interpretar_entrada(entrada)

    especie = SINONIMOS_ESPECIE.get(normalizar_texto(dados.get("especie", "")))
    if not especie:
        raise ErroEntrada("'especie' deve ser 'cao' ou 'gato'")

    peso_kg = converter_numero(dados.get("peso_kg", dados.get("peso", "")), "peso_kg")
    if not 0.3 <= peso_kg <= 100:
        raise ErroEntrada("'peso_kg' deve estar entre 0,3 e 100 kg")

    fase = normalizar_texto(dados.get("fase", "adulto_castrado"))
    fatores = FATORES_ENERGIA[especie]
    if fase not in fatores:
        raise ErroEntrada(f"'fase' para {especie} deve ser uma de: {', '.join(fatores)}")

    tipo = normalizar_texto(dados.get("tipo_alimento", "seca"))
    if tipo not in ENERGIA_PADRAO_KCAL_KG:
        raise ErroEntrada("'tipo_alimento' deve ser 'seca' ou 'umida'")
    if "kcal_por_kg" in dados:
        kcal_por_kg = converter_numero(dados["kcal_por_kg"], "kcal_por_kg")
        origem_energia = "informada no rótulo"
    else:
        kcal_por_kg = ENERGIA_PADRAO_KCAL_KG[tipo]
        origem_energia = "valor típico, confira o rótulo"
    if kcal_por_kg <= 0:
        raise ErroEntrada("'kcal_por_kg' deve ser positivo")

    rer = 70 * peso_kg**0.75  # energia de repouso (kcal/dia)
    mer = rer * fatores[fase]  # energia de manutenção (kcal/dia)
    gramas_dia = mer / kcal_por_kg * 1000

    return (
        f"Espécie: {especie} | peso: {peso_kg:g} kg | fase: {fase} (fator {fatores[fase]})\n"
        f"Energia de repouso (RER = 70 x peso^0,75): {rer:.0f} kcal/dia\n"
        f"Energia diária recomendada (MER): {mer:.0f} kcal/dia\n"
        f"Alimento {tipo} com {kcal_por_kg:.0f} kcal/kg ({origem_energia}): "
        f"{gramas_dia:.0f} g/dia (~{gramas_dia * 30 / 1000:.1f} kg/mês)\n"
        "Observação: estimativa inicial; ajuste com o veterinário conforme escore corporal."
    )


# ---------------------------------------------------------------------------
# 3) Pegada de carbono da alimentação
# ---------------------------------------------------------------------------
# Fatores aproximados (kg CO2e por kg de alimento, como servido), simplificados
# a partir de faixas da literatura (ex.: Alexander et al., 2020, Global
# Environmental Change; Okin, 2017, PLOS ONE). Uso educacional/comparativo.
FATORES_CO2E_KG = {
    "seca": 3.0,
    "umida": 6.0,
    "caseira_com_carne": 9.0,
    "a_base_de_inseto": 1.5,
    "vegetal": 1.2,
}
CO2E_POR_KM_CARRO = 0.12  # carro popular a gasolina, kg CO2e/km
CO2E_ARVORE_MES = 22 / 12  # absorção média de uma árvore adulta, kg CO2e/mês


@proteger
def estimar_pegada_carbono(entrada: str) -> str:
    """Estima kg CO2e/mês de acordo com o consumo e o tipo de alimento."""
    dados = interpretar_entrada(entrada)

    if "gramas_por_dia" in dados:
        gramas_dia = converter_numero(dados["gramas_por_dia"], "gramas_por_dia")
    elif "kg_por_mes" in dados:
        gramas_dia = converter_numero(dados["kg_por_mes"], "kg_por_mes") * 1000 / 30
    else:
        raise ErroEntrada("informe 'gramas_por_dia' ou 'kg_por_mes'")
    if gramas_dia <= 0:
        raise ErroEntrada("a quantidade de alimento deve ser positiva")

    tipo = normalizar_texto(dados.get("tipo_alimento", "seca"))
    if tipo not in FATORES_CO2E_KG:
        raise ErroEntrada(f"'tipo_alimento' deve ser um de: {', '.join(FATORES_CO2E_KG)}")

    kg_mes = gramas_dia * 30 / 1000
    co2e_mes = kg_mes * FATORES_CO2E_KG[tipo]

    comparacao = [
        f"  - {nome}: {kg_mes * fator:.1f} kg CO2e/mês"
        for nome, fator in sorted(FATORES_CO2E_KG.items(), key=lambda item: item[1])
    ]
    return (
        f"Consumo: {gramas_dia:.0f} g/dia = {kg_mes:.1f} kg/mês de alimento '{tipo}'\n"
        f"Pegada estimada: {co2e_mes:.1f} kg CO2e/mês ({co2e_mes * 12:.0f} kg CO2e/ano)\n"
        f"Equivale a ~{co2e_mes / CO2E_POR_KM_CARRO:.0f} km rodados de carro por mês, "
        f"ou à absorção mensal de ~{co2e_mes / CO2E_ARVORE_MES:.1f} árvore(s).\n"
        "Mesma quantidade em outros tipos de alimento:\n" + "\n".join(comparacao) + "\n"
        "Fatores aproximados para comparação educacional, não são inventário oficial."
    )


# ---------------------------------------------------------------------------
# 4) Wikipédia em português
# ---------------------------------------------------------------------------
@proteger
def pesquisar_wikipedia(entrada: str) -> str:
    """Retorna o resumo do verbete mais relevante da Wikipédia (pt)."""
    termo = entrada.strip().strip("\"'")
    if len(termo) < 2:
        raise ErroEntrada("informe um termo de busca")

    busca = requests.get(
        URL_WIKIPEDIA_API,
        params={"action": "query", "list": "search", "srsearch": termo, "srlimit": 3, "format": "json"},
        headers=CABECALHOS_HTTP,
        timeout=TIMEOUT_HTTP_S,
    )
    busca.raise_for_status()
    resultados = busca.json().get("query", {}).get("search", [])
    if not resultados:
        return f"Nenhum verbete encontrado na Wikipédia para '{termo}'."

    titulo = resultados[0]["title"]
    resumo = requests.get(
        URL_WIKIPEDIA_RESUMO.format(titulo=requests.utils.quote(titulo.replace(" ", "_"))),
        headers=CABECALHOS_HTTP,
        timeout=TIMEOUT_HTTP_S,
    )
    resumo.raise_for_status()
    extrato = resumo.json().get("extract", "").strip() or "(verbete sem resumo)"
    outros = ", ".join(r["title"] for r in resultados[1:])
    return f"Wikipédia — {titulo}: {extrato[:1200]}" + (f"\nOutros verbetes relacionados: {outros}" if outros else "")


# ---------------------------------------------------------------------------
# Registro das ferramentas para o LangChain
# ---------------------------------------------------------------------------
def criar_ferramentas() -> list[Tool]:
    """Lista de tools com nome, descrição (lida pelo LLM) e função."""
    return [
        Tool(
            name="buscar_racao",
            func=buscar_racao,
            description=(
                "Busca rações e petiscos reais na base aberta Open Pet Food Facts. "
                "Retorna marca, ingredientes, embalagem e eco-score. "
                "Entrada: apenas o nome e/ou marca do produto, ex.: Golden filhote"
            ),
        ),
        Tool(
            name="calcular_porcao_diaria",
            func=calcular_porcao_diaria,
            description=(
                "Calcula a energia diária (kcal) e a quantidade de alimento em gramas por dia "
                "para cães e gatos. Entrada em JSON com: especie ('cao' ou 'gato'), peso_kg, "
                "fase (cao: filhote_ate_4_meses, filhote, adulto, adulto_castrado, sedentario, "
                "perda_de_peso, ativo, idoso; gato: filhote, adulto, adulto_castrado, sedentario, "
                "perda_de_peso, idoso), tipo_alimento ('seca' ou 'umida') e, opcional, kcal_por_kg do rótulo. "
                'Ex.: {"especie": "cao", "peso_kg": 12, "fase": "adulto_castrado", "tipo_alimento": "seca"}'
            ),
        ),
        Tool(
            name="estimar_pegada_carbono",
            func=estimar_pegada_carbono,
            description=(
                "Estima a pegada de carbono mensal (kg CO2e) da alimentação de um pet e compara tipos "
                "de alimento. Entrada em JSON com gramas_por_dia (ou kg_por_mes) e tipo_alimento "
                "(seca, umida, caseira_com_carne, a_base_de_inseto, vegetal). "
                'Ex.: {"gramas_por_dia": 180, "tipo_alimento": "seca"}'
            ),
        ),
        Tool(
            name="pesquisar_wikipedia",
            func=pesquisar_wikipedia,
            description=(
                "Pesquisa conceitos gerais na Wikipédia em português (raças, doenças, nutrientes, "
                "sustentabilidade). Entrada: termo de busca curto, ex.: Border collie"
            ),
        ),
    ]
