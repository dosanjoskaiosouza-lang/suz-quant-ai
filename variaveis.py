"""
===============================================================================
SUZ Quant AI - Módulo de Definição e Justificativa de Variáveis (variaveis.py)
Disciplina: Mercado Financeiro e de Capitais - PUC-SP
Empresa: Suzano S.A. | Ativo: SUZ (ADR na NYSE)
===============================================================================
Este arquivo define as variáveis econômicas utilizadas no modelo quantitativo,
seus tickers de mercado, fontes de dados, transformações matemáticas e a
fundamentação teórica de cada fator explicativo.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import os


@dataclass
class VariavelEconomica:
    """
    Representa uma variável explicativa ou dependente do modelo quantitativo.
    
    Atributos didáticos para estudantes de Administração:
    - codigo: identificador no modelo (ex: 'Y_SUZ', 'X1_NYA')
    - nome: nome amigável (ex: 'ADR Suzano')
    - ticker: símbolo na base de dados ou Yahoo Finance (ex: 'SUZ', '^NYA')
    - fonte: provedor oficial da série (ex: 'NYSE / Yahoo Finance')
    - papel: 'Dependente (Y)' ou 'Explicativa (X)'
    - hipotese_economica: por que esta variável afeta ou representa o ativo?
    - frequencia_original: periodicidade da coleta (ex: 'Diária')
    - transformacao: cálculo matemático aplicado (ex: 'Retorno percentual simples')
    - limitacoes: riscos, imperfeições ou particularidades da série
    """
    codigo: str
    nome: str
    ticker: str
    fonte: str
    papel: str
    hipotese_economica: str
    frequencia_original: str = "Diária"
    transformacao: str = "Variação percentual diária: (P_t - P_{t-1}) / P_{t-1}"
    limitacoes: str = "Dias sem negociação por feriados locais; fechamento pontual às 16h NY."


# =============================================================================
# DEFINIÇÃO DO CATÁLOGO DE VARIÁVEIS ECONOMICAMENTE JUSTIFICÁVEIS
# =============================================================================

CATALOGO_VARIAVEIS: Dict[str, VariavelEconomica] = {
    # -------------------------------------------------------------------------
    # VARIÁVEL DEPENDENTE (Y)
    # -------------------------------------------------------------------------
    "SUZ": VariavelEconomica(
        codigo="Y_SUZ",
        nome="Suzano S.A. ADR",
        ticker="SUZ",
        fonte="NYSE / Yahoo Finance",
        papel="Dependente (Y)",
        hipotese_economica=(
            "O ADR da Suzano S.A. negociado na Bolsa de Nova York (NYSE) representa "
            "o valor de mercado da maior produtora global de celulose de eucalipto, "
            "precificado em dólares norte-americanos. Variações em seu preço refletem "
            "o risco operacional da empresa, demanda global por papel e celulose, "
            "custos de produção no Brasil, dinâmica da taxa de câmbio e condições "
            "macrofinanceiras globais."
        ),
        limitacoes=(
            "Como é um ADR, reflete tanto os fundamentos intrínsecos da Suzano no Brasil "
            "quanto o apetite a risco de investidores globais em Wall Street. Pode sofrer "
            "descompassos temporários em feriados no Brasil em que a NYSE funciona e vice-versa."
        )
    ),

    # -------------------------------------------------------------------------
    # VARIÁVEL EXPLICATIVA OBRIGATÓRIA (X1)
    # -------------------------------------------------------------------------
    "NYA": VariavelEconomica(
        codigo="X1_NYA",
        nome="NYSE Composite Index",
        ticker="^NYA",
        fonte="NYSE / Yahoo Finance",
        papel="Explicativa Obrigatória (X1)",
        hipotese_economica=(
            "O NYSE Composite cobre todas as ações ordinárias listadas na New York Stock "
            "Exchange, incluindo mais de 2.000 empresas e centenas de ADRs estrangeiros. "
            "Pelo Capital Asset Pricing Model (CAPM), o retorno do mercado geral captura "
            "o risco sistemático (não-diversificável) ao qual qualquer ativo listado nessa bolsa "
            "está sujeito. Espera-se correlação positiva com o ADR da Suzano."
        ),
        limitacoes=(
            "É um índice ponderado por capitalização de mercado, com forte peso no setor financeiro "
            "e industrial americano, podendo responder a choques locais dos EUA que não afetam "
            "diretamente o mercado de commodities florestais."
        )
    ),

    # -------------------------------------------------------------------------
    # VARIÁVEL EXPLICATIVA X2: CÂMBIO
    # -------------------------------------------------------------------------
    "USDBRL": VariavelEconomica(
        codigo="X2_USDBRL",
        nome="Taxa de Câmbio USD/BRL",
        ticker="BRL=X",
        fonte="Mercado Interbancário / Yahoo Finance",
        papel="Explicativa Setorial/Macro (X2)",
        hipotese_economica=(
            "A Suzano é uma empresa predominantemente exportadora: a vasta maioria de sua receita "
            "líquida é atrelada ao dólar (vendas de celulose para China, Europa e América do Norte), "
            "enquanto a maior fatia de seus custos operacionais caixa (madeira, salários industriais, "
            "transporte florestal no Brasil) é incorrida em Reais. Uma desvalorização do Real (alta "
            "do USD/BRL) amplia as margens operacionais em moeda local, embora também majore o "
            "serviço da dívida externa denominada em dólares."
        ),
        limitacoes=(
            "O câmbio opera 24h no mercado internacional, enquanto o mercado de ações tem horário "
            "fixo de pregão. Cotações de fechamento podem capturar momentos ligeiramente distintos."
        )
    ),

    # -------------------------------------------------------------------------
    # VARIÁVEL EXPLICATIVA X3: PROXY DE CELULOSE / PAPEL / FLORESTA
    # -------------------------------------------------------------------------
    "WOOD": VariavelEconomica(
        codigo="X3_WOOD",
        nome="iShares Global Timber & Forestry ETF",
        ticker="WOOD",
        fonte="NASDAQ / BlackRock / Yahoo Finance",
        papel="Explicativa Setorial (X3)",
        hipotese_economica=(
            "Não existe um contrato futuro financeiro padronizado e com alta liquidez diária "
            "negociado em bolsas ocidentais para celulose BEKP (Bleached Eucalyptus Kraft Pulp). "
            "O ETF WOOD é a principal proxy pública global para o setor florestal e de papel & celulose, "
            "investindo nas 25 maiores empresas globais de silvicultura, celulose e embalagens "
            "(incluindo Suzano, Klabin, Smurfit Westrock, Stora Enso e Weyerhaeuser). Ele captura choques "
            "de oferta/demanda global de madeira, celulose e papelão ondulado."
        ),
        limitacoes=(
            "O ETF contém também empresas de papel de imprensa e madeira para construção civil, "
            "cujo ciclo pode pontualmente divergir do ciclo específico da celulose de eucalipto."
        )
    ),

    # -------------------------------------------------------------------------
    # VARIÁVEL EXPLICATIVA X4: JUROS INTERNACIONAIS
    # -------------------------------------------------------------------------
    "US10Y": VariavelEconomica(
        codigo="X4_US10Y",
        nome="Taxa do Tesouro dos EUA 10 Anos (10Y Treasury)",
        ticker="^TNX",
        fonte="Tesouro dos EUA / CBOE / Yahoo Finance",
        papel="Explicativa Macroeconômica (X4)",
        hipotese_economica=(
            "A taxa dos Treasuries de 10 anos é a referência global da taxa livre de risco (Rf) "
            "e custo de oportunidade do capital. A Suzano é uma empresa altamente intensiva em "
            "capital, com investimentos plurianuais de dezenas de bilhões de reais (ex.: Projeto Cerrado) "
            "e endividamento relevante. Uma elevação nos juros longos americanos encarece o "
            "refinanciamento global, eleva o WACC (taxa de desconto de fluxo de caixa futuro) e "
            "estimula a saída de capital de ativos emergentes em direção à renda fixa americana."
        ),
        limitacoes=(
            "O ticker ^TNX representa o rendimento percentual anualizado multiplicado por 10 "
            "(ex.: 43.5 = 4.35% a.a.). A variação percentual diária mede o choque diário nos rendimentos."
        )
    ),

    # -------------------------------------------------------------------------
    # VARIÁVEL CANDIDATA ALTERNATIVA: KLABIN ADR (KLBAY)
    # -------------------------------------------------------------------------
    "KLBAY": VariavelEconomica(
        codigo="X_KLBAY",
        nome="Klabin S.A. ADR",
        ticker="KLBAY",
        fonte="OTC Markets / Yahoo Finance",
        papel="Candidata Alternativa (Setor Nacional)",
        hipotese_economica=(
            "A Klabin é a maior fabricante de papéis e embalagens do Brasil e concorrente no mercado "
            "de celulose de fibra curta e longa. Seu ADR negociado no OTC americano serve como proxy "
            "concorrencial e de risco-país do setor de base florestal brasileiro."
        ),
        limitacoes=(
            "Negociado no mercado de balcão não-organizado (OTC) dos EUA, com menor liquidez diária "
            "e histórico de negociação com alguns pregões sem negócios reportados."
        )
    ),

    # -------------------------------------------------------------------------
    # VARIÁVEL CANDIDATA ALTERNATIVA: RISCO BRASIL (EWZ)
    # -------------------------------------------------------------------------
    "EWZ": VariavelEconomica(
        codigo="X_EWZ",
        nome="iShares MSCI Brazil ETF",
        ticker="EWZ",
        fonte="NYSE Arca / Yahoo Finance",
        papel="Candidata Alternativa (Risco Brasil em USD)",
        hipotese_economica=(
            "ETF que replica o índice de ações brasileiras para investidores internacionais em dólares. "
            "Mede o 'efeito Brasil' e sentimento do investidor estrangeiro em relação a ativos brasileiros."
        ),
        limitacoes=(
            "Fortemente concentrado em Petrobras, Vale e grandes bancos brasileiros, podendo transmitir "
            "ruídos desses setores específicos."
        )
    )
}

# Configuração padrão do sistema: 1 Y + 4 X no máximo
VARIAVEIS_PADRAO = {
    "Y": "SUZ",
    "X1": "NYA",      # Obrigatório conforme enunciado
    "X2": "USDBRL",   # Câmbio USD/BRL
    "X3": "WOOD",     # Proxy setorial Celulose / Florestal
    "X4": "US10Y"     # Juros longos internacionais
}


def obter_especificacao_padrao() -> Dict[str, VariavelEconomica]:
    """Retorna o dicionário de variáveis selecionadas por padrão para o modelo."""
    return {papel: CATALOGO_VARIAVEIS[chave] for papel, chave in VARIAVEIS_PADRAO.items()}


def gerar_documento_justificativa_md(
    caminho_arquivo: str = "outputs/justificativa_variaveis.md",
    vars_selecionadas: Optional[Dict[str, VariavelEconomica]] = None
) -> str:
    """
    Gera o relatório formal em Markdown justificando metodologicamente cada fator
    utilizado nas regressões, conforme exigido no item 4 do trabalho.
    """
    if vars_selecionadas is None:
        vars_selecionadas = obter_especificacao_padrao()

    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)

    linhas = [
        "# SUZ Quant AI — Justificativa Metodológica das Variáveis Selecionadas",
        "",
        "> **Trabalho Universitário:** Mercado Financeiro e de Capitais — PUC-SP  ",
        "> **Ativo Analisado:** Suzano S.A. (ADR: `SUZ` — NYSE)  ",
        "> **Questão Central:** *O grupo recomendaria a compra do ADR da Suzano para revendê-lo em dezembro de 2027?*",
        "",
        "---",
        "",
        "## 1. Princípios de Seleção das Variáveis",
        "",
        "Conforme as melhores práticas de econometria financeira e as diretrizes do trabalho,",
        "**nenhuma variável foi selecionada puramente por apresentar R² elevado ou significância artificial**.",
        "A escolha dos fatores baseou-se rigorosamente na estrutura de receitas, custos, dívida,",
        "mercado consumidor e ambiente macroeconômico em que a Suzano S.A. opera.",
        "",
        "O modelo respeita a restrição estrita de **no máximo 4 variáveis explicativas (X)**,",
        "com a inclusão obrigatória do **NYSE Composite (NYA)** como proxy do mercado acionário.",
        "",
        "---",
        "",
        "## 2. Quadro Resumo dos Fatores",
        "",
        "| Papel | Código | Nome da Série | Ticker | Fonte | Frequência | Transformação Principal |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for papel, var in vars_selecionadas.items():
        linhas.append(
            f"| **{papel}** | `{var.codigo}` | {var.nome} | `{var.ticker}` | {var.fonte} | {var.frequencia_original} | {var.transformacao} |"
        )

    linhas.extend([
        "",
        "---",
        "",
        "## 3. Detalhamento e Hipótese Econômica de Cada Variável",
        ""
    ])

    for papel, var in vars_selecionadas.items():
        linhas.extend([
            f"### {papel}: {var.nome} (`{var.ticker}`)",
            f"- **Papel no Modelo:** {var.papel}",
            f"- **Fonte Oficial dos Dados:** {var.fonte}",
            f"- **Frequência Original:** {var.frequencia_original}",
            f"- **Transformação Aplicada:** {var.transformacao}",
            f"- **Hipótese Econômica:**",
            f"  > {var.hipotese_economica}",
            f"- **Limitações Metodológicas da Série:**",
            f"  > {var.limitacoes}",
            ""
        ])

    linhas.extend([
        "---",
        "",
        "## 4. Candidatas Alternativas Avaliadas",
        "",
        "- **KLBAY (Klabin ADR no OTC):** Avaliada como concorrente direta nacional. Embora rica em informação concorrencial, apresenta menor volume diário de negociação no mercado OTC americano, gerando dias com liquidez reduzida.",
        "- **EWZ (iShares MSCI Brazil ETF):** Avaliado como proxy de risco-país. Foi preterido em favor do câmbio USD/BRL e WOOD para evitar alta multicolinearidade e manter o foco na dinâmica específica de commodities florestais.",
        "",
        "---",
        "*Documento gerado automaticamente pelo módulo `variaveis.py` do SUZ Quant AI.*"
    ])

    conteudo = "\n".join(linhas)
    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        f.write(conteudo)

    return conteudo


if __name__ == "__main__":
    caminho = "outputs/justificativa_variaveis.md"
    gerar_documento_justificativa_md(caminho)
    print(f"[OK] Documento de justificativa gerado com sucesso em: {caminho}")
