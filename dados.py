"""
===============================================================================
SUZ Quant AI - Módulo de Aquisição, Limpeza e Alinhamento de Dados (dados.py)
Disciplina: Mercado Financeiro e de Capitais - PUC-SP
Empresa: Suzano S.A. | Ativo: SUZ (ADR na NYSE)
===============================================================================
Este módulo é responsável por:
1. Baixar cotações ajustadas de fontes públicas (Yahoo Finance via yfinance).
2. Manter cache local em 'data/' para rapidez, reproducibilidade e evitar chamadas excessivas.
3. Tratar datas sem negociação, fusos horários, valores nulos e duplicatas.
4. Calcular retornos diários percentuais individuais sem criar retornos artificiais.
5. Alinhar as séries pela data comum de negociação efetiva.
6. Salvar as bases finais limpas 'dados_alinhados_precos.csv' e 'dados_alinhados_retornos.csv'.
"""

import os
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
import yfinance as yf

from variaveis import CATALOGO_VARIAVEIS, VARIAVEIS_PADRAO


@dataclass
class EstatisticasAlinhamento:
    """Registra com total transparência o processo de limpeza e alinhamento de datas."""
    primeira_data: pd.Timestamp
    ultima_data: pd.Timestamp
    total_observacoes: int
    observacoes_removidas: int
    total_linhas_brutas_uniao: int
    detalhes_por_ativo: Dict[str, Dict[str, Any]]

    def resumo_didatico(self) -> str:
        """Gera texto explicativo para estudantes de Administração."""
        return (
            f"Período comum alinhado: de {self.primeira_data.strftime('%d/%m/%Y')} "
            f"até {self.ultima_data.strftime('%d/%m/%Y')}.\n"
            f"Total de pregões válidos (N): {self.total_observacoes:,} dias.\n"
            f"Observações descartadas por falta de negociação em ao menos um ativo: "
            f"{self.observacoes_removidas:,} dias."
        )


def _sanitizar_nome_arquivo(ticker: str) -> str:
    """Converte símbolos como '^NYA' ou 'BRL=X' em nomes válidos de arquivo no Windows."""
    return ticker.replace("^", "_").replace("=", "_").replace("/", "_")


def baixar_serie_yfinance(
    ticker: str,
    diretorio_cache: str = "data",
    usar_cache: bool = True
) -> pd.Series:
    """
    Baixa o histórico diário completo de preços ajustados para um ticker.
    Utiliza cache local em arquivo CSV para garantir auditabilidade e velocidade.
    
    Regra metodológica:
    - Preços ajustados ('Close' via auto_adjust=True) consideram dividendos,
      juros sobre capital próprio e desdobramentos/grupamentos de ações.
    - Se a API falhar e não houver cache, gera uma exceção explícita. NUNCA inventa dados.
    """
    os.makedirs(diretorio_cache, exist_ok=True)
    nome_limpo = _sanitizar_nome_arquivo(ticker)
    caminho_cache = os.path.join(diretorio_cache, f"cache_{nome_limpo}.csv")

    if usar_cache and os.path.exists(caminho_cache):
        try:
            df_cache = pd.read_csv(caminho_cache, parse_dates=["Date"], index_col="Date")
            serie = df_cache["Preco"].copy()
            serie.index = pd.to_datetime(serie.index).tz_localize(None).normalize()
            serie = serie.sort_index()
            serie = serie[~serie.index.duplicated(keep="last")]
            if len(serie) > 0:
                return serie
        except Exception as e:
            print(f"[Aviso] Falha ao ler cache de {ticker}: {e}. Tentando download...")

    # Download direto via yfinance
    print(f"[Download] Baixando dados para '{ticker}' via Yahoo Finance...")
    try:
        t = yf.Ticker(ticker)
        # auto_adjust=True garante que Close seja o preço ajustado por eventos societários
        hist = t.history(period="max", auto_adjust=True)
    except Exception as err:
        raise ConnectionError(
            f"Erro ao conectar ao Yahoo Finance para o ticker '{ticker}': {err}."
        ) from err

    if hist is None or hist.empty:
        raise ValueError(
            f"ERRO CRÍTICO: Nenhum dado retornado para o ticker '{ticker}'. "
            f"Verifique a conexão de internet ou se o ticker foi alterado pela bolsa."
        )

    # Limpeza e padronização do índice temporal
    # Converte fusos horários locais/UTC para datas ingênuas (naive) no início do dia
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    hist = hist.sort_index()
    # Remove eventuais duplicidades de datas
    hist = hist[~hist.index.duplicated(keep="last")]

    serie_preco = hist["Close"].astype(float)
    serie_preco.name = "Preco"
    # Remove valores nulos ou zeros anômalos
    serie_preco = serie_preco.dropna()
    serie_preco = serie_preco[serie_preco > 0]

    # Salva no cache local em disco
    df_para_salvar = pd.DataFrame({"Preco": serie_preco})
    df_para_salvar.to_csv(caminho_cache, index=True, index_label="Date")

    return serie_preco


def carregar_dados_brutos(
    variaveis: Optional[Dict[str, str]] = None,
    usar_cache: bool = True,
    diretorio_cache: str = "data"
) -> Dict[str, pd.Series]:
    """
    Obtém as séries de preços em nível para todas as variáveis especificadas.
    
    Parâmetros:
    - variaveis: dicionário mapeando nome da coluna (ex: 'SUZ', 'NYA') ao ticker de mercado.
    - usar_cache: se True, aproveita dados já baixados em 'data/'.
    """
    if variaveis is None:
        variaveis = {
            "SUZ": CATALOGO_VARIAVEIS[VARIAVEIS_PADRAO["Y"]].ticker,
            "NYA": CATALOGO_VARIAVEIS[VARIAVEIS_PADRAO["X1"]].ticker,
            "USDBRL": CATALOGO_VARIAVEIS[VARIAVEIS_PADRAO["X2"]].ticker,
            "WOOD": CATALOGO_VARIAVEIS[VARIAVEIS_PADRAO["X3"]].ticker,
            "US10Y": CATALOGO_VARIAVEIS[VARIAVEIS_PADRAO["X4"]].ticker,
        }

    series_dict = {}
    for nome_coluna, ticker in variaveis.items():
        serie = baixar_serie_yfinance(ticker, diretorio_cache=diretorio_cache, usar_cache=usar_cache)
        series_dict[nome_coluna] = serie

    return series_dict


def processar_e_alinhar_series(
    series_precos: Dict[str, pd.Series],
    data_inicio: Optional[str] = None,
    data_fim: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, EstatisticasAlinhamento]:
    """
    Processa os preços brutos e produz bases perfeitamente alinhadas de PREÇOS e RETORNOS.
    
    METODOLOGIA QUANTITATIVA RIGOROSA:
    1. Para cada ativo, o retorno diário é calculado sobre sua PRÓPRIA sequência temporal:
       R_t = (P_t - P_{t-1}) / P_{t-1}
       Isso evita que feriados locais provoquem retornos artificiais multi-dias.
    2. Em seguida, realiza-se a interseção (inner join) das datas em que TODOS os ativos
       possuem simultaneamente dados válidos.
    3. Nunca se faz preenchimento para frente (forward-fill) de retornos, pois criaria
       autocorrelação e volatilidade espúrias.
    """
    df_precos_bruto = pd.DataFrame(series_precos).sort_index()

    # Filtro opcional de intervalo de datas escolhido pelo usuário no Dashboard
    if data_inicio:
        df_precos_bruto = df_precos_bruto[df_precos_bruto.index >= pd.to_datetime(data_inicio)]
    if data_fim:
        df_precos_bruto = df_precos_bruto[df_precos_bruto.index <= pd.to_datetime(data_fim)]

    total_datas_possiveis = len(df_precos_bruto)

    # Detalhamento de cada série antes do alinhamento
    detalhes_ativos = {}
    for col in df_precos_bruto.columns:
        s = df_precos_bruto[col].dropna()
        detalhes_ativos[col] = {
            "linhas_validas": len(s),
            "primeira_data": s.index.min() if len(s) > 0 else None,
            "ultima_data": s.index.max() if len(s) > 0 else None,
        }

    # 1. Base alinhada de COTAÇÕES (em nível)
    df_precos_alinhados = df_precos_bruto.dropna().copy()

    # 2. Base de RETORNOS DIÁRIOS
    # Calcula retorno percentual ativo por ativo
    df_retornos_individuais = df_precos_bruto.pct_change()
    # Alinha os retornos por data comum
    df_retornos_alinhados = df_retornos_individuais.dropna().copy()

    if df_retornos_alinhados.empty:
        raise ValueError(
            "ERRO: Não há dias comuns de negociação suficientes entre as séries selecionadas "
            "no período especificado para calcular os retornos."
        )

    # Estatísticas de auditoria
    total_validas = len(df_retornos_alinhados)
    removidas = total_datas_possiveis - total_validas

    estatisticas = EstatisticasAlinhamento(
        primeira_data=df_retornos_alinhados.index.min(),
        ultima_data=df_retornos_alinhados.index.max(),
        total_observacoes=total_validas,
        observacoes_removidas=removidas,
        total_linhas_brutas_uniao=total_datas_possiveis,
        detalhes_por_ativo=detalhes_ativos
    )

    return df_precos_alinhados, df_retornos_alinhados, estatisticas


def salvar_bases_alinhadas(
    df_precos: pd.DataFrame,
    df_retornos: pd.DataFrame,
    diretorio_destino: str = "data"
) -> Tuple[str, str]:
    """Salva os arquivos finais de dados alinhados exigidos pelo item 3 do projeto."""
    os.makedirs(diretorio_destino, exist_ok=True)
    caminho_precos = os.path.join(diretorio_destino, "dados_alinhados_precos.csv")
    caminho_retornos = os.path.join(diretorio_destino, "dados_alinhados_retornos.csv")

    df_precos.to_csv(caminho_precos, index=True, index_label="Data", float_format="%.6f")
    df_retornos.to_csv(caminho_retornos, index=True, index_label="Data", float_format="%.6f")

    return caminho_precos, caminho_retornos


def carregar_ou_atualizar_dados(
    variaveis: Optional[Dict[str, str]] = None,
    usar_cache: bool = True,
    data_inicio: Optional[str] = None,
    data_fim: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, EstatisticasAlinhamento]:
    """
    Função principal que orquestra download, limpeza, alinhamento e salvamento.
    Retorna (df_precos_alinhados, df_retornos_alinhados, estatisticas).
    """
    series = carregar_dados_brutos(variaveis=variaveis, usar_cache=usar_cache)
    df_precos, df_retornos, stats = processar_e_alinhar_series(
        series_precos=series,
        data_inicio=data_inicio,
        data_fim=data_fim
    )
    salvar_bases_alinhadas(df_precos, df_retornos)
    return df_precos, df_retornos, stats


if __name__ == "__main__":
    print("=== Executando módulo dados.py ===")
    precos, retornos, estats = carregar_ou_atualizar_dados()
    print(estats.resumo_didatico())
    print("\nAmostra dos Retornos Diários:")
    print(retornos.head())
    print("\n[OK] Arquivos salvos em 'data/dados_alinhados_precos.csv' e 'data/dados_alinhados_retornos.csv'")
