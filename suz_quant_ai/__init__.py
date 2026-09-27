"""
SUZ Quant AI - Pacote Quantitativo para Análise do ADR da Suzano S.A.
PUC-SP | Mercado Financeiro e de Capitais
"""

from dados import carregar_ou_atualizar_dados
from variaveis import CATALOGO_VARIAVEIS, VARIAVEIS_PADRAO
from regressao import executar_regressao_simples, executar_regressao_multipla
from diagnosticos import executar_diagnosticos_completos
from modelos import testar_todos_os_modelos, construir_tabela_resumo
from graficos import gerar_todos_os_graficos
from interpretacao import gerar_dossie_dezembro_2027

__version__ = "1.0.0"
__all__ = [
    "carregar_ou_atualizar_dados",
    "CATALOGO_VARIAVEIS",
    "VARIAVEIS_PADRAO",
    "executar_regressao_simples",
    "executar_regressao_multipla",
    "executar_diagnosticos_completos",
    "testar_todos_os_modelos",
    "construir_tabela_resumo",
    "gerar_todos_os_graficos",
    "gerar_dossie_dezembro_2027"
]
