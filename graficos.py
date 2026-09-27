"""
===============================================================================
SUZ Quant AI - Módulo de Visualização Gráfica e Econométrica (graficos.py)
Disciplina: Mercado Financeiro e de Capitais - PUC-SP
Empresa: Suzano S.A. | Ativo: SUZ (ADR na NYSE)
===============================================================================
Este módulo gera e salva em 'outputs/graficos/' os 11 gráficos estatísticos
fundamentais exigidos pelo projeto acadêmico, com padrão visual de relatório
institucional (300 DPI, títulos explicativos, eixos claros, legendas e fontes).
"""

import os
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

from regressao import ResultadoRegressao
from diagnosticos import ResultadoDiagnosticos

# Configurações globais de estilo
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

FONTE_RODAPE = "Fonte: Yahoo Finance / NYSE / CBOE / BACEN | SUZ Quant AI - PUC-SP"
COR_SUZ = "#1b4d3e"      # Verde florestal escuro (identidade Suzano)
COR_NYA = "#1f4e79"      # Azul marinho (mercado NYSE)
COR_RETA = "#c00000"     # Vermelho escuro para retas OLS
COR_CINZA = "#708090"


def _adicionar_rodape(fig: plt.Figure, texto: str = FONTE_RODAPE) -> None:
    """Adiciona rodapé padronizado em cada gráfico para auditoria acadêmica."""
    fig.text(0.99, 0.01, texto, ha="right", va="bottom", fontsize=8, color="#555555", style="italic")


def grafico_01_evolucao_suz(df_precos: pd.DataFrame, pasta_saida: str = "outputs/graficos") -> str:
    """1. Evolução histórica das cotações do ADR SUZ (em USD)."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(df_precos.index, df_precos["SUZ"], color=COR_SUZ, linewidth=1.5, label="Suzano ADR (SUZ - NYSE)")
    ax.set_title("Evolução Histórica do ADR da Suzano S.A. (SUZ)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Data", fontsize=10)
    ax.set_ylabel("Preço Ajustado (USD)", fontsize=10)
    ax.legend(loc="upper left", frameon=True)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _adicionar_rodape(fig)
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, "01_evolucao_suz.png")
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def grafico_02_evolucao_nya(df_precos: pd.DataFrame, pasta_saida: str = "outputs/graficos") -> str:
    """2. Evolução histórica do NYSE Composite Index (NYA)."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(df_precos.index, df_precos["NYA"], color=COR_NYA, linewidth=1.5, label="NYSE Composite (^NYA)")
    ax.set_title("Evolução Histórica do NYSE Composite Index (X1 = NYA)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Data", fontsize=10)
    ax.set_ylabel("Pontos do Índice", fontsize=10)
    ax.legend(loc="upper left", frameon=True)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _adicionar_rodape(fig)
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, "02_evolucao_nya.png")
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def grafico_03_retornos_diarios_suz(df_retornos: pd.DataFrame, pasta_saida: str = "outputs/graficos") -> str:
    """3. Série temporal dos retornos percentuais diários do SUZ."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ret_pct = df_retornos["SUZ"] * 100
    ax.plot(df_retornos.index, ret_pct, color=COR_SUZ, linewidth=0.6, alpha=0.85, label="Retorno Diário SUZ (%)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Retornos Percentuais Diários do ADR da Suzano (SUZ)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Data", fontsize=10)
    ax.set_ylabel("Variação Diária (%)", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _adicionar_rodape(fig)
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, "03_retornos_diarios_suz.png")
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def grafico_04_retornos_suz_x_nya(df_retornos: pd.DataFrame, pasta_saida: str = "outputs/graficos") -> str:
    """4. Comparação sobreposta de retornos diários SUZ × NYA."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(df_retornos.index, df_retornos["SUZ"] * 100, color=COR_SUZ, linewidth=0.6, alpha=0.7, label="SUZ (Suzano)")
    ax.plot(df_retornos.index, df_retornos["NYA"] * 100, color=COR_NYA, linewidth=0.6, alpha=0.7, label="NYA (Mercado NYSE)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Dinâmica Comparada dos Retornos Diários: SUZ vs NYA", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Data", fontsize=10)
    ax.set_ylabel("Retorno Diário (%)", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _adicionar_rodape(fig)
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, "04_retornos_suz_x_nya.png")
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def grafico_05_scatter_suz_nya_regressao(
    df_retornos: pd.DataFrame,
    reg_simples: ResultadoRegressao,
    pasta_saida: str = "outputs/graficos"
) -> str:
    """5. Scatter plot SUZ × NYA com a reta estimada da regressão simples OLS."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    x = df_retornos["NYA"] * 100
    y = df_retornos["SUZ"] * 100

    ax.scatter(x, y, color="#2b5b84", alpha=0.35, s=14, edgecolors="none", label="Observações Diárias")

    # Linha OLS
    alfa = reg_simples.tabela_coeficientes.loc["const", "coeficiente"] * 100
    beta = reg_simples.tabela_coeficientes.loc["NYA", "coeficiente"]
    x_grid = np.linspace(x.min(), x.max(), 200)
    y_grid = alfa + beta * x_grid

    r2_pct = reg_simples.r2 * 100
    label_reta = f"Reta OLS: R_SUZ = {alfa:.3f}% + {beta:.4f}·R_NYA (R² = {r2_pct:.2f}%)"
    ax.plot(x_grid, y_grid, color=COR_RETA, linewidth=2.0, label=label_reta)

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)

    ax.set_title("Regressão Linear Simples: Retorno SUZ vs Retorno NYA", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Retorno Diário do NYA (%)", fontsize=10)
    ax.set_ylabel("Retorno Diário da SUZ (%)", fontsize=10)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    _adicionar_rodape(fig)
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, "05_scatter_suz_nya_regressao.png")
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def grafico_06_matriz_correlacao(df_retornos: pd.DataFrame, pasta_saida: str = "outputs/graficos") -> str:
    """6. Matriz de Correlação de Pearson com anotações numéricas."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    corr = df_retornos.corr()

    mascara = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8, "label": "Correlação de Pearson (r)"},
        ax=ax
    )
    ax.set_title("Matriz de Correlação Linear entre os Retornos Diários", fontsize=12, fontweight="bold", pad=12)
    _adicionar_rodape(fig)
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, "06_matriz_correlacao.png")
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def grafico_07_residuos_modelo(reg_multi: ResultadoRegressao, pasta_saida: str = "outputs/graficos") -> str:
    """7. Resíduos vs Valores Ajustados (Fitted Values) do modelo."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ajustados = reg_multi.valores_ajustados * 100
    residuos = reg_multi.residuos * 100

    ax.scatter(ajustados, residuos, color="#3b6978", alpha=0.4, s=15, edgecolors="none")
    ax.axhline(0, color=COR_RETA, linestyle="--", linewidth=1.2)

    ax.set_title("Diagnóstico de Homocedasticidade: Resíduos vs Valores Ajustados", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Valores Estimados / Ajustados pelo Modelo (%)", fontsize=10)
    ax.set_ylabel("Resíduos / Erros de Estimativa (%)", fontsize=10)
    _adicionar_rodape(fig)
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, "07_residuos_modelo.png")
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def grafico_08_residuos_ao_longo_do_tempo(reg_multi: ResultadoRegressao, pasta_saida: str = "outputs/graficos") -> str:
    """8. Resíduos da regressão múltipla ao longo do tempo com bandas de 3 desvios."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    res = reg_multi.residuos * 100
    sigma3 = 3 * res.std()

    ax.plot(res.index, res, color=COR_CINZA, linewidth=0.6, alpha=0.8, label="Resíduo Diário (%)")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.axhline(sigma3, color=COR_RETA, linestyle="--", linewidth=1.0, label=f"+3σ ({sigma3:.2f}%)")
    ax.axhline(-sigma3, color=COR_RETA, linestyle="--", linewidth=1.0, label=f"-3σ (-{sigma3:.2f}%)")

    ax.set_title("Resíduos da Regressão Múltipla ao Longo do Tempo (Identificação de Outliers)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Data", fontsize=10)
    ax.set_ylabel("Resíduo (%)", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _adicionar_rodape(fig)
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, "08_residuos_ao_longo_do_tempo.png")
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def grafico_09_distribuicao_residuos(reg_multi: ResultadoRegressao, pasta_saida: str = "outputs/graficos") -> str:
    """9. Distribuição dos resíduos (Histograma + KDE vs Curva Normal Teórica)."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    res = reg_multi.residuos.dropna() * 100

    # Limita o eixo X aos percentis 0.5% e 99.5% para visualização clara sem compressão por outliers extremos
    limite = np.percentile(np.abs(res), 99.5)
    res_filtrado = res[np.abs(res) <= limite]

    sns.histplot(res_filtrado, bins=60, kde=True, stat="density", color="#4682b4", alpha=0.5, label="Distribuição Real (KDE)", ax=ax)

    # Curva Normal teórica com mesma média e desvio
    x_axis = np.linspace(-limite, limite, 300)
    pdf_normal = stats.norm.pdf(x_axis, loc=res.mean(), scale=res.std())
    ax.plot(x_axis, pdf_normal, color=COR_RETA, linestyle="--", linewidth=2.0, label="Normal Teórica Gaussiana")

    ax.set_title("Distribuição dos Resíduos vs Distribuição Normal Teórica", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Resíduo Diário (%)", fontsize=10)
    ax.set_ylabel("Densidade de Probabilidade", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    _adicionar_rodape(fig)
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, "09_distribuicao_residuos.png")
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def grafico_10_comparacao_r2_ajustado(
    resultados_reg: Dict[str, ResultadoRegressao],
    pasta_saida: str = "outputs/graficos"
) -> str:
    """10. Comparação horizontal do R² Ajustado entre as 8 especificações de retornos."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    nomes = list(resultados_reg.keys())
    r2_adjs = [reg.r2_ajustado * 100 for reg in resultados_reg.values()]

    barras = ax.barh(nomes, r2_adjs, color="#2e6b8e", edgecolor="#1a3c50", height=0.6)

    # Adiciona rótulos percentuais nas barras
    for b, val in zip(barras, r2_adjs):
        ax.text(val + 0.08, b.get_y() + b.get_height() / 2, f"{val:.2f}%", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(max(r2_adjs) * 1.35, 10))
    ax.set_title("Comparação do R² Ajustado entre os Modelos de Retornos Diários", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("R² Ajustado (%)", fontsize=10)
    ax.set_ylabel("Especificação do Modelo", fontsize=10)

    # Linha pontilhada no limiar de 50%
    ax.axvline(50, color=COR_RETA, linestyle=":", linewidth=1.5, label="Critério de Busca: R² > 50%")
    ax.legend(loc="lower right", frameon=True)
    _adicionar_rodape(fig)
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, "10_comparacao_r2_ajustado.png")
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def grafico_11_comparacao_retornos_cotacoes(
    reg_retornos: ResultadoRegressao,
    reg_cotacoes: ResultadoRegressao,
    dw_retornos: float,
    dw_cotacoes: float,
    pasta_saida: str = "outputs/graficos"
) -> str:
    """
    11. Comparação lado a lado entre Retornos Diários e Cotações em Nível,
    evidenciando didaticamente o risco de Regressão Espúria (R² alto com DW minúsculo).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    categorias = ["Retornos Diários\n(Metodologia Correta)", "Cotações em Nível\n(Risco Espúrio)"]

    # Gráfico 1: R² Ajustado
    r2_vals = [reg_retornos.r2_ajustado * 100, reg_cotacoes.r2_ajustado * 100]
    cores_r2 = ["#2b7a78", "#d9534f"]
    b1 = ax1.bar(categorias, r2_vals, color=cores_r2, width=0.5, edgecolor="black")
    for b, v in zip(b1, r2_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.2f}%", ha="center", fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.set_title("Poder Explicativo Aparente (R² Ajustado)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("R² Ajustado (%)", fontsize=10)

    # Gráfico 2: Estatística Durbin-Watson
    dw_vals = [dw_retornos, dw_cotacoes]
    cores_dw = ["#2b7a78", "#d9534f"]
    b2 = ax2.bar(categorias, dw_vals, color=cores_dw, width=0.5, edgecolor="black")
    for b, v in zip(b2, dw_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.3f}", ha="center", fontweight="bold")
    ax2.axhline(2.0, color="green", linestyle="--", linewidth=1.2, label="Valor Ideal sem Autocorrelação (d = 2,0)")
    ax2.set_ylim(0, 2.5)
    ax2.set_title("Estatística Durbin-Watson (Autocorrelação)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Estatística d", fontsize=10)
    ax2.legend(loc="upper right", frameon=True, fontsize=8)

    fig.suptitle("COMPARAÇÃO ECONOMÉTRICA: Retornos Diários vs Cotações em Nível", fontsize=13, fontweight="bold", y=0.98)
    _adicionar_rodape(fig, "Nota: Cotações geram R² alto artificialmente, com DW próximo de 0 (Regra de Granger-Newbold de Regressão Espúria).")
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, "11_comparacao_retornos_cotacoes.png")
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


def gerar_todos_os_graficos(
    df_precos: pd.DataFrame,
    df_retornos: pd.DataFrame,
    reg_simples: ResultadoRegressao,
    reg_multi: ResultadoRegressao,
    todos_modelos_ret: Dict[str, ResultadoRegressao],
    reg_cotacoes: ResultadoRegressao,
    dw_retornos: float,
    dw_cotacoes: float,
    pasta_saida: str = "outputs/graficos"
) -> Dict[str, str]:
    """Gera e salva a suíte completa com todos os 11 gráficos exigidos."""
    os.makedirs(pasta_saida, exist_ok=True)
    caminhos = {}

    caminhos["01_evolucao_suz"] = grafico_01_evolucao_suz(df_precos, pasta_saida)
    caminhos["02_evolucao_nya"] = grafico_02_evolucao_nya(df_precos, pasta_saida)
    caminhos["03_retornos_diarios_suz"] = grafico_03_retornos_diarios_suz(df_retornos, pasta_saida)
    caminhos["04_retornos_suz_x_nya"] = grafico_04_retornos_suz_x_nya(df_retornos, pasta_saida)
    caminhos["05_scatter_suz_nya_regressao"] = grafico_05_scatter_suz_nya_regressao(df_retornos, reg_simples, pasta_saida)
    caminhos["06_matriz_correlacao"] = grafico_06_matriz_correlacao(df_retornos, pasta_saida)
    caminhos["07_residuos_modelo"] = grafico_07_residuos_modelo(reg_multi, pasta_saida)
    caminhos["08_residuos_ao_longo_do_tempo"] = grafico_08_residuos_ao_longo_do_tempo(reg_multi, pasta_saida)
    caminhos["09_distribuicao_residuos"] = grafico_09_distribuicao_residuos(reg_multi, pasta_saida)
    caminhos["10_comparacao_r2_ajustado"] = grafico_10_comparacao_r2_ajustado(todos_modelos_ret, pasta_saida)
    caminhos["11_comparacao_retornos_cotacoes"] = grafico_11_comparacao_retornos_cotacoes(
        reg_multi, reg_cotacoes, dw_retornos, dw_cotacoes, pasta_saida
    )

    return caminhos


if __name__ == "__main__":
    from dados import carregar_ou_atualizar_dados
    from regressao import executar_regressao_simples, executar_regressao_multipla
    from modelos import testar_todos_os_modelos

    precos, retornos, _ = carregar_ou_atualizar_dados()
    regs_ret, diags_ret = testar_todos_os_modelos(retornos, "SUZ", tipo_dados="Retornos Diários")
    regs_cot, diags_cot = testar_todos_os_modelos(precos, "SUZ", tipo_dados="Cotações em Nível")

    simples = regs_ret["M1_NYA"]
    multi = regs_ret["M8_COMPLETO"]
    cot = regs_cot["M8_COMPLETO"]

    dw_r = diags_ret["M8_COMPLETO"].durbin_watson_stat
    dw_c = diags_cot["M8_COMPLETO"].durbin_watson_stat

    print("Gerando todos os 11 gráficos em 'outputs/graficos/'...")
    cams = gerar_todos_os_graficos(precos, retornos, simples, multi, regs_ret, cot, dw_r, dw_c)
    print(f"[OK] {len(cams)} gráficos gerados com sucesso!")
    for k, v in cams.items():
        print(f" - {v}")
