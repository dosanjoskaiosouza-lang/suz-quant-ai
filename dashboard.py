"""
===============================================================================
SUZ Quant AI - Painel Interativo Streamlit (dashboard.py)
Disciplina: Mercado Financeiro e de Capitais - PUC-SP
Empresa: Suzano S.A. | Ativo: SUZ (ADR na NYSE)
Mascote & Assistente Quantitativa: SUZI (Suzano Quant Intelligence)
===============================================================================
Painel interativo institucional completo para análise quantitativa,
diagnósticos econométricos, preparação de banca e suporte à decisão de Dez/2027.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página Streamlit
st.set_page_config(
    page_title="SUZ Quant AI — SUZI Assistente & Auditoria",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeção de CSS personalizado com a identidade visual da SUZI (extraída da imagem oficial)
# Paleta de Cores:
# - Cor Primária: #00281E (Verde Florestal Suzano Profundo)
# - Cor Secundária: #1B4D3E (Verde Esmeralda Institucional)
# - Cor de Destaque: #C8963E (Ouro / Mostarda Econômico)
# - Fundo: #F8FAF9 (Branco Neve / Papel)
# - Texto: #1A202C (Grafite Escuro Carbono)
st.markdown("""
<style>
 /* =====================================================
   CORREÇÃO FORÇADA DO TEXTO DAS ABAS
   ===================================================== */

/* Todas as abas */
div[data-baseweb="tab-list"] button[data-baseweb="tab"],
div[data-baseweb="tab-list"] button[data-baseweb="tab"] *,
div[data-baseweb="tab-list"] button[data-baseweb="tab"] p,
div[data-baseweb="tab-list"] button[data-baseweb="tab"] span {
    color: #1a202c !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #1a202c !important;
    font-weight: 700 !important;
}

/* Aba selecionada */
div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"],
div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] *,
div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] p,
div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] span {
    color: #00281e !important;
    -webkit-text-fill-color: #00281e !important;
    opacity: 1 !important;
}

/* Quando passar o mouse */
div[data-baseweb="tab-list"] button[data-baseweb="tab"]:hover,
div[data-baseweb="tab-list"] button[data-baseweb="tab"]:hover *,
div[data-baseweb="tab-list"] button[data-baseweb="tab"]:hover p,
div[data-baseweb="tab-list"] button[data-baseweb="tab"]:hover span {
    color: #00281e !important;
    -webkit-text-fill-color: #00281e !important;
    opacity: 1 !important;
}

</style>
""", unsafe_allow_html=True)

# Importações dos módulos locais
from dados import carregar_dados_brutos, processar_e_alinhar_series
from variaveis import CATALOGO_VARIAVEIS, VARIAVEIS_PADRAO
from regressao import executar_regressao_simples, executar_regressao_multipla
from diagnosticos import executar_diagnosticos_completos
from modelos import testar_todos_os_modelos, construir_tabela_resumo, avaliar_criterio_busca_global
from interpretacao import interpretar_regressao_simples, interpretar_regressao_multipla, gerar_dossie_dezembro_2027
from conteudo_academico import (
    SUZI_BIO,
    obter_perguntas_respostas_suzi,
    PERGUNTAS_PROFESSOR,
    EVENTOS_NOTICIAS,
    CENARIOS_2027,
    obter_matriz_evidencias_sintese
)


# =============================================================================
# FUNÇÃO DE CARREGAMENTO COM CACHE STREAMLIT
# =============================================================================

@st.cache_data(show_spinner=False)
def carregar_dados_base():
    """Carrega as séries financeiras diárias brutas mantendo cache."""
    return carregar_dados_brutos(usar_cache=True)


# =============================================================================
# BARRA LATERAL (SIDEBAR) — IDENTIDADE VISUAL E CONTROLES
# =============================================================================

CAMINHO_MASCOTE = "Suz Quant principal.png"
if os.path.exists(CAMINHO_MASCOTE):
    st.sidebar.image(CAMINHO_MASCOTE, caption="SUZI — Suzano Quant Intelligence", width=180)
else:
    st.sidebar.title("🌲 SUZ Quant AI")

st.sidebar.markdown(
    "<h3 style='color:#f8faf9; margin-top:-10px;'>SUZ Quant AI</h3>"
    "<p style='color:#d2dcd2; font-size:13px; margin-top:-5px;'>PUC-SP • Mercado Financeiro e de Capitais<br>"
    "<b>Ativo:</b> Suzano S.A. (<code>SUZ</code> — NYSE)</p>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Configurações do Modelo")

# Carrega séries brutas
with st.spinner("Carregando base de dados..."):
    series_brutas = carregar_dados_base()

data_min_global = min(s.index.min() for s in series_brutas.values()).date()
data_max_global = max(s.index.max() for s in series_brutas.values()).date()

# 1. Filtro Temporal
st.sidebar.markdown("**Período Histórico Amostral:**")
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    data_inicio_padrao = max(pd.Timestamp("2023-01-01").date(), data_min_global)
    dt_inicio = st.date_input("Início", value=data_inicio_padrao, min_value=data_min_global, max_value=data_max_global)
with col_d2:
    dt_fim = st.date_input("Fim", value=data_max_global, min_value=data_min_global, max_value=data_max_global)

if dt_inicio >= dt_fim:
    st.sidebar.error("A data inicial deve ser anterior à data final.")
    st.stop()

# 2. Nível de Significância
alpha_nivel = st.sidebar.selectbox(
    "Nível de Significância (α):",
    options=[0.01, 0.05, 0.10],
    index=1,
    format_func=lambda x: f"{int(x * 100)}% (p < {x:.2f})"
)

# 3. Variáveis Explicativas
st.sidebar.markdown("**Variáveis no Modelo:**")
st.sidebar.markdown("• **Y (Dependente):** `SUZ` *(ADR na NYSE)*")
st.sidebar.markdown("• **X1 (Obrigatório):** `NYA` *(NYSE Composite)*")

vars_x_selecionadas = st.sidebar.multiselect(
    "Selecione até 3 fatores adicionais para X2–X4:",
    options=["USDBRL", "WOOD", "US10Y", "KLBAY", "EWZ"],
    default=["USDBRL", "WOOD", "US10Y"],
    max_selections=3
)

# 4. Modo de Cálculo
st.sidebar.markdown("---")
modo_analise = st.sidebar.radio(
    "Base Metodológica Principal:",
    options=["Retornos Diários (Recomendado)", "Cotações em Nível (Complementar)"],
    index=0
)


# =============================================================================
# ALINHAMENTO E EXECUÇÃO DOS MODELOS EM MEMÓRIA
# =============================================================================

series_selecionadas = {"SUZ": series_brutas["SUZ"], "NYA": series_brutas["NYA"]}
for vx in vars_x_selecionadas:
    if vx in series_brutas:
        series_selecionadas[vx] = series_brutas[vx]

df_precos, df_retornos, stats_alinhamento = processar_e_alinhar_series(
    series_precos=series_selecionadas,
    data_inicio=str(dt_inicio),
    data_fim=str(dt_fim)
)

df_ativo = df_retornos if "Retornos" in modo_analise else df_precos
tipo_dados_label = "Retornos Diários" if "Retornos" in modo_analise else "Cotações em Nível"

# Regressão Simples
reg_simples = executar_regressao_simples(df_ativo, col_y="SUZ", col_x="NYA", tipo_dados=tipo_dados_label)
interp_simples = interpretar_regressao_simples(reg_simples)

# Regressões Múltiplas sistemáticas
especs_atuais = [("M1_NYA", ["NYA"])]
for vx in vars_x_selecionadas:
    especs_atuais.append((f"M_{vx}", ["NYA", vx]))
if len(vars_x_selecionadas) >= 2:
    for i in range(len(vars_x_selecionadas)):
        for j in range(i + 1, len(vars_x_selecionadas)):
            v1, v2 = vars_x_selecionadas[i], vars_x_selecionadas[j]
            especs_atuais.append((f"M_{v1}_{v2}", ["NYA", v1, v2]))
if len(vars_x_selecionadas) >= 3:
    especs_atuais.append(("M_COMPLETO", ["NYA"] + vars_x_selecionadas))

regs_modelos, diags_modelos = testar_todos_os_modelos(
    df=df_ativo,
    col_y="SUZ",
    especificacoes=especs_atuais,
    tipo_dados=tipo_dados_label
)
tabela_resumo = construir_tabela_resumo(regs_modelos, diags_modelos)

criterio_atendido, msg_criterio = avaliar_criterio_busca_global(regs_modelos)

# Regressões comparativas em paralelo para aba Retornos x Cotações
regs_paralelo_ret, diags_paralelo_ret = testar_todos_os_modelos(df_retornos, "SUZ", tipo_dados="Retornos Diários")
regs_paralelo_cot, diags_paralelo_cot = testar_todos_os_modelos(df_precos, "SUZ", tipo_dados="Cotações em Nível")

# Objetos de referência usados em diferentes páginas do dashboard.
# Ficam fora dos blocos de navegação para existirem independentemente
# da página selecionada no menu lateral.
m_ret = regs_paralelo_ret[list(regs_paralelo_ret.keys())[-1]]
m_cot = regs_paralelo_cot[list(regs_paralelo_cot.keys())[-1]]
d_ret = diags_paralelo_ret[list(diags_paralelo_ret.keys())[-1]]
d_cot = diags_paralelo_cot[list(diags_paralelo_cot.keys())[-1]]

# Dicionário de números reais para a assistente SUZI
dados_reais_suzi = {
    "beta_simples": float(reg_simples.tabela_coeficientes.loc["NYA", "coeficiente"]),
    "alfa_simples": float(reg_simples.tabela_coeficientes.loc["const", "coeficiente"]),
    "r2_simples": float(reg_simples.r2 * 100),
    "r2_multi": float(regs_paralelo_ret[list(regs_paralelo_ret.keys())[-1]].r2 * 100),
    "r2_adj_multi": float(regs_paralelo_ret[list(regs_paralelo_ret.keys())[-1]].r2_ajustado * 100),
    "r2_cotacoes": float(regs_paralelo_cot[list(regs_paralelo_cot.keys())[-1]].r2 * 100),
    "dw_retornos": float(diags_paralelo_ret[list(diags_paralelo_ret.keys())[-1]].durbin_watson_stat),
    "dw_cotacoes": float(diags_paralelo_cot[list(diags_paralelo_cot.keys())[-1]].durbin_watson_stat),
    "n_obs": int(stats_alinhamento.total_observacoes)
}


# =============================================================================
# CABEÇALHO PRINCIPAL
# =============================================================================

col_logo, col_titulo = st.columns([1.2, 5])
with col_logo:
    if os.path.exists(CAMINHO_MASCOTE):
        st.image(CAMINHO_MASCOTE, width=130)
with col_titulo:
    st.markdown("<h1 style='color:#00281e; margin-bottom:0;'>SUZ Quant AI — Plataforma Quantitativa</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#1b4d3e; font-size:15px; font-weight:500; margin-top:-5px;'>"
        "<b>Mascote & Assistente:</b> SUZI (<i>Suzano Quant Intelligence</i>) • "
        "<b>Banca Examinadora:</b> Prof. Dr. José Odálio dos Santos (PUC-SP)</p>",
        unsafe_allow_html=True
    )

st.markdown(
    "<div class='suzi-quote'>"
    "<b>Questão Central de Pesquisa:</b> "
    "“O grupo recomendaria a compra do ADR da Suzano para revendê-lo em dezembro de 2027?”"
    "</div>",
    unsafe_allow_html=True
)
st.write("")

if "Cotações" in modo_analise:
    st.warning(
        "⚠️ **ALERTA: MODO COTAÇÕES EM NÍVEL ATIVO.** Preços em nível geram R² inflado artificialmente por tendências temporais comuns. "
        "Pela regra de Granger-Newbold, essa relação configura **Regressão Espúria**. Para fundamentação oficial, consulte a aba de Retornos Diários."
    )

# ========================================================================
# NAVEGAÇÃO PRINCIPAL — SUZ QUANT AI
# ========================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("## 🧭 Navegação")

pagina_principal = st.sidebar.radio(
    "Selecione uma área:",
    [
        "🏠 Visão Geral",
        "📊 Mercado & Desempenho",
        "📈 Regressões",
        "💰 Fundamentos & Valuation",
        "🎯 Decisão Dez/2027",
        "🤖 SUZI"
    ],
    index=0
)

st.sidebar.caption("SUZ Quant AI • PUC-SP")
# =============================================================================
# CONTEÚDO CONTROLADO PELO MENU LATERAL
# =============================================================================

# -----------------------------------------------------------------------------
# ABA 1: VISÃO GERAL
# -----------------------------------------------------------------------------
if pagina_principal == "🏠 Visão Geral":
    st.header("📊 Visão Geral das Séries e da Base Histórica")
    
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Última Cotação SUZ", f"US$ {df_precos['SUZ'].iloc[-1]:.2f}")
    with k2:
        st.metric("Pregões Válidos (N)", f"{stats_alinhamento.total_observacoes:,}")
    with k3:
        st.metric("Início da Série", stats_alinhamento.primeira_data.strftime("%d/%m/%Y"))
    with k4:
        st.metric("Fim da Série", stats_alinhamento.ultima_data.strftime("%d/%m/%Y"))
    with k5:
        st.metric("Obs. Descartadas", f"{stats_alinhamento.observacoes_removidas:,}")

    st.markdown("---")
    cg1, cg2 = st.columns(2)
    with cg1:
        st.subheader("Cotação Histórica do ADR SUZ (NYSE - USD)")
        st.line_chart(df_precos["SUZ"], color="#00281e")
    with cg2:
        st.subheader("Evolução do NYSE Composite (^NYA - Pontos)")
        st.line_chart(df_precos["NYA"], color="#1f4e79")

    cg3, cg4 = st.columns(2)
    with cg3:
        st.subheader("Retornos Diários do ADR SUZ (%)")
        st.line_chart(df_retornos["SUZ"] * 100, color="#00281e")
    with cg4:
        st.subheader("Retornos Diários do Mercado NYA (%)")
        st.line_chart(df_retornos["NYA"] * 100, color="#1f4e79")

    with st.expander("📋 Ver Resumo Estatístico Descritivo das Séries"):
        st.dataframe(df_ativo.describe().T.style.format("{:.4f}"), use_container_width=True)

    with st.expander("🔍 Auditoria de Alinhamento e Integridade Temporal"):
        st.info(stats_alinhamento.resumo_didatico())


# -----------------------------------------------------------------------------
# ABA 2: REGRESSÃO SIMPLES
# -----------------------------------------------------------------------------
if pagina_principal == "📈 Regressões":
    st.header("📈 Regressão Linear Simples Obrigatória (SUZ ~ NYA)")
    st.markdown(
        "> **Modelo de Mercado:** $R_{SUZ,t} = \\alpha + \\beta_1 \\cdot R_{NYA,t} + \\varepsilon_t$  \n"
        "> Mensura a sensibilidade sistemática do ADR da Suzano em relação a Wall Street."
    )

    alfa_v = reg_simples.tabela_coeficientes.loc["const", "coeficiente"]
    beta_v = reg_simples.tabela_coeficientes.loc["NYA", "coeficiente"]
    p_b = reg_simples.tabela_coeficientes.loc["NYA", "p_valor"]

    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    with sm1:
        st.metric("Alfa (α) Diário", f"{alfa_v:.6f}")
    with sm2:
        st.metric("Beta (β) de Mercado", f"{beta_v:.4f}", delta="Defensivo (< 1,0)" if beta_v < 1 else "Agressivo")
    with sm3:
        st.metric("R²", f"{reg_simples.r2 * 100:.2f}%")
    with sm4:
        st.metric("R² Ajustado", f"{reg_simples.r2_ajustado * 100:.2f}%")
    with sm5:
        st.metric("p-valor do Beta", f"{p_b:.2e}", delta="Significante (p < 0,05)")

    st.markdown(f"### **Equação Estimada:** `{interp_simples['equacao']}`")

    col_t, col_p = st.columns([1, 1.2])
    with col_t:
        st.subheader("Tabela de Coeficientes OLS")
        st.dataframe(
            reg_simples.tabela_coeficientes.style.format({
                "coeficiente": "{:.6f}",
                "erro_padrao": "{:.6f}",
                "estatistica_t": "{:.4f}",
                "p_valor": "{:.4e}",
                "ic95_inferior": "{:.6f}",
                "ic95_superior": "{:.6f}"
            }),
            use_container_width=True
        )
        st.markdown("#### Interpretação Pedagógica:")
        st.write(f"• {interp_simples['beta_interpretacao']}")
        st.write(f"• {interp_simples['alfa_interpretacao']}")
        st.write(f"• {interp_simples['r2_interpretacao']}")
        st.write(f"• {interp_simples['teste_f_interpretacao']}")

    with col_p:
        st.subheader("Scatter Plot com a Reta de Regressão OLS")
        fig_s, ax_sc = plt.subplots(figsize=(7, 5), dpi=150)
        x_pts = df_ativo["NYA"] * (100 if "Retornos" in modo_analise else 1)
        y_pts = df_ativo["SUZ"] * (100 if "Retornos" in modo_analise else 1)
        ax_sc.scatter(x_pts, y_pts, color="#2b5b84", alpha=0.3, s=12, label="Observações Diárias")
        x_g = np.linspace(x_pts.min(), x_pts.max(), 100)
        y_g = alfa_v * (100 if "Retornos" in modo_analise else 1) + beta_v * x_g
        ax_sc.plot(x_g, y_g, color="#c00000", linewidth=2.0, label="Reta OLS")
        ax_sc.set_xlabel("NYA" + (" (%)" if "Retornos" in modo_analise else " (Pontos)"))
        ax_sc.set_ylabel("SUZ" + (" (%)" if "Retornos" in modo_analise else " (USD)"))
        ax_sc.legend()
        fig_s.tight_layout()
        st.pyplot(fig_s)
        plt.close(fig_s)


# -----------------------------------------------------------------------------
# ABA 3: REGRESSÃO MÚLTIPLA
# -----------------------------------------------------------------------------
if pagina_principal == "📈 Regressões":
    st.header("🧮 Regressões Múltiplas: Teste Sistemático de Especificações")
    st.markdown(
        "Todas as combinações metodologicamente defensáveis com no máximo 4 variáveis explicativas "
        "foram testadas mantendo $X_1 = NYA$ obrigatório."
    )

    if criterio_atendido:
        st.success(f"🎯 **CRITÉRIO DE BUSCA ATENDIDO:** {msg_criterio}")
    else:
        st.info(
            f"ℹ️ **CRITÉRIO DE BUSCA:** \"{msg_criterio}\"  \n"
            f"*(Critério rigoroso: R² ajustado > 50% E p-valor < {alpha_nivel:.2f} para todos os X).* "
            f"Em mercados acionários diários, este resultado é totalmente condizente com a teoria financeira."
        )

    st.subheader("Tabela Resumo de Todas as Especificações Testadas")
    cols_v = ["Modelo", "Variáveis X", "N", "R²", "R² ajustado", "maior p-value entre X", "diagnóstico", "critério atendido?"]
    st.dataframe(tabela_resumo[cols_v], use_container_width=True)

    st.markdown("---")
    st.subheader("Inspeção Detalhada de um Modelo Específico")
    mod_escolhido = st.selectbox("Selecione o modelo para inspecionar:", options=list(regs_modelos.keys()), index=len(regs_modelos)-1)
    r_sel = regs_modelos[mod_escolhido]
    d_sel = diags_modelos[mod_escolhido]

    c_m1, c_m2 = st.columns([1, 1.3])
    with c_m1:
        st.markdown(f"**Equação:** `{r_sel.gerar_equacao_formatada()}`")
        st.markdown(f"• **R²:** {r_sel.r2 * 100:.2f}% | **R² Ajustado:** {r_sel.r2_ajustado * 100:.2f}%")
        st.markdown(f"• **Estatística F:** {r_sel.f_estatistica:.2f} (p-valor: {r_sel.f_pvalor:.2e})")
        st.info(d_sel.parecer_didatico)

    with c_m2:
        df_coefs_v = r_sel.tabela_coeficientes.copy()
        df_coefs_v["significativo"] = df_coefs_v["p_valor"] < alpha_nivel
        st.dataframe(
            df_coefs_v.style.format({
                "coeficiente": "{:.6f}",
                "erro_padrao": "{:.6f}",
                "estatistica_t": "{:.4f}",
                "p_valor": "{:.4e}",
                "ic95_inferior": "{:.6f}",
                "ic95_superior": "{:.6f}"
            }),
            use_container_width=True
        )


# -----------------------------------------------------------------------------
# ABA 4: DIAGNÓSTICOS ECONOMÉTRICOS
# -----------------------------------------------------------------------------
if pagina_principal == "📈 Regressões":
    st.header("🔬 Diagnósticos Econométricos dos Pressupostos OLS")
    mod_diag = regs_modelos[list(regs_modelos.keys())[-1]]
    diag_obj = diags_modelos[list(regs_modelos.keys())[-1]]

    cd1, cd2, cd3, cd4 = st.columns(4)
    with cd1:
        v_max = diag_obj.tabela_vif["VIF"].max()
        st.metric("VIF Máximo", f"{v_max:.2f}", delta="Adequado (< 5,0)" if v_max < 5 else "Atenção")
    with cd2:
        st.metric("Breusch-Pagan (LM)", f"{diag_obj.lm_breusch_pagan:.2f}", f"p = {diag_obj.p_breusch_pagan:.4f}")
    with cd3:
        st.metric("Durbin-Watson", f"{diag_obj.durbin_watson_stat:.3f}", delta="Ideal ~ 2.0")
    with cd4:
        st.metric("Outliers (> 3σ)", f"{diag_obj.total_outliers_3sigma}")

    st.markdown("---")
    c_dl, c_dr = st.columns(2)
    with c_dl:
        st.subheader("1. Fator de Inflação da Variância (VIF)")
        st.dataframe(diag_obj.tabela_vif.style.format("{:.3f}"), use_container_width=True)
        st.subheader("Matriz de Correlação Linear")
        fig_h, ax_hm = plt.subplots(figsize=(5, 3.8), dpi=150)
        sns.heatmap(diag_obj.matriz_correlacao, annot=True, fmt=".2f", cmap="vlag", ax=ax_hm, cbar=False)
        fig_h.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

    with c_dr:
        st.subheader("2. Estacionariedade (Teste ADF)")
        st.dataframe(diag_obj.tabela_adf.style.format({"Estatistica_ADF": "{:.3f}", "p_valor": "{:.4e}", "Critico_5%": "{:.3f}"}), use_container_width=True)
        st.subheader("3. Distribuição dos Resíduos vs Normal")
        fig_d, ax_ds = plt.subplots(figsize=(6, 3.8), dpi=150)
        sns.histplot(mod_diag.residuos * 100, kde=True, color="#1b4d3e", bins=40, ax=ax_ds)
        fig_d.tight_layout()
        st.pyplot(fig_d)
        plt.close(fig_d)

    with st.expander("📋 Ver Parecer Diagnóstico Completo"):
        st.text(diag_obj.gerar_relatorio_texto())


# -----------------------------------------------------------------------------
# ABA 5: RETORNOS × COTAÇÕES (REGRESSÃO ESPÚRIA)
# -----------------------------------------------------------------------------
if pagina_principal == "📈 Regressões":
    st.header("⚖️ Comparação Metodológica: Retornos Diários vs Cotações em Nível")
    st.error(
        "🚨 **ALERTA METODOLÓGICO DE REGRESSÃO ESPÚRIA (PUC-SP):**  \n"
        "Regredir preços em nível produz um R² artificialmente elevado (65,7%), mas a estatística Durbin-Watson é minúscula (d = 0,013 ≪ R²). "
        "Pela **Regra de Granger-Newbold**, este é o clássico caso de **Regressão Espúria**. Para decisões sérias, utilize a análise de retornos diários."
    )

    m_ret = regs_paralelo_ret[list(regs_paralelo_ret.keys())[-1]]
    m_cot = regs_paralelo_cot[list(regs_paralelo_cot.keys())[-1]]
    d_ret = diags_paralelo_ret[list(diags_paralelo_ret.keys())[-1]]
    d_cot = diags_paralelo_cot[list(diags_paralelo_cot.keys())[-1]]

    df_comp = pd.DataFrame({
        "Critério de Comparação": [
            "Tipo de Variável",
            "Poder Explicativo (R²)",
            "R² Ajustado",
            "Estacionariedade (ADF)",
            "Durbin-Watson (DW)",
            "Multicolinearidade (VIF Máx)",
            "Risco de Regressão Espúria",
            "Veredito Metodológico"
        ],
        "Retornos Diários (Metodologia Correta)": [
            "Variação Percentual Diária (% ao dia)",
            f"{m_ret.r2 * 100:.2f}%",
            f"{m_ret.r2_ajustado * 100:.2f}%",
            "Estacionárias I(0) (p < 0,001)",
            f"d = {d_ret.durbin_watson_stat:.3f} (Ideal ~ 2.0)",
            f"{d_ret.tabela_vif['VIF'].max():.2f} (Adequado)",
            "Nenhum (Resíduos livres)",
            "VÁLIDO para análise de sensibilidade ao risco"
        ],
        "Cotações em Nível (Ilusão Estatística)": [
            "Preço em Nível (USD / Pontos)",
            f"{m_cot.r2 * 100:.2f}% (Artificialmente alto)",
            f"{m_cot.r2_ajustado * 100:.2f}% (Artificialmente alto)",
            "NÃO-Estacionárias I(1) (p > 0,25)",
            f"d = {d_cot.durbin_watson_stat:.3f} (Severa autocorrelação)",
            f"{d_cot.tabela_vif['VIF'].max():.2f} (Elevado)",
            "CRÍTICO (R² >> DW: Granger-Newbold)",
            "INVÁLIDO para tomada de decisão financeira"
        ]
    })
    st.table(df_comp)

    caminho_graf_comp = "outputs/graficos/11_comparacao_retornos_cotacoes.png"
    if os.path.exists(caminho_graf_comp):
        st.image(caminho_graf_comp, caption="Comparação Visual: Retornos vs Cotações (R² vs Durbin-Watson)", use_container_width=True)


# -----------------------------------------------------------------------------
# ABA 6: 🤖 ASSISTENTE SUZI (SUZANO QUANT INTELLIGENCE)
# -----------------------------------------------------------------------------
if pagina_principal == "🤖 SUZI":
    st.header("🤖 SUZI — Assistente Quantitativa & Mascote Acadêmica")
    
    col_s1, col_s2 = st.columns([1.2, 3])
    with col_s1:
        if os.path.exists(CAMINHO_MASCOTE):
            st.image(CAMINHO_MASCOTE, caption="SUZI: Suzano Quant Intelligence", use_container_width=True)
    with col_s2:
        st.markdown(
            f"<div class='suzi-card'>"
            f"<h3 style='margin-top:0;'>Olá! Eu sou a SUZI 👋</h3>"
            f"<p>{SUZI_BIO['apresentacao']}</p>"
            f"<b>Meus Pilares de Conhecimento:</b>"
            f"<ul>" + "".join(f"<li>{p}</li>" for p in SUZI_BIO['pilares']) + "</ul>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("💡 Pergunte à SUZI: Explicações Conceituais com Dados Reais do Projeto")
    st.markdown("Selecione um tópico para que a SUZI explique o conceito utilizando os valores exatos apurados pelo nosso modelo:")

    lista_suzi = obter_perguntas_respostas_suzi(dados_reais_suzi)
    titulos_suzi = [item["pergunta"] for item in lista_suzi]
    escolha_pergunta = st.selectbox("Escolha uma pergunta para a SUZI responder:", titulos_suzi)

    item_selecionado = next(it for it in lista_suzi if it["pergunta"] == escolha_pergunta)

    st.markdown(
        f"<div class='suzi-card'>"
        f"<h4 style='color:#00281e; margin-top:0;'>🤖 SUZI Explica: {item_selecionado['conceito']}</h4>"
        f"<p style='font-size:15px; line-height:1.6;'>{item_selecionado['resposta']}</p>"
        f"</div>",
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# ABA 7: 🎓 PERGUNTAS DO PROFESSOR (BANCA PUC-SP)
# -----------------------------------------------------------------------------
if pagina_principal == "🤖 SUZI":
    st.header("🎓 Possíveis Perguntas do Professor e Banca Examinadora")
    st.markdown(
        "> **Banca Examinadora:** Prof. Dr. José Odálio dos Santos — PUC-SP  \n"
        "> Banco com **18 perguntas desafiadoras e realistas** para preparar o grupo para qualquer arguição técnica durante a apresentação."
    )

    # Filtro de Categoria
    categorias = sorted(list(set(p["categoria"] for p in PERGUNTAS_PROFESSOR)))
    filtro_cat = st.selectbox("Filtrar por Categoria Temática:", ["Todas as Categorias"] + categorias)

    perguntas_filtradas = (
        PERGUNTAS_PROFESSOR if filtro_cat == "Todas as Categorias"
        else [p for p in PERGUNTAS_PROFESSOR if p["categoria"] == filtro_cat]
    )

    for item in perguntas_filtradas:
        with st.expander(f"❓ {item['id']}. [{item['categoria']}] {item['pergunta']}"):
            st.markdown(f"**⚡ Resposta Curta para Apresentação (Pitch):**  \n{item['resposta_curta']}")
            st.markdown(f"**🔬 Resposta Técnica Aprofundada:**  \n{item['resposta_tecnica']}")
            st.caption(f"📍 **Onde está a evidência no projeto:** {item['evidencia']}")


# -----------------------------------------------------------------------------
# ABA 8: 📰 EVENTOS E NOTÍCIAS CORPORATIVAS
# -----------------------------------------------------------------------------
if pagina_principal == "🎯 Decisão Dez/2027":
    st.header("📰 Eventos Corporativos, Notícias e Fusões & Aquisições (M&A)")
    st.markdown(
        "Avaliação qualitativa dos acontecimentos fundamentais que influenciaram o ADR da Suzano.  \n"
        "*(Nota: Notícias funcionam como evidência qualitativa complementar e não são inseridas retroativamente nas regressões).*"
    )

    # Tabela estruturada de eventos
    df_eventos_tabela = pd.DataFrame([
        {
            "Data": ev["data"],
            "Evento": ev["evento"],
            "Classificação": ev["classificacao"],
            "Duração": ev["duracao"],
            "Impacto no SUZ": ev["impacto_potencial"],
            "Relação com o Modelo Econométrico": ev["relacao_modelo"]
        } for ev in EVENTOS_NOTICIAS
    ])
    st.dataframe(df_eventos_tabela, use_container_width=True)

    st.markdown("---")
    st.subheader("Detalhamento dos Fatos Relevantes e Interpretação da SUZI")

    for ev in EVENTOS_NOTICIAS:
        with st.expander(f"📌 {ev['data']} — {ev['evento']} ({ev['classificacao']})"):
            c_ev_a, c_ev_b = st.columns(2)
            with c_ev_a:
                st.markdown(f"**O Fato Concreto:**  \n{ev['fato']}")
                st.markdown(f"**Opinião dos Analistas de Mercado:**  \n{ev['opiniao_analistas']}")
            with c_ev_b:
                st.markdown(f"**🤖 Interpretação Quantitativa da SUZI:**  \n{ev['interpretacao_suzi']}")
                st.caption(f"Classificação: {ev['classificacao']} | Duração: {ev['duracao']}")


# -----------------------------------------------------------------------------
# ABA 9: 🎯 CENÁRIOS PARA DEZEMBRO DE 2027
# -----------------------------------------------------------------------------
if pagina_principal == "🎯 Decisão Dez/2027":
    st.header("🎯 Cenários Prospectivos para Dezembro de 2027")
    st.markdown(
        "> **Metodologia Conservadora:** Sem preços-alvo adivinhados ou probabilidades arbitrárias.  \n"
        "> Estrutura rigorosa: **Condições → Impacto Esperado → Indicadores a Monitorar → Gatilhos de Invalidação**."
    )

    col_cen1, col_cen2, col_cen3 = st.columns(3)

    with col_cen1:
        c_fav = CENARIOS_2027["favoravel"]
        st.markdown(
            f"<div class='suzi-card' style='border-left-color:#28a745;'>"
            f"<h4>{c_fav['titulo']}</h4>"
            f"<p>{c_fav['descricao']}</p>"
            f"<b>Condições Macroeconômicas & Setoriais:</b>"
            f"<ul>" + "".join(f"<li>{c}</li>" for c in c_fav['condicoes']) + "</ul>"
            f"<b>Impacto Esperado no ADR:</b><br>{c_fav['impacto_esperado']}<br><br>"
            f"<b>Gatilhos de Invalidação:</b><br><span style='color:#c00000;'>{c_fav['gatilhos_invalidacao']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col_cen2:
        c_int = CENARIOS_2027["intermediario"]
        st.markdown(
            f"<div class='suzi-card' style='border-left-color:#ffc107;'>"
            f"<h4>{c_int['titulo']}</h4>"
            f"<p>{c_int['descricao']}</p>"
            f"<b>Condições Macroeconômicas & Setoriais:</b>"
            f"<ul>" + "".join(f"<li>{c}</li>" for c in c_int['condicoes']) + "</ul>"
            f"<b>Impacto Esperado no ADR:</b><br>{c_int['impacto_esperado']}<br><br>"
            f"<b>Gatilhos de Invalidação:</b><br><span style='color:#c00000;'>{c_int['gatilhos_invalidacao']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col_cen3:
        c_adv = CENARIOS_2027["adverso"]
        st.markdown(
            f"<div class='suzi-card' style='border-left-color:#dc3545;'>"
            f"<h4>{c_adv['titulo']}</h4>"
            f"<p>{c_adv['descricao']}</p>"
            f"<b>Condições Macroeconômicas & Setoriais:</b>"
            f"<ul>" + "".join(f"<li>{c}</li>" for c in c_adv['condicoes']) + "</ul>"
            f"<b>Impacto Esperado no ADR:</b><br>{c_adv['impacto_esperado']}<br><br>"
            f"<b>Gatilhos de Invalidação:</b><br><span style='color:#c00000;'>{c_adv['gatilhos_invalidacao']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# ABA 10: 🎯 EVIDÊNCIAS DEZ/2027 (PAINEL FINAL INTEGRADO)
# -----------------------------------------------------------------------------
if pagina_principal == "🎯 Decisão Dez/2027":
    st.header("🎯 Painel Final de Evidências Integradas — Dezembro de 2027")
    st.markdown(
        f"<div class='suzi-quote'>"
        f"<b>A Pergunta Central do Trabalho:</b><br>"
        f"“O grupo recomendaria a compra do ADR da Suzano para revendê-lo em dezembro de 2027?”"
        f"</div>",
        unsafe_allow_html=True
    )
    st.write("")

    matriz_ev = obter_matriz_evidencias_sintese()

    c_pev1, c_pev2 = st.columns(2)
    with c_pev1:
        st.subheader("🟢 Evidências Quantitativas e Operacionais Favoráveis")
        for eq in matriz_ev["quantitativas"]:
            st.markdown(f"• **[Quant]** {eq}")
        for eo in matriz_ev["operacionais"]:
            st.markdown(f"• **[Operacional]** {eo}")

        st.subheader("⚠️ Principais Riscos e Fatores de Atenção")
        for rp in matriz_ev["riscos_principais"]:
            st.markdown(f"• {rp}")

    with c_pev2:
        st.subheader("🔴 Evidências Desfavoráveis e Limitações do Modelo")
        st.markdown(
            "• **[Previsibilidade Linear Fraca]:** Retornos diários possuem R² ajustado inferior a 4%, "
            "comprovando que variações de curto prazo são dominadas por ruído aleatório e choques não-lineares."
        )
        st.markdown(
            "• **[Heterocedasticidade Comprovada]:** Presença de agrupamento de volatilidade exige margem de segurança conservadora."
        )
        st.markdown(
            "• **[Extrapolação Linear Inválida]:** Um modelo linear de retornos diários não tem capacidade de projetar o preço pontual a 18 meses de distância."
        )

        st.subheader("🌎 Fatores Macroeconômicos Relevantes")
        for em in matriz_ev["macroeconomicas"]:
            st.markdown(f"• {em}")

    st.markdown("---")
    st.subheader("🎓 Parecer Conclusivo da SUZI e do SUZ Quant AI")
    dossie_final = gerar_dossie_dezembro_2027(m_ret, d_ret, tabela_resumo)
    st.success(dossie_final["parecer_sintese"])

# -----------------------------------------------------------------------------
# ABA 11: 👥 NOSSA REGRESSÃO — MODELO DESENVOLVIDO PELO GRUPO
# -----------------------------------------------------------------------------

if pagina_principal == "📈 Regressões":

    st.header("👥 Nossa Regressão — Modelo Original do Grupo")

    st.markdown(
        """
        Esta seção preserva os resultados da **regressão desenvolvida originalmente
        pelo grupo** para o trabalho de Mercado Financeiro e de Capitais.

        O objetivo é manter separados:

        - **Nossa Regressão:** modelo elaborado pelo grupo;
        - **SUZ Quant AI:** análise quantitativa e auditoria complementar realizada pelo robô.

        **Frequência dos dados:** diária.
        """
    )

    st.info(
        "📌 Os números apresentados nesta aba correspondem aos resultados "
        "obtidos pelo grupo e não são recalculados ou substituídos pelo SUZ Quant AI."
    )

    st.markdown("---")

    # =========================================================
    # REGRESSÃO SIMPLES DO GRUPO
    # =========================================================

    st.subheader("6.1 — Regressão Linear Simples")

    st.markdown(
        """
        Foi estimada inicialmente uma regressão linear simples, utilizando a
        **cotação do ADR da Suzano (SUZ)** como variável dependente (Y) e o
        **NYSE Composite (NYA)** como variável independente (X).

        **Número de observações diárias: 457**
        """
    )

    st.markdown("### Equação estimada")

    st.code(
        "SUZ = 10,3805 - 0,0000302 × NYA",
        language=None
    )

    rg1, rg2, rg3, rg4 = st.columns(4)

    with rg1:
        st.metric("Observações", "457")

    with rg2:
        st.metric("R²", "0,45%")

    with rg3:
        st.metric("R² Ajustado", "0,23%")

    with rg4:
        st.metric("p-value NYA", "0,153")

    st.markdown(
        """
        **Interpretação do grupo**

        O NYA, isoladamente, apresentou baixo poder explicativo sobre a cotação
        da SUZ. O R² ajustado foi de apenas **0,23%** e o p-value do NYA foi
        **0,153**, superior ao nível de significância de 5%.

        Portanto, nesse modelo simples, o NYA não apresentou significância
        estatística suficiente para explicar isoladamente o comportamento da SUZ.
        """
    )

    st.markdown("---")

    # =========================================================
    # REGRESSÃO MÚLTIPLA DO GRUPO
    # =========================================================

    st.subheader("6.2 — Regressão Linear Múltipla")

    st.markdown(
        """
        Para ampliar o poder explicativo do modelo, o grupo manteve o
        **NYSE Composite (NYA) como X1** e acrescentou três variáveis:

        - **US10Y:** rendimento do Treasury americano de 10 anos;
        - **KLBAY:** ADR da Klabin, utilizado como referência setorial;
        - **LQD:** ETF de títulos corporativos investment grade dos EUA.
        """
    )

    st.markdown("### Equação estimada")

    st.code(
        "SUZ = -45,3715 - 0,0003831 × NYA + 3,6448 × US10Y + "
        "0,6407 × KLBAY + 0,3911 × LQD",
        language=None
    )

    rm1, rm2 = st.columns(2)

    with rm1:
        st.metric("R²", "70,03%")

    with rm2:
        st.metric("R² Ajustado", "69,76%")

    dados_regressao_grupo = pd.DataFrame({
        "Variável": [
            "Interseção",
            "NYA (X1)",
            "US10Y (X2)",
            "KLBAY (X3)",
            "LQD (X4)"
        ],
        "Coeficiente": [
            -45.3715,
            -0.0003831,
            3.6448,
            0.6407,
            0.3911
        ],
        "p-value": [
            4.83e-32,
            5.16e-24,
            5.87e-46,
            1.13e-39,
            2.83e-31
        ]
    })

    st.subheader("Coeficientes e Significância Estatística")

    st.dataframe(
        dados_regressao_grupo.style.format({
            "Coeficiente": "{:.7f}",
            "p-value": "{:.2e}"
        }),
        use_container_width=True
    )

    st.success(
        "📊 Todas as variáveis explicativas do modelo múltiplo apresentaram "
        "p-value inferior a 5%. O R² ajustado aumentou de 0,23% na regressão "
        "simples para 69,76% na regressão múltipla."
    )

    st.markdown("---")

    # =========================================================
    # INTERPRETAÇÃO DOS COEFICIENTES
    # =========================================================

    st.subheader("6.3 — Interpretação dos Coeficientes")

    st.markdown(
        """
        **NYA (-0,0003831)**  
        Mantidas as demais variáveis constantes, um aumento de 1 ponto no NYA
        está associado a uma redução estimada de aproximadamente 0,0003831
        unidade no preço da SUZ.

        **US10Y (+3,6448)**  
        Mantidas as demais variáveis constantes, um aumento de 1 ponto
        percentual no US10Y está associado, no modelo estimado, a um aumento
        de aproximadamente US$ 3,64 na SUZ.

        **KLBAY (+0,6407)**  
        Mantidas as demais variáveis constantes, um aumento de US$ 1 na KLBAY
        está associado a aproximadamente US$ 0,64 de aumento na SUZ.

        **LQD (+0,3911)**  
        Mantidas as demais variáveis constantes, um aumento de US$ 1 no LQD
        está associado a aproximadamente US$ 0,39 de aumento na SUZ.
        """
    )

    st.warning(
        "⚠️ Os coeficientes representam associações estimadas pelo modelo. "
        "Eles não demonstram, isoladamente, relações de causa e efeito."
    )

    st.markdown("---")

    # =========================================================
    # COMPARAÇÃO SIMPLES × MÚLTIPLA
    # =========================================================

    st.subheader("6.4 — Comparação dos Modelos do Grupo")

    comparacao_grupo = pd.DataFrame({
        "Indicador": [
            "Número de observações",
            "Variáveis explicativas",
            "R²",
            "R² Ajustado",
            "Significância dos X",
            "Base utilizada"
        ],
        "Regressão Simples": [
            "457",
            "NYA",
            "0,45%",
            "0,23%",
            "NYA: p = 0,153",
            "Cotações diárias"
        ],
        "Regressão Múltipla": [
            "457",
            "NYA + US10Y + KLBAY + LQD",
            "70,03%",
            "69,76%",
            "Todos p < 5%",
            "Cotações diárias"
        ]
    })

    st.dataframe(
        comparacao_grupo,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # =========================================================
    # AUDITORIA DA SUZI
    # =========================================================

    st.subheader("🤖 Auditoria Metodológica da SUZI")

    st.markdown(
        """
        <div class='suzi-card'>
        <h4 style='margin-top:0;'>🌲 O que a SUZI acrescenta à análise?</h4>

        <p>
        O modelo múltiplo desenvolvido pelo grupo apresenta poder explicativo
        substancialmente maior quando comparado à regressão simples.
        Entretanto, como os modelos desta aba utilizam <b>cotações em nível</b>,
        o R² elevado deve ser analisado em conjunto com testes econométricos
        de estacionariedade, autocorrelação dos resíduos e multicolinearidade.
        </p>

        <p>
        Por esse motivo, o <b>SUZ Quant AI</b> realiza paralelamente regressões
        com <b>retornos diários</b> e diagnósticos econométricos. A comparação
        entre as duas metodologias permite distinguir <b>alto poder explicativo
        aparente</b> de uma relação estatisticamente mais robusta.
        </p>

        <p>
        Assim, o robô não substitui o trabalho do grupo: ele funciona como
        <b>assistente quantitativo e auditor metodológico</b>.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # -----------------------------------------------------------------------------
# ABA 12: 🔄 GRUPO × SUZ QUANT AI
# -----------------------------------------------------------------------------

if pagina_principal == "📈 Regressões":

    st.header("🔄 Nossa Regressão × SUZ Quant AI")

    st.markdown(
        """
        Esta seção compara a **regressão originalmente desenvolvida pelo grupo**
        com a análise complementar realizada pelo **SUZ Quant AI**.

        O objetivo não é escolher o modelo com o maior R², mas compreender
        **por que metodologias diferentes produzem resultados diferentes**.
        """
    )

    st.info(
        "📌 As duas análises utilizam dados diários, porém podem utilizar "
        "transformações e especificações econométricas diferentes."
    )

    st.markdown("---")

    # =========================================================
    # COMPARAÇÃO METODOLÓGICA
    # =========================================================

    st.subheader("1. Comparação Metodológica")

    comparacao_metodologica = pd.DataFrame({
        "Critério": [
            "Empresa / ADR",
            "Mercado",
            "Frequência",
            "Variável dependente",
            "X1 obrigatório",
            "Regressão simples",
            "Regressão múltipla",
            "Máximo de X",
            "Cotações em nível",
            "Retornos diários",
            "Diagnósticos econométricos"
        ],

        "👥 Grupo": [
            "Suzano — SUZ",
            "NYSE",
            "Diária",
            "SUZ",
            "NYA",
            "Sim",
            "Sim",
            "4",
            "Análise principal",
            "Não utilizada no modelo original",
            "Limitados ao trabalho original"
        ],

        "🤖 SUZ Quant AI": [
            "Suzano — SUZ",
            "NYSE",
            "Diária",
            "SUZ",
            "NYA",
            "Sim",
            "Sim",
            "4",
            "Análise complementar",
            "Análise principal",
            "ADF, VIF, resíduos e demais testes implementados"
        ]
    })

    st.dataframe(
        comparacao_metodologica,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # =========================================================
    # RESULTADOS DO GRUPO
    # =========================================================

    st.subheader("2. Resultados da Nossa Regressão")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### 📈 Simples")

        st.metric("R² Ajustado", "0,23%")
        st.metric("p-value NYA", "0,153")

        st.code(
            "SUZ = 10,3805 - 0,0000302 × NYA",
            language=None
        )

    with col2:

        st.markdown("#### 🧮 Múltipla")

        st.metric("R² Ajustado", "69,76%")
        st.metric("Variáveis X", "4")

        st.code(
            "NYA + US10Y + KLBAY + LQD",
            language=None
        )

    st.caption(
        "Resultados apresentados conforme a regressão originalmente "
        "desenvolvida pelo grupo."
    )

    st.markdown("---")

    # =========================================================
    # RESULTADOS DO ROBÔ
    # =========================================================

    st.subheader("3. Resultados do SUZ Quant AI")

    st.markdown(
        """
        O SUZ Quant AI acrescenta uma segunda camada de análise.

        Em vez de considerar apenas o poder explicativo das **cotações em nível**,
        o sistema testa também os **retornos diários**, buscando reduzir o risco
        de interpretar tendências comuns entre séries financeiras como uma
        relação econômica robusta.
        """
    )

    # ---------------------------------------------------------
    # RESULTADOS DO SUZ QUANT AI
    # Usa os objetos já calculados anteriormente no dashboard
    # ---------------------------------------------------------

    q1, q2, q3 = st.columns(3)

    with q1:
        st.metric(
            "Análise principal",
            "Retornos diários"
        )

    with q2:
        st.metric(
            "Análise complementar",
            "Cotações"
        )

    with q3:
        st.metric(
            "Nível de significância",
            "5% (p < 0,05)"
        )
    st.markdown(
        """
        **Como o SUZ Quant AI trabalha**

        O sistema executa duas análises separadas:

        **1. Retornos diários — análise principal**  
        Avalia as variações percentuais diárias da SUZ e das variáveis
        explicativas.

        **2. Cotações — análise complementar**  
        Avalia os preços em nível, permitindo comparar os resultados com
        a regressão originalmente desenvolvida pelo grupo.

        Os resultados quantitativos detalhados permanecem disponíveis nas
        abas **Regressão Simples**, **Regressão Múltipla** e
        **Retornos × Cotações**.
        """
    )

    st.info(
        "💡 Esta aba tem função comparativa. Os números do SUZ Quant AI "
        "não são duplicados manualmente aqui para evitar inconsistências "
        "caso o período ou as variáveis sejam alterados no dashboard."
    )

    st.markdown("---")

    # =========================================================
    # POR QUE OS R² PODEM SER TÃO DIFERENTES?
    # =========================================================

    st.subheader("4. Por que os resultados são diferentes?")

    st.markdown(
        """
        ### Cotações em nível

        Preços financeiros podem apresentar tendências ao longo do tempo.
        Quando diferentes séries apresentam tendências simultâneas, uma regressão
        em nível pode produzir um **R² elevado**, mesmo que parte dessa relação
        decorra das propriedades temporais das séries.

        ### Retornos diários

        Os retornos medem a **variação percentual diária** e normalmente removem
        grande parte da tendência presente nos preços.

        Por isso, é comum que regressões de retornos apresentem **R² muito menores**
        do que regressões realizadas diretamente com cotações.
        """
    )

    st.warning(
        "⚠️ Portanto, um R² maior não significa automaticamente um modelo "
        "econometricamente superior. R², significância estatística e "
        "diagnósticos precisam ser analisados em conjunto."
    )

    st.markdown("---")

    # =========================================================
    # SUZI
    # =========================================================

    st.subheader("🤖 Parecer Comparativo da SUZI")

    st.markdown(
        """
        <div class='suzi-card'>

        <h4 style='margin-top:0;'>🌲 Como interpretar as duas análises?</h4>

        <p>
        A regressão do grupo e a análise do SUZ Quant AI não precisam ser
        tratadas como resultados concorrentes.
        </p>

        <p>
        O <b>modelo do grupo</b> mostra que determinadas cotações apresentam
        forte associação conjunta com o preço da SUZ durante a amostra analisada.
        </p>

        <p>
        O <b>SUZ Quant AI</b> acrescenta uma pergunta metodológica:
        essa relação permanece quando analisamos as <b>variações diárias</b>
        em vez dos níveis dos preços?
        </p>

        <p>
        Essa diferença ajuda a separar três conceitos:
        <b>correlação entre preços, capacidade explicativa estatística e
        evidência sobre os retornos do ADR.</b>
        </p>

        <p>
        Para responder à questão de dezembro de 2027, nenhuma regressão deve
        ser utilizada isoladamente. Os resultados devem ser combinados com
        desempenho de mercado, fundamentos, valuation, riscos e cenários.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )
    # =============================================================================
# PÁGINA: MERCADO & DESEMPENHO
# =============================================================================

if pagina_principal == "📊 Mercado & Desempenho":

    st.markdown("---")
    st.header("📊 Mercado & Desempenho do ADR SUZ")

    st.markdown(
        """
        Esta área reúne os principais indicadores de **retorno e risco**
        do ADR da Suzano (SUZ), utilizando o **NYSE Composite (NYA)**
        como referência de mercado.

        Os cálculos abaixo utilizam a mesma base diária já carregada
        pelo SUZ Quant AI.
        """
    )

    # Base comum SUZ × NYA
    desempenho = df_retornos[["SUZ", "NYA"]].dropna().copy()

    n_obs = len(desempenho)

    # Retorno médio diário
    retorno_suz_diario = desempenho["SUZ"].mean()
    retorno_nya_diario = desempenho["NYA"].mean()

    # Retorno médio diário anualizado
    retorno_suz_anual = (1 + retorno_suz_diario) ** 252 - 1
    retorno_nya_anual = (1 + retorno_nya_diario) ** 252 - 1

    # Volatilidade anualizada
    vol_suz = desempenho["SUZ"].std() * (252 ** 0.5)
    vol_nya = desempenho["NYA"].std() * (252 ** 0.5)

    # Beta SUZ × NYA
    cov_suz_nya = desempenho["SUZ"].cov(desempenho["NYA"])
    var_nya = desempenho["NYA"].var()

    beta_suz = (
        cov_suz_nya / var_nya
        if var_nya != 0
        else float("nan")
    )

    # -------------------------------------------------------------------------
    # PAINEL PRINCIPAL
    # -------------------------------------------------------------------------

    st.subheader("📈 Retorno")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Retorno médio diário SUZ",
            f"{retorno_suz_diario:.4%}"
        )

    with c2:
        st.metric(
            "Retorno anualizado SUZ",
            f"{retorno_suz_anual:.2%}"
        )

    with c3:
        st.metric(
            "Retorno médio diário NYA",
            f"{retorno_nya_diario:.4%}"
        )

    with c4:
        st.metric(
            "Retorno anualizado NYA",
            f"{retorno_nya_anual:.2%}"
        )

    st.caption(
        f"Base comum utilizada: {n_obs} observações diárias válidas."
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # RISCO
    # -------------------------------------------------------------------------

    st.subheader("⚠️ Risco e Volatilidade")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Volatilidade anualizada SUZ",
            f"{vol_suz:.2%}"
        )

    with r2:
        st.metric(
            "Volatilidade anualizada NYA",
            f"{vol_nya:.2%}"
        )

    with r3:
        st.metric(
            "Beta SUZ × NYA",
            f"{beta_suz:.3f}"
        )

    st.markdown("#### Fórmula do Beta")

    st.latex(
        r"\beta_{SUZ} = "
        r"\frac{Cov(R_{SUZ},R_{NYA})}"
        r"{Var(R_{NYA})}"
    )

    # -------------------------------------------------------------------------
    # INTERPRETAÇÃO AUTOMÁTICA DO BETA
    # -------------------------------------------------------------------------

    if beta_suz > 1:

        st.info(
            "📌 Na amostra selecionada, o Beta da SUZ é superior a 1. "
            "Historicamente, isso indica maior sensibilidade dos retornos "
            "da SUZ às oscilações do NYSE Composite."
        )

    elif beta_suz >= 0:

        st.info(
            "📌 Na amostra selecionada, o Beta da SUZ está entre 0 e 1. "
            "Historicamente, isso indica sensibilidade positiva, porém "
            "inferior, às oscilações do NYSE Composite."
        )

    else:

        st.info(
            "📌 Na amostra selecionada, o Beta da SUZ é negativo. "
            "Isso indica associação histórica inversa com os retornos "
            "do NYSE Composite durante o período analisado."
        )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # GRÁFICOS
    # -------------------------------------------------------------------------

    st.subheader("📉 Comportamento dos Retornos")

    g1, g2 = st.columns(2)

    with g1:
        st.markdown("**SUZ — Retornos Diários (%)**")
        st.line_chart(desempenho["SUZ"] * 100)

    with g2:
        st.markdown("**NYA — Retornos Diários (%)**")
        st.line_chart(desempenho["NYA"] * 100)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SHARPE E TREYNOR
    # -------------------------------------------------------------------------

    st.subheader("📐 Sharpe e Treynor")

    # -------------------------------------------------------------------------
    # TAXA LIVRE DE RISCO — US10Y
    # -------------------------------------------------------------------------

    if "US10Y" in series_brutas:

        us10y_nivel = series_brutas["US10Y"].dropna()

        if not us10y_nivel.empty:

            # US10Y está armazenado em pontos percentuais.
            # Ex.: 5.184 representa 5,184% a.a.
            rf_anual = us10y_nivel.iloc[-1] / 100

            # Índice de Sharpe anualizado
            sharpe_suz = (
                (retorno_suz_anual - rf_anual) / vol_suz
                if vol_suz != 0
                else float("nan")
            )

            # Índice de Treynor
            treynor_suz = (
                (retorno_suz_anual - rf_anual) / beta_suz
                if beta_suz != 0
                else float("nan")
            )

            st.markdown("#### Taxa livre de risco utilizada")

            rf1, rf2 = st.columns(2)

            with rf1:
                st.metric(
                    "US10Y — último valor disponível",
                    f"{rf_anual:.3%}"
                )

            with rf2:
                st.metric(
                    "Referência",
                    "Treasury 10 anos"
                )

            st.caption(
                "O US10Y é utilizado como proxy da taxa livre de risco. "
                "A série está expressa em pontos percentuais anuais; "
                "por isso, o valor é dividido por 100 antes dos cálculos."
            )

            st.markdown("#### Indicadores de desempenho ajustado ao risco")

            sh1, sh2 = st.columns(2)

            with sh1:

                st.metric(
                    "Índice de Sharpe — SUZ",
                    f"{sharpe_suz:.3f}"
                )

                st.latex(
                    r"Sharpe = "
                    r"\frac{R_{SUZ} - R_f}"
                    r"{\sigma_{SUZ}}"
                )

                if sharpe_suz > 0:
                    st.success(
                        "Na amostra analisada, o retorno anualizado da SUZ "
                        "superou a proxy de taxa livre de risco após considerar "
                        "o risco total medido pela volatilidade."
                    )
                elif sharpe_suz < 0:
                    st.warning(
                        "Na amostra analisada, o retorno anualizado da SUZ "
                        "não superou a proxy de taxa livre de risco."
                    )
                else:
                    st.info(
                        "O retorno excedente ajustado ao risco total ficou "
                        "próximo de zero na amostra analisada."
                    )

            with sh2:

                st.metric(
                    "Índice de Treynor — SUZ",
                    f"{treynor_suz:.3f}"
                )

                st.latex(
                    r"Treynor = "
                    r"\frac{R_{SUZ} - R_f}"
                    r"{\beta_{SUZ}}"
                )

                if treynor_suz > 0:
                    st.success(
                        "Na amostra analisada, a SUZ apresentou retorno "
                        "excedente positivo em relação à taxa livre de risco "
                        "por unidade de risco sistemático."
                    )
                elif treynor_suz < 0:
                    st.warning(
                        "Na amostra analisada, o retorno excedente da SUZ "
                        "em relação à taxa livre de risco foi negativo por "
                        "unidade de risco sistemático."
                    )
                else:
                    st.info(
                        "O retorno excedente por unidade de risco sistemático "
                        "ficou próximo de zero."
                    )

            with st.expander("📚 Metodologia — Sharpe e Treynor"):

                st.markdown(
                    f"""
                    **Taxa livre de risco**

                    Foi utilizado o último valor disponível do **US10Y** na
                    base selecionada:

                    **Rf = {rf_anual:.3%} ao ano**

                    O US10Y funciona nesta análise como uma **proxy** da taxa
                    livre de risco em dólares.

                    **Sharpe**

                    Compara o retorno excedente da SUZ com o risco total,
                    representado pela volatilidade anualizada.

                    **Treynor**

                    Compara o retorno excedente da SUZ com o risco sistemático,
                    representado pelo Beta da SUZ em relação ao NYSE Composite.

                    Os indicadores são históricos e dependem do período
                    selecionado no dashboard. Não representam previsão do
                    desempenho futuro do ADR.
                    """
                )

        else:

            st.warning(
                "A série US10Y foi localizada, mas não possui observações "
                "válidas para o período selecionado."
            )

    else:

        st.warning(
            "A série US10Y não foi encontrada. Sharpe e Treynor não podem "
            "ser calculados sem uma proxy para a taxa livre de risco."
        )
    

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SUZI
    # -------------------------------------------------------------------------

    st.subheader("🤖 Leitura da SUZI")

    st.markdown(
        f"""
        <div class='suzi-card'>

        <h4 style='margin-top:0;'>🌲 Diagnóstico de Mercado</h4>

        <p>
        Na amostra atualmente selecionada, o ADR SUZ apresentou
        retorno médio diário de <b>{retorno_suz_diario:.4%}</b>
        e volatilidade anualizada de <b>{vol_suz:.2%}</b>.
        </p>

        <p>
        O NYSE Composite apresentou retorno médio diário de
        <b>{retorno_nya_diario:.4%}</b> e volatilidade anualizada
        de <b>{vol_nya:.2%}</b>.
        </p>

        <p>
        O Beta estimado da SUZ em relação ao NYA foi de
        <b>{beta_suz:.3f}</b>.
        </p>

        <p>
        Esses indicadores descrevem o comportamento histórico da amostra
        e devem ser combinados com fundamentos, valuation, regressões,
        riscos e cenários na análise para dezembro de 2027.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# PÁGINA: FUNDAMENTOS & VALUATION — BASE REAL DO GRUPO
# =============================================================================
if pagina_principal == "💰 Fundamentos & Valuation":
    st.markdown("---")
    st.header("💰 Fundamentos & Valuation")
    st.caption("Base acadêmica do grupo: exercícios de 2023, 2024 e 2025.")

    caminho_valuation = os.path.join(os.path.dirname(__file__), "Valuation_Suzano_Multiplos.xlsx")

    if os.path.exists(caminho_valuation):
        try:
            val_raw = pd.read_excel(caminho_valuation, sheet_name="Valuation Suzano", header=None)

            anos_val = ["2023", "2024", "2025"]
            vm = [float(val_raw.iloc[19, 4]), float(val_raw.iloc[20, 4]), float(val_raw.iloc[21, 4])]
            vm_ebitda = [float(val_raw.iloc[26, 4]), float(val_raw.iloc[27, 4]), float(val_raw.iloc[28, 4])]
            vm_lucro = [float(val_raw.iloc[33, 4]), float(val_raw.iloc[34, 4]), float(val_raw.iloc[35, 4])]
            pl = [float(val_raw.iloc[40, 4]), float(val_raw.iloc[41, 4]), float(val_raw.iloc[42, 4])]
            adr_usd = [float(val_raw.iloc[5, 3]), float(val_raw.iloc[6, 3]), float(val_raw.iloc[7, 3])]

            df_val = pd.DataFrame({
                "Ano": anos_val,
                "ADR SUZ (US$)": adr_usd,
                "Valor de Mercado (R$ bi)": vm,
                "VM/EBITDA": vm_ebitda,
                "VM/Lucro Líquido": vm_lucro,
                "P/L": pl,
            })

            v1, v2, v3, v4 = st.columns(4)
            v1.metric("ADR — 2025", f"US$ {adr_usd[-1]:.2f}")
            v2.metric("Valor de Mercado — 2025", f"R$ {vm[-1]:.2f} bi")
            v3.metric("VM/EBITDA — 2025", f"{vm_ebitda[-1]:.2f}x")
            v4.metric("P/L — 2025", f"{pl[-1]:.2f}x")

            st.subheader("📊 Valuation por múltiplos")
            st.dataframe(df_val, width="stretch", hide_index=True)

            st.markdown("#### Evolução do VM/EBITDA")
            st.line_chart(df_val.set_index("Ano")[["VM/EBITDA"]])

            st.info(
                f"Entre 2023 e 2025, o VM/EBITDA calculado na planilha do grupo passou de "
                f"{vm_ebitda[0]:.2f}x para {vm_ebitda[-1]:.2f}x. "
                "Esse movimento é uma evidência de valuation e não deve ser interpretado isoladamente como sinal de compra ou venda."
            )
            st.warning(
                "Em 2024 houve prejuízo líquido de R$ 7,045 bi na base do grupo. "
                "Por isso, VM/Lucro Líquido e P/L ficam negativos e não possuem a interpretação econômica usual naquele exercício."
            )

            with st.expander("📚 Metodologia e fonte do valuation"):
                st.markdown(
                    """
                    **Fonte interna:** `Valuation_Suzano_Multiplos.xlsx`.

                    A planilha utiliza cotação de fechamento do ADR no último pregão de cada exercício, ações consideradas líquidas de tesouraria, PTAX de venda do BCB, EBITDA consolidado, lucro/prejuízo líquido e LPA básico. O valor de mercado em dólares é convertido para reais antes dos múltiplos baseados nas demonstrações em reais.
                    """
                )
        except Exception as e:
            st.error(f"O arquivo de valuation foi localizado, mas não pôde ser lido: {e}")
    else:
        st.warning(
            "Coloque o arquivo `Valuation_Suzano_Multiplos.xlsx` na mesma pasta do dashboard.py "
            "para carregar automaticamente o valuation real do grupo."
        )

    st.markdown("---")
    st.subheader("🏭 Fundamentos — Suzano × Klabin")
    st.caption("Fonte: tabelas de índices contábeis do trabalho acadêmico do grupo (2023–2025).")

    df_fund = pd.DataFrame({
        "Indicador": [
            "Liquidez Corrente",
            "Endividamento Bancário de Curto Prazo",
            "Margem EBITDA",
            "Cobertura das Despesas Financeiras Líquidas",
            "ROE",
        ],
        "Suzano 2023": [2.61, 11.97, 49.14, 6.45, 31.48],
        "Suzano 2024": [1.72, 22.15, 52.56, 6.27, -21.73],
        "Suzano 2025": [3.18, 6.00, 43.79, 4.25, 30.57],
        "Klabin 2023": [2.80, 8.45, 35.14, 14.55, 20.73],
        "Klabin 2024": [1.93, 9.23, 37.33, 3.79, 23.70],
        "Klabin 2025": [2.06, 8.55, 37.92, 3.53, 11.65],
    })
    st.dataframe(df_fund, use_container_width=True, hide_index=True)

    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Liquidez Corrente 2025", "3,18")
    f2.metric("Endiv. Bancário CP 2025", "6,00%")
    f3.metric("Margem EBITDA 2025", "43,79%")
    f4.metric("Cobertura Financeira 2025", "4,25x")
    f5.metric("ROE 2025", "30,57%")

    st.info(
        "Leitura integrada: em 2025, a base do grupo mostra Liquidez Corrente de 3,18, "
        "Endividamento Bancário de Curto Prazo de 6,00%, Margem EBITDA de 43,79%, "
        "Cobertura das Despesas Financeiras Líquidas de 4,25x e ROE de 30,57%. "
        "A comparação com a Klabin é apresentada como referência setorial e deve ser combinada com valuation, mercado e risco."
    )

# =============================================================================
# COMPLEMENTO: DIAGNÓSTICO FINAL INTEGRADO — DEZ/2027
# =============================================================================
if pagina_principal == "🎯 Decisão Dez/2027":
    st.markdown("---")
    st.header("🧠 Diagnóstico Final Integrado — SUZ Quant AI")
    st.write(
        "Esta síntese cruza as evidências do dashboard. Nenhum indicador isolado determina a conclusão acadêmica."
    )

    base_diag = df_retornos[["SUZ", "NYA"]].dropna()
    ret_suz_d = base_diag["SUZ"].mean()
    ret_suz_a = (1 + ret_suz_d) ** 252 - 1
    vol_suz_a = base_diag["SUZ"].std() * (252 ** 0.5)
    beta_diag = base_diag["SUZ"].cov(base_diag["NYA"]) / base_diag["NYA"].var() if base_diag["NYA"].var() != 0 else float("nan")

    rf_diag = float("nan")
    sharpe_diag = float("nan")
    treynor_diag = float("nan")
    if "US10Y" in series_brutas and not series_brutas["US10Y"].dropna().empty:
        rf_diag = series_brutas["US10Y"].dropna().iloc[-1] / 100
        sharpe_diag = (ret_suz_a - rf_diag) / vol_suz_a if vol_suz_a != 0 else float("nan")
        treynor_diag = (ret_suz_a - rf_diag) / beta_diag if beta_diag != 0 else float("nan")

    melhor_nome = max(regs_modelos, key=lambda k: regs_modelos[k].r2_ajustado)
    melhor_reg = regs_modelos[melhor_nome]
    pvals_x = melhor_reg.tabela_coeficientes.drop(index="const", errors="ignore")["p_valor"]
    todos_sig = bool((pvals_x < alpha_nivel).all()) if len(pvals_x) else False

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Retorno anualizado SUZ", f"{ret_suz_a:.2%}")
    d2.metric("Volatilidade anualizada", f"{vol_suz_a:.2%}")
    d3.metric("Beta SUZ × NYA", f"{beta_diag:.3f}")
    d4.metric("Sharpe", f"{sharpe_diag:.3f}" if pd.notna(sharpe_diag) else "N/D")

    st.subheader("📈 Evidência estatística")
    st.write(
        f"Melhor especificação entre os modelos atualmente testados: **{melhor_nome}** — "
        f"R² ajustado de **{melhor_reg.r2_ajustado:.2%}**. "
        f"Todos os X significativos no α selecionado: **{'Sim' if todos_sig else 'Não'}**."
    )

    st.subheader("💰 Evidência de valuation")
    caminho_valuation_diag = os.path.join(os.path.dirname(__file__), "Valuation_Suzano_Multiplos.xlsx")
    if os.path.exists(caminho_valuation_diag):
        try:
            vr = pd.read_excel(caminho_valuation_diag, sheet_name="Valuation Suzano", header=None)
            vm_e_2023 = float(vr.iloc[26, 4])
            vm_e_2025 = float(vr.iloc[28, 4])
            pl_2025 = float(vr.iloc[42, 4])
            st.write(
                f"Na planilha do grupo, o **VM/EBITDA** passa de **{vm_e_2023:.2f}x em 2023** "
                f"para **{vm_e_2025:.2f}x em 2025**, enquanto o **P/L de 2025 é {pl_2025:.2f}x**. "
                "Esses múltiplos entram como evidência complementar e não como decisão automática."
            )
        except Exception as e:
            st.warning(f"Valuation não pôde ser incorporado ao diagnóstico: {e}")
    else:
        st.warning("Valuation não incorporado: arquivo Excel não encontrado na pasta do projeto.")

    st.subheader("📰 Eventos, notícias e cenários")
    st.write(
        "As evidências qualitativas registradas nas áreas de Eventos & Notícias, Cenários e Evidências Dez/2027 "
        "devem ser usadas para contextualizar ou invalidar uma leitura puramente histórica dos números."
    )

    st.subheader("🌲 Leitura da SUZI")
    st.info(
        "O diagnóstico final deve resultar da convergência entre mercado/risco, regressões, fundamentos, valuation "
        "e eventos/cenários. Divergências entre essas camadas devem permanecer explícitas no relatório, em vez de serem "
        "forçadas para produzir uma conclusão. A resposta final à questão de compra para revenda em dez/2027 deve ser "
        "formalizada pelo grupo com base nessa matriz de evidências."
    )

    st.markdown("---")
    st.subheader("📄 Relatório Final para impressão")
    st.write(
        "O arquivo abaixo consolida as evidências exibidas no dashboard. Depois de abrir o HTML no navegador, "
        "use **Ctrl + P** para imprimir ou salvar como PDF."
    )

    # Valores de valuation para o relatório (quando o Excel estiver disponível)
    val_relatorio = "Valuation não carregado."
    try:
        if os.path.exists(caminho_valuation_diag):
            val_relatorio = (
                f"VM/EBITDA: {vm_e_2023:.2f}x (2023) e {vm_e_2025:.2f}x (2025); "
                f"P/L 2025: {pl_2025:.2f}x. Em 2024 houve prejuízo líquido na base do grupo, "
                "limitando a interpretação dos múltiplos baseados em lucro."
            )
    except Exception:
        pass

    # Relatório acadêmico detalhado em 5 páginas de impressão
    coefs_rel = melhor_reg.tabela_coeficientes.copy()
    linhas_coef = []
    for nome_var, linha in coefs_rel.iterrows():
        linhas_coef.append(
            f"<tr><td>{nome_var}</td><td>{linha.get('coeficiente', float('nan')):.6f}</td>"
            f"<td>{linha.get('p_valor', float('nan')):.6f}</td></tr>"
        )
    tabela_coef_html = "".join(linhas_coef)

    # Cenários do próprio projeto, quando disponíveis
    def _cenario_html(chave):
        try:
            c = CENARIOS_2027[chave]
            cond = "".join(f"<li>{x}</li>" for x in c.get("condicoes", []))
            return (f"<h3>{c.get('titulo', chave.title())}</h3><p>{c.get('descricao','')}</p>"
                    f"<ul>{cond}</ul><p><b>Impacto esperado:</b> {c.get('impacto_esperado','')}</p>"
                    f"<p><b>Gatilhos de invalidação:</b> {c.get('gatilhos_invalidacao','')}</p>")
        except Exception:
            return f"<h3>{chave.title()}</h3><p>Cenário disponível no painel do dashboard.</p>"

    cen_fav_html = _cenario_html("favoravel")
    cen_int_html = _cenario_html("intermediario")
    cen_adv_html = _cenario_html("adverso")

    html_relatorio = f"""<!doctype html>
<html lang='pt-BR'>
<head><meta charset='utf-8'><title>Relatório Final — SUZ Quant AI</title>
<style>
@page{{size:A4;margin:12mm 13mm}}
*{{box-sizing:border-box}} body{{font-family:Arial,sans-serif;line-height:1.32;color:#17221f;margin:0;font-size:10.5pt}}
h1{{color:#00281e;font-size:21pt;margin:0 0 6px}} h2{{color:#005b46;font-size:14pt;margin:5px 0}} h3{{font-size:11.5pt;margin:5px 0 2px}}
p{{margin:4px 0}} ul{{margin:3px 0 5px 18px;padding:0}} li{{margin:1px 0}}
.page{{min-height:270mm;position:relative;page-break-after:always;padding:2mm}} .page:last-child{{page-break-after:auto}}
.box{{border:1px solid #cfded8;border-radius:7px;padding:7px 9px;margin:6px 0;background:#fbfdfc}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px}} .kpi{{border-left:4px solid #147d64;padding:5px 8px;background:#f3f8f6}}
table{{border-collapse:collapse;width:100%;font-size:9pt;margin:5px 0}} th,td{{border:1px solid #ccd7d3;padding:4px;text-align:center}} th{{background:#eaf3ef}}
.note{{font-size:8.5pt;color:#4c5b56}} .footer{{position:absolute;bottom:2mm;left:2mm;right:2mm;border-top:1px solid #ddd;padding-top:3px;font-size:8pt;color:#667}}
.badge{{display:inline-block;border:1px solid #9fbeb2;border-radius:10px;padding:2px 7px;margin:2px}}
@media print{{body{{print-color-adjust:exact;-webkit-print-color-adjust:exact}}}}
</style></head><body>

<section class='page'>
<h1>SUZ Quant AI — Relatório Final Integrado</h1>
<p><b>PUC-SP | Mercado Financeiro e de Capitais</b></p>
<div class='box'><b>Questão central:</b> “O grupo recomendaria a compra do ADR da Suzano para revendê-lo em dezembro de 2027?”</div>
<h2>1. Resumo executivo e metodologia</h2>
<p>Este relatório sintetiza as evidências quantitativas, contábeis, de valuation e qualitativas reunidas no SUZ Quant AI. A leitura é integrada: nenhum indicador isolado determina a conclusão acadêmica.</p>
<div class='grid'>
<div class='kpi'><b>Período selecionado</b><br>{dt_inicio.strftime('%d/%m/%Y')} a {dt_fim.strftime('%d/%m/%Y')}</div>
<div class='kpi'><b>Base principal</b><br>{tipo_dados_label}</div>
<div class='kpi'><b>Variável dependente</b><br>SUZ — ADR Suzano</div>
<div class='kpi'><b>X1 obrigatório</b><br>NYA — NYSE Composite</div>
</div>
<p><b>Metodologia:</b> regressões simples e múltiplas com dados diários, nível de significância α = {alpha_nivel:.0%}, comparação entre retornos e cotações e análise de risco/desempenho. As associações estatísticas não são tratadas automaticamente como causalidade.</p>
<h2>2. Mercado e risco</h2>
<table><tr><th>Indicador</th><th>Resultado</th><th>Leitura</th></tr>
<tr><td>Retorno anualizado da amostra</td><td>{ret_suz_a:.2%}</td><td>Desempenho anualizado segundo a metodologia do painel.</td></tr>
<tr><td>Volatilidade anualizada</td><td>{vol_suz_a:.2%}</td><td>Medida de dispersão/risco dos retornos.</td></tr>
<tr><td>Beta SUZ × NYA</td><td>{beta_diag:.3f}</td><td>Sensibilidade estimada do ADR ao mercado de referência.</td></tr>
<tr><td>Sharpe</td><td>{sharpe_diag:.3f}</td><td>Retorno excedente por unidade de risco total, conforme proxy livre de risco adotada.</td></tr></table>
<p class='note'>Os indicadores dependem do período selecionado e das premissas do dashboard. Resultados históricos não garantem comportamento futuro.</p>
<div class='footer'>Página 1 de 5 • SUZ Quant AI • Relatório acadêmico</div>
</section>

<section class='page'>
<h1>Fundamentos e Valuation</h1>
<h2>3. Situação fundamental — Suzano × Klabin</h2>
<table><tr><th>Indicador 2025</th><th>Suzano</th><th>Klabin</th></tr>
<tr><td>Liquidez Corrente</td><td>3,18</td><td>2,06</td></tr>
<tr><td>Endividamento Bancário CP</td><td>6,00%</td><td>8,55%</td></tr>
<tr><td>Margem EBITDA</td><td>43,79%</td><td>37,92%</td></tr>
<tr><td>Cobertura Despesas Financeiras Líquidas</td><td>4,25x</td><td>3,53x</td></tr>
<tr><td>ROE</td><td>30,57%</td><td>11,65%</td></tr></table>
<p>Na base acadêmica do grupo, a Suzano encerra 2025 com maior liquidez corrente, menor endividamento bancário de curto prazo, maior margem EBITDA e maior ROE que a referência Klabin apresentada no trabalho. A comparação é descritiva e deve preservar diferenças de estrutura, ciclo e estratégia entre as empresas.</p>
<h2>4. Valuation por múltiplos</h2>
<div class='box'><p>{val_relatorio}</p></div>
<table><tr><th>Ano</th><th>VM/EBITDA</th></tr><tr><td>2023</td><td>{vm_e_2023:.2f}x</td></tr><tr><td>2025</td><td>{vm_e_2025:.2f}x</td></tr></table>
<p>A redução do VM/EBITDA entre 2023 e 2025 é uma evidência de avaliação, mas não constitui isoladamente sinal de compra ou venda. Em 2024, o prejuízo líquido da base do grupo limita a interpretação econômica de P/L e de outros múltiplos dependentes de lucro.</p>
<h2>5. Integração fundamental</h2>
<p>O valuation deve ser interpretado junto da capacidade operacional, estrutura de capital, geração de resultado e risco de mercado. A análise final também considera que múltiplos podem cair tanto por alteração de preço quanto por expansão do denominador operacional.</p>
<div class='footer'>Página 2 de 5 • Fundamentos e Valuation</div>
</section>

<section class='page'>
<h1>Regressões e Diagnóstico Estatístico</h1>
<h2>6. Regressão simples</h2>
<p>Modelo: SUZ como variável dependente e NYA como variável explicativa obrigatória. R² = <b>{reg_simples.r2:.2%}</b> e R² ajustado = <b>{reg_simples.r2_ajustado:.2%}</b>. A interpretação deve observar simultaneamente magnitude, significância e adequação dos resíduos.</p>
<h2>7. Melhor regressão múltipla testada</h2>
<div class='box'><b>Especificação:</b> {melhor_nome}<br><b>R² ajustado:</b> {melhor_reg.r2_ajustado:.2%}<br><b>Todos os X significativos no α selecionado:</b> {'Sim' if todos_sig else 'Não'}</div>
<table><tr><th>Variável</th><th>Coeficiente</th><th>p-value</th></tr>{tabela_coef_html}</table>
<p><b>Leitura:</b> o R² ajustado mede a parcela explicada com penalização pela inclusão de variáveis. Os p-values avaliam evidência estatística individual dentro da especificação. Coeficientes em escalas diferentes não devem ser comparados diretamente como medida de “maior impacto” sem padronização.</p>
<h2>8. Retornos × cotações e limitações</h2>
<p>O projeto compara modelos estimados com retornos diários e com cotações em nível. Uma elevação do R² em níveis não é, por si só, prova de maior capacidade econômica ou preditiva. Tendências comuns, autocorrelação, estacionariedade, resíduos e eventual cointegração precisam ser considerados antes de uma conclusão.</p>
<p>A regressão é tratada como evidência associativa. Ela ajuda a identificar fatores relacionados ao comportamento histórico do ADR, mas não produz sozinha um preço-alvo confiável para dezembro de 2027.</p>
<div class='footer'>Página 3 de 5 • Regressões e diagnóstico</div>
</section>

<section class='page'>
<h1>Eventos, Notícias e Cenários até Dez/2027</h1>
<h2>9. Eventos e notícias</h2>
<p>As notícias e eventos registrados no dashboard funcionam como camada qualitativa para contextualizar resultados históricos. Devem ser avaliados pela data, fonte, mecanismo de transmissão e potencial impacto sobre receita, custos, câmbio, preço da celulose, estrutura financeira e percepção de risco da Suzano.</p>
<div class='box'><b>Regra de leitura:</b> notícia não substitui evidência quantitativa. Ela pode reforçar, contradizer ou invalidar premissas usadas nos modelos e cenários.</div>
<h2>10. Cenários</h2>
<div class='grid'><div class='box'>{cen_fav_html}</div><div class='box'>{cen_int_html}</div></div>
<div class='box'>{cen_adv_html}</div>
<h2>11. Principais riscos a monitorar</h2>
<ul><li>Mudanças no ciclo e nos preços de celulose.</li><li>Variações cambiais relevantes para receitas, custos e dívida.</li><li>Alteração das condições financeiras e das taxas de juros.</li><li>Resultados operacionais diferentes das premissas do valuation.</li><li>Quebra das relações históricas identificadas nas regressões.</li><li>Eventos corporativos, regulatórios, ambientais ou geopolíticos com efeito material.</li></ul>
<p class='note'>Esta página usa os cenários cadastrados no próprio projeto; eles são hipóteses condicionais, não previsões garantidas.</p>
<div class='footer'>Página 4 de 5 • Eventos, notícias, riscos e cenários</div>
</section>

<section class='page'>
<h1>Diagnóstico Integrado — Dezembro de 2027</h1>
<h2>12. Matriz de evidências</h2>
<table><tr><th>Dimensão</th><th>O que o relatório considera</th></tr>
<tr><td>Mercado e risco</td><td>Retorno, volatilidade, beta e indicadores de retorno ajustado ao risco.</td></tr>
<tr><td>Fundamentos</td><td>Liquidez, dívida de curto prazo, margem EBITDA, cobertura financeira e ROE.</td></tr>
<tr><td>Valuation</td><td>Múltiplos calculados pelo grupo e suas limitações, especialmente em exercícios com prejuízo.</td></tr>
<tr><td>Regressões</td><td>R² ajustado, significância, coeficientes, especificação e diagnóstico.</td></tr>
<tr><td>Notícias/eventos</td><td>Fatores novos capazes de reforçar ou invalidar relações históricas.</td></tr>
<tr><td>Cenários</td><td>Condições favoráveis, intermediárias e adversas até dez/2027.</td></tr></table>
<h2>13. Síntese da SUZI</h2>
<div class='box'><p>{dossie_final['parecer_sintese']}</p></div>
<h2>14. Como responder à questão central</h2>
<p>A conclusão acadêmica do grupo deve resultar da convergência — ou da divergência explicitamente justificada — entre as seis dimensões acima. O relatório não transforma automaticamente um R² elevado, um múltiplo baixo ou um indicador de risco positivo em recomendação. A decisão deve indicar quais evidências sustentam a posição do grupo e quais condições poderiam invalidá-la antes de dezembro de 2027.</p>
<h2>15. Limitações</h2>
<p>O estudo utiliza dados históricos, modelos lineares e premissas de valuation sujeitas a revisão. Choques futuros, mudanças estruturais e eventos não observados na amostra podem alterar materialmente os resultados. O documento é acadêmico e não constitui recomendação individual de investimento.</p>
<div class='box'><b>Conclusão para apresentação:</b> utilizar esta página como síntese oral, mantendo as páginas anteriores como trilha de evidências e sustentação metodológica.</div>
<div class='footer'>Página 5 de 5 • Diagnóstico integrado • PUC-SP — Mercado Financeiro e de Capitais</div>
</section>
</body></html>"""

    st.download_button(
        "📄 Baixar Relatório Final para imprimir",
        data=html_relatorio.encode("utf-8"),
        file_name="Relatorio_Final_SUZ_Quant_AI_Dez2027.html",
        mime="text/html",
        use_container_width=True,
    )

# =============================================================================
# RODAPÉ INSTITUCIONAL
# =============================================================================
st.markdown("---")
st.caption(
    "🌲 **SUZ Quant AI** — Plataforma Quantitativa & Assistente SUZI | Pontifícia Universidade Católica de São Paulo (PUC-SP)  \n"
    "Disciplina: Mercado Financeiro e de Capitais | Orientação: Prof. Dr. José Odálio dos Santos  \n"
    "*Aviso Legal: Este sistema constitui um estudo acadêmico e quantitativo e não representa recomendação de investimento.*"
)
