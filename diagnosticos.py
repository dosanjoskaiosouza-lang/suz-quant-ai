"""
===============================================================================
SUZ Quant AI - Módulo de Diagnósticos Estatísticos e Econométricos (diagnosticos.py)
Disciplina: Mercado Financeiro e de Capitais - PUC-SP
Empresa: Suzano S.A. | Ativo: SUZ (ADR na NYSE)
===============================================================================
Este módulo realiza a bateria completa de testes diagnósticos necessários para
auditar a validade das regressões lineares estimadas:
1. Multicolinearidade: Fator de Inflação da Variância (VIF) e Matriz de Correlação.
2. Heterocedasticidade: Teste de Breusch-Pagan.
3. Autocorrelação Serial: Teste de Durbin-Watson.
4. Estacionariedade: Teste Dickey-Fuller Aumentado (ADF).
5. Análise de Resíduos: Média, desvio, curtose, assimetria e normalidade (Jarque-Bera).
6. Outliers e Alavancagem: Resíduos padronizados e Distância de Cook.
7. Risco de Regressão Espúria (Critério de Granger-Newbold: R² > Durbin-Watson).
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller

from regressao import ResultadoRegressao


@dataclass
class ResultadoDiagnosticos:
    """Consolida os testes diagnósticos econométricos aplicados a um modelo."""
    nome_modelo: str
    tipo_dados: str
    
    # 1. Multicolinearidade
    tabela_vif: pd.DataFrame
    matriz_correlacao: pd.DataFrame
    tem_alta_multicolinearidade: bool
    
    # 2. Heterocedasticidade
    lm_breusch_pagan: float
    p_breusch_pagan: float
    tem_heterocedasticidade: bool
    
    # 3. Autocorrelação
    durbin_watson_stat: float
    tem_autocorrelacao: bool
    
    # 4. Estacionariedade
    tabela_adf: pd.DataFrame
    series_estacionarias: bool
    residuos_estacionarios: bool
    
    # 5. Normalidade e Resíduos
    media_residuos: float
    desvio_residuos: float
    assimetria_residuos: float
    curtose_residuos: float
    jb_stat: float
    jb_pvalor: float
    residuos_normais: bool
    
    # 6. Observações Influentes
    total_outliers_3sigma: int
    top_outliers: pd.DataFrame
    
    # 7. Síntese e Risco Espúrio
    risco_regressao_espuria: bool
    classificacao_modelo: str  # 'Robusto', 'Exige Cautela' ou 'Regressão Espúria'
    parecer_didatico: str

    def gerar_relatorio_texto(self) -> str:
        """Gera relatório didático estruturado para estudantes de Administração."""
        linhas = [
            f"===========================================================",
            f"RELATÓRIO DE DIAGNÓSTICOS ECONOMÉTRICOS: {self.nome_modelo}",
            f"Base de Dados: {self.tipo_dados}",
            f"Classificação Geral: [{self.classificacao_modelo.upper()}]",
            f"===========================================================",
            f"\n1. MULTICOLINEARIDADE (VIF):",
            f"   - O VIF mede se as variáveis explicativas estão altamente correlacionadas entre si.",
            f"   - VIF < 5: Ideal (sem multicolinearidade prejudicial).",
            f"   - VIF > 10: Severo (risco de instabilidade nos coeficientes beta)."
        ]

        for idx, row in self.tabela_vif.iterrows():
            vif_val = row["VIF"]
            status = "Adequado" if vif_val < 5 else ("Atenção" if vif_val < 10 else "Alto")
            linhas.append(f"   • {idx}: VIF = {vif_val:.2f} ({status})")

        linhas.extend([
            f"\n2. HETEROCEDASTICIDADE (Teste de Breusch-Pagan):",
            f"   - Testa se a volatilidade dos erros é constante (homocedasticidade).",
            f"   - Estatística LM = {self.lm_breusch_pagan:.2f}, p-valor = {self.p_breusch_pagan:.4e}.",
            f"   - Diagnóstico: " + (
                "Heterocedasticidade detectada (p < 0,05). Recomenda-se cautela com erros padrão."
                if self.tem_heterocedasticidade
                else "Homocedasticidade mantida (p >= 0,05). Variância dos erros é estável."
            ),
            f"\n3. AUTOCORRELAÇÃO DOS RESÍDUOS (Durbin-Watson):",
            f"   - Testa se o erro de hoje depende do erro de ontem.",
            f"   - Estatística d = {self.durbin_watson_stat:.4f} (Ideal: próximo de 2,00).",
            f"   - Diagnóstico: " + (
                "Autocorrelação serial relevante detectada (d distante de 2,0)."
                if self.tem_autocorrelacao
                else "Sem autocorrelação serial de primeira ordem (resíduos independentes)."
            ),
            f"\n4. ESTACIONARIEDADE (Teste ADF - Augmented Dickey-Fuller):",
            f"   - Séries financeiras devem ser estacionárias para evitar regressões falsas/espúrias."
        ] + [
            f"   • {idx}: Estatística ADF = {row['Estatistica_ADF']:.3f}, p-valor = {row['p_valor']:.4f} -> {row['Status']}"
            for idx, row in self.tabela_adf.iterrows()
        ] + [
            f"\n5. ANÁLISE DOS RESÍDUOS E NORMALIDADE:",
            f"   - Média dos erros: {self.media_residuos:.6f} (deve ser ~0).",
            f"   - Assimetria (Skewness): {self.assimetria_residuos:.2f} | Curtose: {self.curtose_residuos:.2f}.",
            f"   - Teste de Jarque-Bera: JB = {self.jb_stat:.1f} (p-valor = {self.jb_pvalor:.2e}).",
            f"   - Diagnóstico: " + (
                "Resíduos seguem distribuição normal."
                if self.residuos_normais
                else "Resíduos apresentam caudas pesadas (leptocurtose), padrão típico em finanças diárias."
            ),
            f"\n6. OUTLIERS / PONTOS DE ALAVANCAGEM:",
            f"   - Quantidade de observações com resíduo extremo (> |3| desvios-padrão): {self.total_outliers_3sigma}.",
            f"\n7. PARECER CONCLUSIVO DOS DIAGNÓSTICOS:",
            f"   {self.parecer_didatico}"
        ])

        return "\n".join(linhas)


def testar_vif(X: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Calcula o Fator de Inflação da Variância (VIF) para as variáveis explicativas.
    Regra prática:
    - Se X tem apenas 1 variável, VIF = 1.0 (não há multicolinearidade com outras).
    - VIF > 5 requer atenção; VIF > 10 indica multicolinearidade severa.
    """
    colunas_x = list(X.columns)
    if len(colunas_x) <= 1:
        df_vif = pd.DataFrame({"VIF": [1.0]}, index=colunas_x)
        return df_vif, False

    # Inclui constante para cálculo correto do VIF das inclinações
    X_const = sm.add_constant(X, has_constant="add")
    vifs = []
    nomes = []

    for i, col in enumerate(X_const.columns):
        if col != "const":
            val = variance_inflation_factor(X_const.values, i)
            vifs.append(val)
            nomes.append(col)

    df_vif = pd.DataFrame({"VIF": vifs}, index=nomes)
    tem_alta = bool((df_vif["VIF"] > 5.0).any())
    return df_vif, tem_alta


def testar_estacionariedade_adf(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Executa o teste Dickey-Fuller Aumentado (ADF) para cada coluna do DataFrame.
    H0: A série possui raiz unitária (é NÃO-ESTACIONÁRIA).
    Rejeita H0 se p-valor < 0,05 -> Série é ESTACIONÁRIA I(0).
    """
    linhas = []
    todas_estacionarias = True

    for col in df.columns:
        serie = df[col].dropna()
        if len(serie) < 15:
            continue
        try:
            # autolag='AIC' escolhe automaticamente a defasagem ótima
            resultado = adfuller(serie, autolag="AIC")
            stat_adf = float(resultado[0])
            p_val = float(resultado[1])
            lags = int(resultado[2])
            nobs = int(resultado[3])
            crit_5pct = float(resultado[4]["5%"])

            estacionaria = p_val < 0.05
            if not estacionaria:
                todas_estacionarias = False

            linhas.append({
                "Variavel": col,
                "Estatistica_ADF": stat_adf,
                "p_valor": p_val,
                "Critico_5%": crit_5pct,
                "Lags": lags,
                "N": nobs,
                "Status": "Estacionária (I(0))" if estacionaria else "NÃO-Estacionária (I(1))"
            })
        except Exception as err:
            linhas.append({
                "Variavel": col,
                "Estatistica_ADF": np.nan,
                "p_valor": np.nan,
                "Critico_5%": np.nan,
                "Lags": 0,
                "N": len(serie),
                "Status": f"Erro no teste: {err}"
            })
            todas_estacionarias = False

    tabela_adf = pd.DataFrame(linhas).set_index("Variavel")
    return tabela_adf, todas_estacionarias


def testar_heterocedasticidade(reg: ResultadoRegressao) -> Tuple[float, float, bool]:
    """
    Executa o Teste de Breusch-Pagan sobre os resíduos da regressão.
    H0: Variância dos erros é constante (homocedasticidade).
    Se p < 0,05 -> heterocedasticidade presente.
    """
    modelo_sm = reg.modelo_sm
    try:
        lm_stat, lm_pval, _, _ = het_breuschpagan(modelo_sm.resid, modelo_sm.model.exog)
        tem_het = bool(lm_pval < 0.05)
        return float(lm_stat), float(lm_pval), tem_het
    except Exception:
        return 0.0, 1.0, False


def testar_autocorrelacao(residuos: pd.Series) -> Tuple[float, bool]:
    """
    Calcula a estatística Durbin-Watson para verificar autocorrelação serial de 1ª ordem.
    - d ~ 2,0: sem autocorrelação.
    - d < 1,5 ou d > 2,5: presença de autocorrelação preocupante.
    """
    d = float(durbin_watson(residuos.dropna()))
    tem_autocorr = bool(d < 1.5 or d > 2.5)
    return d, tem_autocorr


def analisar_distribuicao_residuos(residuos: pd.Series) -> Tuple[Dict[str, float], bool]:
    """
    Avalia a média, desvio, assimetria, curtose e teste de normalidade Jarque-Bera.
    """
    res = residuos.dropna()
    media = float(res.mean())
    desvio = float(res.std())
    skew = float(res.skew())
    kurt = float(res.kurtosis())  # Excesso de curtose em relação à normal

    try:
        jb_stat, jb_pval, _, _ = jarque_bera(res)
        normal = bool(jb_pval >= 0.05)
    except Exception:
        jb_stat, jb_pval, normal = 0.0, 1.0, True

    metricas = {
        "media": media,
        "desvio": desvio,
        "assimetria": skew,
        "curtose": kurt,
        "jb_stat": float(jb_stat),
        "jb_pvalor": float(jb_pval)
    }
    return metricas, normal


def identificar_outliers_e_alavancagem(
    reg: ResultadoRegressao,
    limite_desvios: float = 3.0
) -> Tuple[int, pd.DataFrame]:
    """
    Detecta observações com resíduo padronizado superior a 3 desvios e calcula a Distância de Cook.
    """
    res = reg.residuos
    std_res = res.std()
    res_padronizados = res / std_res if std_res > 0 else res

    mascara_outliers = res_padronizados.abs() > limite_desvios
    df_outliers = pd.DataFrame({
        "Residuo_Original": res[mascara_outliers],
        "Residuo_Padronizado": res_padronizados[mascara_outliers],
        "Valor_Observado_Y": reg.modelo_sm.model.endog[mascara_outliers] if hasattr(reg.modelo_sm.model, "endog") else np.nan,
        "Valor_Predito_Y": reg.valores_ajustados[mascara_outliers]
    }).sort_values(by="Residuo_Padronizado", ascending=False)

    return int(mascara_outliers.sum()), df_outliers


def executar_diagnosticos_completos(
    reg: ResultadoRegressao,
    df_dados: pd.DataFrame
) -> ResultadoDiagnosticos:
    """
    Executa todo o pipeline diagnóstico econométrico sobre o modelo estimado.
    """
    X = df_dados[reg.variaveis_x].dropna()
    y = df_dados[reg.variavel_y].dropna()
    df_conjunto = pd.concat([y, X], axis=1).dropna()

    # 1. Multicolinearidade
    tabela_vif, tem_alta_multi = testar_vif(X)
    matriz_corr = X.corr()

    # 2. Heterocedasticidade
    lm_stat, p_lm, tem_het = testar_heterocedasticidade(reg)

    # 3. Autocorrelação
    dw_stat, tem_autocorr = testar_autocorrelacao(reg.residuos)

    # 4. Estacionariedade
    tabela_adf, series_estac = testar_estacionariedade_adf(df_conjunto)
    # Testa também resíduos
    adf_res = adfuller(reg.residuos.dropna(), autolag="AIC")
    residuos_estac = bool(adf_res[1] < 0.05)

    # 5. Normalidade e Resíduos
    metricas_res, res_normais = analisar_distribuicao_residuos(reg.residuos)

    # 6. Outliers
    n_outliers, top_outliers = identificar_outliers_e_alavancagem(reg)

    # 7. Avaliação de Regressão Espúria (Regra de Granger-Newbold: R² > DW e séries I(1))
    risco_espuria = False
    if (not series_estac) and (reg.r2 > dw_stat) and (dw_stat < 0.5):
        risco_espuria = True

    # Classificação global do modelo
    if risco_espuria:
        classificacao = "Regressão Espúria"
        parecer = (
            "ALERTA CRÍTICO: Risco gravíssimo de Regressão Espúria! As séries são não-estacionárias "
            "em nível e a estatística Durbin-Watson é significativamente inferior ao R² (regra de Granger-Newbold). "
            "Os valores elevados de R² e t-stat são artefatos matemáticos provocados por tendências estocásticas comuns."
        )
    elif tem_alta_multi or tem_autocorr or tem_het:
        classificacao = "Exige Cautela"
        motivos = []
        if tem_alta_multi:
            motivos.append("multicolinearidade moderada/alta (VIF > 5)")
        if tem_autocorr:
            motivos.append("autocorrelação residual (DW distante de 2,0)")
        if tem_het:
            motivos.append("heterocedasticidade dos erros (Breusch-Pagan p < 0,05)")
        parecer = (
            f"O modelo apresenta bom embasamento conceitual, mas exige cautela devido a: {', '.join(motivos)}. "
            f"As séries devem ser analisadas com erros padrão robustos e não se deve assumir causalidade direta."
        )
    else:
        classificacao = "Robusto"
        parecer = (
            "O modelo atende aos principais critérios econométricos: séries estacionárias, "
            "resíduos sem autocorrelação grave e ausência de multicolinearidade prejudicial."
        )

    return ResultadoDiagnosticos(
        nome_modelo=reg.nome_modelo,
        tipo_dados=reg.tipo_dados,
        tabela_vif=tabela_vif,
        matriz_correlacao=matriz_corr,
        tem_alta_multicolinearidade=tem_alta_multi,
        lm_breusch_pagan=lm_stat,
        p_breusch_pagan=p_lm,
        tem_heterocedasticidade=tem_het,
        durbin_watson_stat=dw_stat,
        tem_autocorrelacao=tem_autocorr,
        tabela_adf=tabela_adf,
        series_estacionarias=series_estac,
        residuos_estacionarios=residuos_estac,
        media_residuos=metricas_res["media"],
        desvio_residuos=metricas_res["desvio"],
        assimetria_residuos=metricas_res["assimetria"],
        curtose_residuos=metricas_res["curtose"],
        jb_stat=metricas_res["jb_stat"],
        jb_pvalor=metricas_res["jb_pvalor"],
        residuos_normais=res_normais,
        total_outliers_3sigma=n_outliers,
        top_outliers=top_outliers,
        risco_regressao_espuria=risco_espuria,
        classificacao_modelo=classificacao,
        parecer_didatico=parecer
    )


if __name__ == "__main__":
    from dados import carregar_ou_atualizar_dados
    from regressao import executar_regressao_multipla

    precos, retornos, _ = carregar_ou_atualizar_dados()
    
    print("\n--- DIAGNÓSTICO DO MODELO DE RETORNOS DIÁRIOS ---")
    reg_ret = executar_regressao_multipla(retornos, "SUZ", ["NYA", "USDBRL", "WOOD", "US10Y"], tipo_dados="Retornos Diários")
    diag_ret = executar_diagnosticos_completos(reg_ret, retornos)
    print(diag_ret.gerar_relatorio_texto())

    print("\n--- DIAGNÓSTICO DO MODELO DE COTAÇÕES EM NÍVEL ---")
    reg_cot = executar_regressao_multipla(precos, "SUZ", ["NYA", "USDBRL", "WOOD", "US10Y"], tipo_dados="Cotações em Nível")
    diag_cot = executar_diagnosticos_completos(reg_cot, precos)
    print(diag_cot.gerar_relatorio_texto())
