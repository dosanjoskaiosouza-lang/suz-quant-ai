"""
===============================================================================
SUZ Quant AI - Módulo de Regressão Linear por OLS (regressao.py)
Disciplina: Mercado Financeiro e de Capitais - PUC-SP
Empresa: Suzano S.A. | Ativo: SUZ (ADR na NYSE)
===============================================================================
Este módulo executa regressões lineares simples e múltiplas por Mínimos Quadrados
Ordinários (OLS / MQO), calculando todos os estimadores estatísticos fundamentais
e gerando saídas estruturadas com explicação didática para estudantes de Administração.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class ResultadoRegressao:
    """
    Estrutura que armazena os resultados de uma regressão linear OLS.
    
    Conceitos didáticos:
    - R²: Coeficiente de determinação (fração da variação de Y explicada pelos X).
    - R² ajustado: Penaliza a inclusão de variáveis desnecessárias.
    - Estatística F e Prob(F): Testa se o conjunto de variáveis X é relevante.
    - Beta (β): Sensibilidade de Y em relação a cada X.
    - p-valor (P>|t|): Probabilidade de o efeito ser mero acaso. Se < 0,05 (5%), é significante.
    """
    nome_modelo: str
    variavel_y: str
    variaveis_x: List[str]
    tipo_dados: str  # 'Retornos Diários' ou 'Cotações em Nível'
    n_observacoes: int
    r2: float
    r2_ajustado: float
    f_estatistica: float
    f_pvalor: float
    tabela_coeficientes: pd.DataFrame
    residuos: pd.Series
    valores_ajustados: pd.Series
    modelo_sm: Any  # Objeto statsmodels OLS Results
    todos_x_significativos_5pct: bool
    atende_criterio_busca: bool  # R² ajustado > 50% E p-valor < 5% em todos os X

    def gerar_equacao_formatada(self) -> str:
        """Gera a fórmula matemática estimada com os coeficientes numéricos."""
        linhas = []
        intercepto = self.tabela_coeficientes.loc["const", "coeficiente"] if "const" in self.tabela_coeficientes.index else 0.0
        sinal = "+" if intercepto >= 0 else "-"
        partes = [f"{sinal} {abs(intercepto):.6f}"]

        for x in self.variaveis_x:
            if x in self.tabela_coeficientes.index:
                coef = self.tabela_coeficientes.loc[x, "coeficiente"]
                sinal_x = "+" if coef >= 0 else "-"
                partes.append(f"{sinal_x} {abs(coef):.6f}*{x}")

        equacao = f"{self.variavel_y}_t = " + " ".join(partes)
        return equacao

    def explicacao_didatica(self) -> str:
        """Produz explicação em português simples do modelo para trabalho acadêmico."""
        equacao = self.gerar_equacao_formatada()
        r2_pct = self.r2 * 100
        r2_adj_pct = self.r2_ajustado * 100

        linhas = [
            f"=== INTERPRETAÇÃO DIDÁTICA: {self.nome_modelo} ===",
            f"Tipo de dados analisados: {self.tipo_dados}",
            f"Número de pregões considerados (N): {self.n_observacoes:,} dias.",
            f"Equação Estimada: {equacao}",
            f"Poder Explicativo (R²): {r2_pct:.2f}% dos movimentos de {self.variavel_y} são explicados pelos fatores incluídos.",
            f"R² Ajustado: {r2_adj_pct:.2f}% (ajuste que penaliza o excesso de variáveis).",
            f"Teste F Global: Estatística F = {self.f_estatistica:.2f} (p-valor = {self.f_pvalor:.2e}). "
            + ("O conjunto de variáveis é estatisticamente significante em conjunto." if self.f_pvalor < 0.05 else "O conjunto não tem relevância estatística global."),
            "\nDetalhamento dos Coeficientes Individuais:"
        ]

        for idx, row in self.tabela_coeficientes.iterrows():
            nome_var = idx
            coef = row["coeficiente"]
            pval = row["p_valor"]
            sig = "Significante ao nível de 5% (p < 0,05)" if pval < 0.05 else "NÃO significante estatisticamente (p >= 0,05)"

            if nome_var == "const":
                linhas.append(
                    f"  • Intercepto (Alfa) = {coef:.6f} (p-valor = {pval:.4f}) -> {sig}. "
                    f"Representa o retorno médio esperado quando todos os fatores explicativos são zero."
                )
            else:
                impacto = "aumento" if coef > 0 else "redução"
                linhas.append(
                    f"  • Beta de {nome_var} = {coef:.4f} (p-valor = {pval:.4f}) -> {sig}. "
                    f"Para cada variação de 1,00 ponto percentual em {nome_var}, espera-se em média um {impacto} "
                    f"de {abs(coef):.4f} ponto percentual em {self.variavel_y}, mantidos os demais fatores constantes."
                )

        if self.atende_criterio_busca:
            linhas.append(
                "\n[CRITÉRIO ATENDIDO] Este modelo superou R² ajustado > 50% E possui p-valor < 5% para todas as variáveis explicativas!"
            )
        else:
            linhas.append(
                "\n[CRITÉRIO NÃO ATENDIDO] Este modelo NÃO preencheu simultaneamente R² ajustado > 50% e p-valor < 5% em todos os fatores."
            )

        return "\n".join(linhas)


def ajustar_ols(
    y: pd.Series,
    X: pd.DataFrame,
    nome_modelo: str = "Regressão OLS",
    tipo_dados: str = "Retornos Diários"
) -> ResultadoRegressao:
    """
    Ajusta uma regressão linear por OLS (Ordinary Least Squares) utilizando statsmodels.
    
    Calcula:
    - Intercepto (const)
    - Coeficientes beta
    - Erro-padrão (std err)
    - Estatística t (t-stat)
    - p-valor (P>|t|)
    - R² e R² ajustado
    - Estatística F e Prob(F)
    - Intervalo de Confiança de 95%
    """
    # Alinhamento das séries
    df_conjunto = pd.concat([y.rename("Y"), X], axis=1).dropna()
    y_alinhado = df_conjunto["Y"]
    X_alinhado = df_conjunto.drop(columns=["Y"])

    # Adiciona a constante (alfa)
    X_com_constante = sm.add_constant(X_alinhado, has_constant="add")

    # Ajusta o modelo OLS
    modelo_ols = sm.OLS(y_alinhado, X_com_constante).fit()

    # Monta a tabela detalhada de coeficientes
    conf_int = modelo_ols.conf_int(alpha=0.05)
    tabela_coef = pd.DataFrame({
        "variavel": modelo_ols.params.index,
        "coeficiente": modelo_ols.params.values,
        "erro_padrao": modelo_ols.bse.values,
        "estatistica_t": modelo_ols.tvalues.values,
        "p_valor": modelo_ols.pvalues.values,
        "ic95_inferior": conf_int[0].values,
        "ic95_superior": conf_int[1].values,
    }).set_index("variavel")

    tabela_coef["significativo_5pct"] = tabela_coef["p_valor"] < 0.05

    # Avaliação dos critérios de busca do trabalho
    # Verifica apenas as variáveis explicativas X (excluindo a constante const)
    p_valores_x = [modelo_ols.pvalues[col] for col in X_alinhado.columns if col in modelo_ols.pvalues]
    todos_x_sig = all(p < 0.05 for p in p_valores_x) if p_valores_x else False
    criterio_atendido = bool((modelo_ols.rsquared_adj > 0.50) and todos_x_sig)

    resultado = ResultadoRegressao(
        nome_modelo=nome_modelo,
        variavel_y=str(y.name),
        variaveis_x=list(X_alinhado.columns),
        tipo_dados=tipo_dados,
        n_observacoes=int(modelo_ols.nobs),
        r2=float(modelo_ols.rsquared),
        r2_ajustado=float(modelo_ols.rsquared_adj),
        f_estatistica=float(modelo_ols.fvalue) if not np.isnan(modelo_ols.fvalue) else 0.0,
        f_pvalor=float(modelo_ols.f_pvalue) if not np.isnan(modelo_ols.f_pvalue) else 1.0,
        tabela_coeficientes=tabela_coef,
        residuos=modelo_ols.resid,
        valores_ajustados=modelo_ols.fittedvalues,
        modelo_sm=modelo_ols,
        todos_x_significativos_5pct=todos_x_sig,
        atende_criterio_busca=criterio_atendido
    )

    return resultado


def executar_regressao_simples(
    df: pd.DataFrame,
    col_y: str = "SUZ",
    col_x: str = "NYA",
    tipo_dados: str = "Retornos Diários"
) -> ResultadoRegressao:
    """
    Executa obrigatoriamente a Regressão Simples:
    R_SUZ,t = α + β₁·R_NYA,t + ε_t
    """
    if col_y not in df.columns or col_x not in df.columns:
        raise KeyError(f"Colunas {col_y} ou {col_x} não encontradas no DataFrame.")

    y = df[col_y]
    X = df[[col_x]]
    nome = f"Regressão Simples: {col_y} ~ {col_x}"
    return ajustar_ols(y, X, nome_modelo=nome, tipo_dados=tipo_dados)


def executar_regressao_multipla(
    df: pd.DataFrame,
    col_y: str = "SUZ",
    cols_x: Optional[List[str]] = None,
    tipo_dados: str = "Retornos Diários",
    nome_modelo: str = ""
) -> ResultadoRegressao:
    """
    Executa a Regressão Múltipla com as variáveis especificadas (máximo de 4 X no total).
    R_SUZ,t = α + β₁X₁,t + β₂X₂,t + ... + ε_t
    """
    if cols_x is None:
        cols_x = ["NYA", "USDBRL", "WOOD", "US10Y"]

    if len(cols_x) > 4:
        raise ValueError(
            f"Limite metodológico excedido: máximo de 4 variáveis X permitido pelo projeto. "
            f"Foram passadas {len(cols_x)}."
        )

    if "NYA" not in cols_x:
        raise ValueError("A variável X1 = NYA é obrigatória em todas as regressões.")

    for c in [col_y] + cols_x:
        if c not in df.columns:
            raise KeyError(f"Variável '{c}' não existe na base de dados.")

    y = df[col_y]
    X = df[cols_x]
    if not nome_modelo:
        nome_modelo = f"Regressão Múltipla: {col_y} ~ {' + '.join(cols_x)}"

    return ajustar_ols(y, X, nome_modelo=nome_modelo, tipo_dados=tipo_dados)


if __name__ == "__main__":
    from dados import carregar_ou_atualizar_dados
    precos, retornos, _ = carregar_ou_atualizar_dados()
    
    print("\n--- TESTANDO REGRESSÃO SIMPLES ---")
    reg_simples = executar_regressao_simples(retornos, "SUZ", "NYA")
    print(reg_simples.explicacao_didatica())

    print("\n--- TESTANDO REGRESSÃO MÚLTIPLA COMPLETA ---")
    reg_multi = executar_regressao_multipla(retornos, "SUZ", ["NYA", "USDBRL", "WOOD", "US10Y"])
    print(reg_multi.explicacao_didatica())
