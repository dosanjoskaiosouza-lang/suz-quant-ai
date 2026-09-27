"""
===============================================================================
SUZ Quant AI - Módulo de Comparação Sistemática de Modelos (modelos.py)
Disciplina: Mercado Financeiro e de Capitais - PUC-SP
Empresa: Suzano S.A. | Ativo: SUZ (ADR na NYSE)
===============================================================================
Este módulo testa de forma exaustiva e auditável todas as 8 combinações
metodologicamente válidas de variáveis explicativas, sempre mantendo X1 = NYA
como fator obrigatório de mercado:
- M1: NYA
- M2: NYA + X2
- M3: NYA + X3
- M4: NYA + X4
- M5: NYA + X2 + X3
- M6: NYA + X2 + X4
- M7: NYA + X3 + X4
- M8: NYA + X2 + X3 + X4

Gera e salva para fins de auditoria:
- outputs/modelos_testados.csv
- outputs/coeficientes.csv
"""

from typing import List, Dict, Tuple, Optional, Any
import os
import pandas as pd
import numpy as np

from regressao import ResultadoRegressao, executar_regressao_multipla, executar_regressao_simples
from diagnosticos import executar_diagnosticos_completos, ResultadoDiagnosticos


# Lista padronizada das 8 especificações metodológicas
ESPECIFICACOES_PADRAO = [
    ("M1_NYA", ["NYA"]),
    ("M2_NYA_USDBRL", ["NYA", "USDBRL"]),
    ("M3_NYA_WOOD", ["NYA", "WOOD"]),
    ("M4_NYA_US10Y", ["NYA", "US10Y"]),
    ("M5_NYA_USDBRL_WOOD", ["NYA", "USDBRL", "WOOD"]),
    ("M6_NYA_USDBRL_US10Y", ["NYA", "USDBRL", "US10Y"]),
    ("M7_NYA_WOOD_US10Y", ["NYA", "WOOD", "US10Y"]),
    ("M8_COMPLETO", ["NYA", "USDBRL", "WOOD", "US10Y"])
]


def gerar_especificacoes_customizadas(
    x1: str = "NYA",
    x2: str = "USDBRL",
    x3: str = "WOOD",
    x4: str = "US10Y"
) -> List[Tuple[str, List[str]]]:
    """Permite alterar dinamicamente os nomes dos fatores X2, X3, X4 mantendo X1."""
    return [
        (f"M1_{x1}", [x1]),
        (f"M2_{x1}_{x2}", [x1, x2]),
        (f"M3_{x1}_{x3}", [x1, x3]),
        (f"M4_{x1}_{x4}", [x1, x4]),
        (f"M5_{x1}_{x2}_{x3}", [x1, x2, x3]),
        (f"M6_{x1}_{x2}_{x4}", [x1, x2, x4]),
        (f"M7_{x1}_{x3}_{x4}", [x1, x3, x4]),
        (f"M8_{x1}_{x2}_{x3}_{x4}", [x1, x2, x3, x4])
    ]


def testar_todos_os_modelos(
    df: pd.DataFrame,
    col_y: str = "SUZ",
    especificacoes: Optional[List[Tuple[str, List[str]]]] = None,
    tipo_dados: str = "Retornos Diários"
) -> Tuple[Dict[str, ResultadoRegressao], Dict[str, ResultadoDiagnosticos]]:
    """
    Executa sistematicamente todas as especificações e roda diagnósticos para cada uma.
    """
    if especificacoes is None:
        especificacoes = ESPECIFICACOES_PADRAO

    resultados_reg = {}
    resultados_diag = {}

    for codigo_modelo, variaveis_x in especificacoes:
        # Validação da obrigatoriedade do NYA
        if "NYA" not in variaveis_x:
            raise ValueError(f"O modelo {codigo_modelo} viola a regra de conter X1 = NYA obrigatoriamente.")

        # Validação do limite máximo de 4 variáveis X
        if len(variaveis_x) > 4:
            raise ValueError(f"O modelo {codigo_modelo} excede o limite máximo de 4 variáveis X ({len(variaveis_x)}).")

        nome_amigavel = f"{codigo_modelo}: {col_y} ~ {' + '.join(variaveis_x)}"
        reg = executar_regressao_multipla(
            df=df,
            col_y=col_y,
            cols_x=variaveis_x,
            tipo_dados=tipo_dados,
            nome_modelo=nome_amigavel
        )
        diag = executar_diagnosticos_completos(reg, df)

        resultados_reg[codigo_modelo] = reg
        resultados_diag[codigo_modelo] = diag

    return resultados_reg, resultados_diag


def construir_tabela_resumo(
    resultados_reg: Dict[str, ResultadoRegressao],
    resultados_diag: Dict[str, ResultadoDiagnosticos]
) -> pd.DataFrame:
    """
    Constrói a tabela resumo exigida para a apresentação final:
    Modelo | Variáveis X | N | R² | R² ajustado | maior p-value entre X | diagnóstico | critério atendido?
    """
    linhas = []
    for cod, reg in resultados_reg.items():
        diag = resultados_diag[cod]
        
        # Maior p-valor entre as variáveis X (ignorando const)
        p_vals_x = [
            reg.tabela_coeficientes.loc[var, "p_valor"]
            for var in reg.variaveis_x
            if var in reg.tabela_coeficientes.index
        ]
        maior_p = max(p_vals_x) if p_vals_x else 1.0

        linhas.append({
            "Modelo": cod,
            "Variáveis X": " + ".join(reg.variaveis_x),
            "N": reg.n_observacoes,
            "R²": f"{reg.r2 * 100:.2f}%",
            "R² ajustado": f"{reg.r2_ajustado * 100:.2f}%",
            "R2_Adj_Num": reg.r2_ajustado,
            "maior p-value entre X": f"{maior_p:.4f}",
            "Maior_P_Num": maior_p,
            "diagnóstico": diag.classificacao_modelo,
            "critério atendido?": "SIM" if reg.atende_criterio_busca else "NÃO"
        })

    df_resumo = pd.DataFrame(linhas)
    return df_resumo


def salvar_auditoria_csv(
    resultados_reg_retornos: Dict[str, ResultadoRegressao],
    resultados_reg_cotacoes: Optional[Dict[str, ResultadoRegressao]] = None,
    pasta_destino: str = "outputs"
) -> Tuple[str, str]:
    """
    Salva os dois arquivos de auditoria formal obrigatórios:
    1. outputs/modelos_testados.csv:
       modelo | Y | Xs | período | N | R² | R² ajustado | F | p-value F | critérios atendidos
    2. outputs/coeficientes.csv:
       modelo | variável | coeficiente | erro padrão | t | p-value | IC95 inferior | IC95 superior
    """
    os.makedirs(pasta_destino, exist_ok=True)
    caminho_modelos = os.path.join(pasta_destino, "modelos_testados.csv")
    caminho_coeficientes = os.path.join(pasta_destino, "coeficientes.csv")

    linhas_modelos = []
    linhas_coeficientes = []

    todos_modelos = list(resultados_reg_retornos.items())
    if resultados_reg_cotacoes:
        todos_modelos.extend([
            (f"{k}_COTACOES", v) for k, v in resultados_reg_cotacoes.items()
        ])

    for cod, reg in todos_modelos:
        # Linha para modelos_testados.csv
        p_inicial = reg.residuos.index.min().strftime("%Y-%m-%d") if hasattr(reg.residuos.index, "min") else "N/A"
        p_final = reg.residuos.index.max().strftime("%Y-%m-%d") if hasattr(reg.residuos.index, "max") else "N/A"
        periodo_str = f"{p_inicial} a {p_final}"

        linhas_modelos.append({
            "modelo": cod,
            "tipo_dados": reg.tipo_dados,
            "Y": reg.variavel_y,
            "Xs": " + ".join(reg.variaveis_x),
            "período": periodo_str,
            "N": reg.n_observacoes,
            "R²": f"{reg.r2:.6f}",
            "R² ajustado": f"{reg.r2_ajustado:.6f}",
            "F": f"{reg.f_estatistica:.4f}",
            "p-value F": f"{reg.f_pvalor:.6e}",
            "critérios atendidos": "SIM" if reg.atende_criterio_busca else "NÃO"
        })

        # Linhas para coeficientes.csv
        for var_nome, row in reg.tabela_coeficientes.iterrows():
            linhas_coeficientes.append({
                "modelo": cod,
                "tipo_dados": reg.tipo_dados,
                "variável": var_nome,
                "coeficiente": f"{row['coeficiente']:.6f}",
                "erro padrão": f"{row['erro_padrao']:.6f}",
                "t": f"{row['estatistica_t']:.4f}",
                "p-value": f"{row['p_valor']:.6f}",
                "IC95 inferior": f"{row['ic95_inferior']:.6f}",
                "IC95 superior": f"{row['ic95_superior']:.6f}"
            })

    df_mod = pd.DataFrame(linhas_modelos)
    df_mod.to_csv(caminho_modelos, index=False, encoding="utf-8-sig")

    df_coef = pd.DataFrame(linhas_coeficientes)
    df_coef.to_csv(caminho_coeficientes, index=False, encoding="utf-8-sig")

    return caminho_modelos, caminho_coeficientes


def avaliar_criterio_busca_global(resultados_reg: Dict[str, ResultadoRegressao]) -> Tuple[bool, str]:
    """
    Verifica se alguma especificação atingiu simultaneamente R² adj > 50% e p-val < 5% em todos os X.
    Gera a declaração formal exigida no item 7.
    """
    atendidos = [cod for cod, reg in resultados_reg.items() if reg.atende_criterio_busca]
    if atendidos:
        msg = f"As seguintes especificações atingiram ambos os critérios de busca: {', '.join(atendidos)}."
        return True, msg
    else:
        msg = "Nenhuma especificação testada atingiu simultaneamente R² ajustado superior a 50% e significância de 5% para todas as variáveis explicativas."
        return False, msg


if __name__ == "__main__":
    from dados import carregar_ou_atualizar_dados
    precos, retornos, stats = carregar_ou_atualizar_dados()

    print("\n=== TESTANDO 8 MODELOS DE RETORNOS DIÁRIOS ===")
    regs_ret, diags_ret = testar_todos_os_modelos(retornos, "SUZ", tipo_dados="Retornos Diários")
    
    print("\n=== TESTANDO 8 MODELOS DE COTAÇÕES EM NÍVEL ===")
    regs_cot, diags_cot = testar_todos_os_modelos(precos, "SUZ", tipo_dados="Cotações em Nível")

    # Salva CSVs de auditoria
    f_mod, f_coef = salvar_auditoria_csv(regs_ret, regs_cot)
    print(f"\n[OK] Auditoria salva:\n - {f_mod}\n - {f_coef}")

    # Tabela resumo
    tabela = construir_tabela_resumo(regs_ret, diags_ret)
    print("\nTABELA RESUMO DE RETORNOS DIÁRIOS:")
    cols_mostrar = ["Modelo", "Variáveis X", "N", "R²", "R² ajustado", "maior p-value entre X", "diagnóstico", "critério atendido?"]
    print(tabela[cols_mostrar].to_string(index=False))

    # Veredito do critério de busca
    sucesso, veredito = avaliar_criterio_busca_global(regs_ret)
    print(f"\nVeredito do Critério de Busca:\n'{veredito}'")
