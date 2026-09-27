import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

print("=== SISTEMA DE PREVISÃO DE DEMANDA DE MEDICAMENTOS NO SUS ===")
print("Objetivo: Otimizar o abastecimento e evitar desabastecimento na rede municipal.\n")

# 1. Simulando um histórico de consumo mensal (em unidades) de um medicamento de alto fluxo (ex: Insulina/Losartana)
dados = {
    'Mes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Consumo_Real': [1200, 1250, 1180, 1300, 1350, 1400, 1380, 1450, 1500, 1520, 1490, 1580]
}

df = pd.DataFrame(dados)

# 2. Estatística Básica Descritiva
media_consumo = df['Consumo_Real'].mean()
desvio_padrao = df['Consumo_Real'].std()

print(f"--- ANÁLISE ESTATÍSTICA BÁSICA ---")
print(f"Média de consumo mensal: {media_consumo:.2f} unidades")
print(f"Desvio padrão (volatilidade): {desvio_padrao:.2f}\n")

# 3. Preparando os dados para Machine Learning (Regressão Linear)
# X = Mês, y = Consumo Real
X = df[['Mes']]
y = df['Consumo_Real']

# Treinando o modelo preditivo
modelo = LinearRegression()
modelo.fit(X, y)

# 4. Predição para o próximo mês (Mês 13)
proximo_mes = pd.DataFrame({'Mes': [13]})
previsao = modelo.predict(proximo_mes)

print(f"--- MODELO PREDITIVO (MACHINE LEARNING) ---")
print(f"Previsão de demanda estimada para o próximo mês (Mês 13): {previsao[0]:.0f} unidades")
print("Recomendação logística: Ajustar o pedido de compra ao fornecedor considerando margem de segurança de estoque.")
