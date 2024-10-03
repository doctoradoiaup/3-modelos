# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:02:31 2024

@author: jperezr
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
import riskfolio as rp
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Función para descargar los datos de Yahoo Finance
def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    return data['Adj Close']

# Cargar los datos
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "XOM", "CVX", "JPM", "BAC", "WFC"]  # Ejemplo con empresas de diferentes sectores
start_date = '2018-01-01'
end_date = '2022-12-31'

st.title('Optimización de Portafolio Comparativa: MVP, HRP, y Autoencoder')
data = get_data(tickers, start_date, end_date)
st.write("Datos de precios históricos:", data.head())

# Cálculo de retornos y matriz de covarianza
returns = data.pct_change().dropna()
mean_returns = mean_historical_return(data)
cov_matrix = CovarianceShrinkage(returns).ledoit_wolf()

# --- Modelo MVP (Media-Varianza) ---
st.subheader("Optimización de Media-Varianza (MVP)")
ef = EfficientFrontier(mean_returns, cov_matrix)
weights_mvp = ef.max_sharpe()
performance_mvp = ef.portfolio_performance()
st.write(f"Pesos MVP: {weights_mvp}")
st.write(f"Desempeño MVP: {performance_mvp}")

# --- Modelo HRP usando riskfolio-lib ---
st.subheader("Optimización de Paridad de Riesgo Jerárquica (HRP) con riskfolio-lib")

# Crear el portafolio y calcular estadísticas
port = rp.Portfolio(returns=returns)
method_mu = 'hist'  # Usar media histórica
method_cov = 'hist'  # Usar covarianza histórica
port.assets_stats(method_mu=method_mu, method_cov=method_cov)

# Optimización HRP
weights_hrp = port.optimization(model='HRP', codependence='pearson', rm='MV', rf=0, linkage='ward')
performance_hrp = port.performance(weights_hrp, rm=method_cov, rf=0, alpha=0.05)

st.write(f"Pesos HRP: {weights_hrp}")
st.write(f"Desempeño HRP: {performance_hrp}")

# --- Modelo Autoencoder ---
st.subheader("Optimización basada en Autoencoder")
def build_autoencoder(input_shape):
    input_layer = Input(shape=(input_shape,))
    encoder = Dense(5, activation="relu")(input_layer)
    decoder = Dense(input_shape, activation="linear")(encoder)
    autoencoder = Model(input_layer, decoder)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder

# Construir y entrenar el autoencoder
autoencoder = build_autoencoder(returns.shape[1])
autoencoder.fit(returns.values, returns.values, epochs=100, batch_size=10, verbose=0)

# Obtener los pesos del autoencoder
encoded_weights = autoencoder.predict(returns.mean().values.reshape(1, -1)).flatten()
encoded_weights = encoded_weights / np.sum(encoded_weights)  # Normalizamos los pesos
performance_enc = np.dot(encoded_weights, mean_returns.values), np.sqrt(np.dot(encoded_weights.T, np.dot(cov_matrix, encoded_weights)))

st.write(f"Pesos Autoencoder: {encoded_weights}")
st.write(f"Desempeño Autoencoder: {performance_enc}")

# --- Comparación de Resultados ---
st.subheader("Comparación de Resultados")
st.write("Desempeño MVP:", performance_mvp)
st.write("Desempeño HRP:", performance_hrp)
st.write("Desempeño Autoencoder:", performance_enc)

# Gráfica comparativa de los portafolios
performance_data = {
    'Modelo': ['MVP', 'HRP', 'Autoencoder'],
    'Retorno': [performance_mvp[0], performance_hrp[0], performance_enc[0]],
    'Volatilidad': [performance_mvp[1], performance_hrp[1], performance_enc[1]]
}
df_performance = pd.DataFrame(performance_data)
st.write(sns.barplot(x='Modelo', y='Retorno', data=df_performance))
plt.title('Comparación de Retornos')
st.pyplot()