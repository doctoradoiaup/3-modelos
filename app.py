# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:53:42 2024

@author: jperezr
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cvxpy as cp

# 1. Cargar los datos de los 10 tickers
tickers = ['RELIANCE.NS', 'ULTRACEMCO.NS', 'TATASTEEL.NS', 'NTPC.NS', 'JSWSTEEL.NS',
           'ONGC.NS', 'GRASIM.NS', 'HINDALCO.NS', 'COALINDIA.NS', 'UPL.NS']

start_train = "2018-01-01"
end_train = "2021-12-31"
start_test = "2022-01-01"
end_test = "2022-12-31"

# Obtener los datos históricos
st.write("Cargando datos de Yahoo Finance...")
data_train = yf.download(tickers, start=start_train, end=end_train)['Adj Close']
data_test = yf.download(tickers, start=start_test, end=end_test)['Adj Close']

st.write("Datos de entrenamiento")
st.dataframe(data_train)

st.write("Datos de prueba")
st.dataframe(data_test)

# Cálculo de rendimientos utilizando precios ajustados
returns_train = data_train.pct_change().dropna()
returns_test = data_test.pct_change().dropna()

# Ponderaciones iniciales del índice sectorial del sector de materias primas
initial_weights = {
    'RELIANCE.NS': 10.13,
    'ULTRACEMCO.NS': 7.52,
    'TATASTEEL.NS': 7.52,
    'NTPC.NS': 7.26,
    'JSWSTEEL.NS': 5.64,
    'ONGC.NS': 5.32,
    'GRASIM.NS': 5.31,
    'HINDALCO.NS': 5.23,
    'COALINDIA.NS': 4.05,
    'UPL.NS': 3.32
}

# Normalizar las ponderaciones iniciales a porcentaje (100%)
initial_weights_normalized = {k: v / 100 for k, v in initial_weights.items()}

# 2. Implementar el Modelo MVP
st.write("**Modelo MVP (Mean-Variance Portfolio)**")
mu = expected_returns.mean_historical_return(data_train).values
S = risk_models.sample_cov(data_train)

# Definir las variables de decisión
weights = cp.Variable(len(tickers))

# Definir la función objetivo: minimizar la varianza del portafolio
risk = cp.quad_form(weights, S)
objective = cp.Minimize(risk)

# Agregar restricciones
constraints = [
    cp.sum(weights) == 1,  # La suma de los pesos debe ser 1
    weights >= 0            # No se permiten posiciones cortas
]

# Problema de optimización
problem = cp.Problem(objective, constraints)

# Resolver el problema
try:
    problem.solve()
except Exception as e:
    st.error(f"Ocurrió un error al resolver el problema: {e}")
    st.stop()

# Verificar si la solución es válida
if weights.value is None:
    st.error("No se pudo encontrar una solución válida para el modelo MVP.")
else:
    # Obtener los pesos optimizados
    cleaned_weights_mvp = {tickers[i]: weights.value[i] for i in range(len(tickers))}
    st.write("Pesos MVP:", cleaned_weights_mvp)

# 3. Implementar el Modelo HRP
st.write("**Modelo HRP (Hierarchical Risk Parity)**")
cov_matrix = returns_train.cov()

# Enlace jerárquico y creación de clusters
corr_matrix = returns_train.corr()
dist = 1 - corr_matrix
Z = linkage(dist, 'ward')
cluster_labels = fcluster(Z, 10, criterion='maxclust')

# Asignación de pesos HRP (inverso de la varianza) respetando las ponderaciones iniciales
hrp_weights = {}
for i in np.unique(cluster_labels):
    cluster_indices = np.where(cluster_labels == i)[0]
    cluster_cov = cov_matrix.iloc[cluster_indices, cluster_indices]
    inv_var_weights = 1 / np.diag(cluster_cov)
    inv_var_weights /= inv_var_weights.sum()
    for j, idx in enumerate(cluster_indices):
        hrp_weights[data_train.columns[idx]] = inv_var_weights[j] * initial_weights_normalized[data_train.columns[idx]]

st.write("Pesos HRP:", hrp_weights)

# 4. Implementar el Modelo ENC (Autoencoder)
st.write("**Modelo ENC (Autoencoder-based Portfolio)**")
# Usar PCA como sustituto del autoencoder para simplificar
pca = PCA(n_components=5)
pca.fit(returns_train)
pca_weights = np.abs(pca.components_).sum(axis=0)
pca_weights /= pca_weights.sum()

enc_weights = {ticker: weight * initial_weights_normalized[ticker] for ticker, weight in zip(tickers, pca_weights)}
st.write("Pesos ENC:", enc_weights)

# 5. Mostrar la tabla comparativa de los 3 modelos
st.write("**Comparación de Pesos**")
weights_df = pd.DataFrame({
    'Tickers': tickers,
    'Pesos MVP': [cleaned_weights_mvp.get(ticker, 0) if weights.value is not None else 0 for ticker in tickers],
    'Pesos HRP': [hrp_weights.get(ticker, 0) for ticker in tickers],
    'Pesos ENC': [enc_weights.get(ticker, 0) for ticker in tickers]
})

st.write(weights_df)

# Gráfico de las ponderaciones
st.write("**Gráfico de Ponderaciones**")
weights_df.set_index('Tickers', inplace=True)
weights_df.plot(kind='bar', figsize=(10, 6))
st.pyplot(plt)