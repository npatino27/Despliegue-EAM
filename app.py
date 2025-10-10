import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================================
# 🔹 Cargar modelo y transformadores
# ==========================================================
model = joblib.load("best_random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏠 Predicción del Valor de Casas (SalePrice) - Iowa")
st.markdown("Esta aplicación utiliza un modelo **Random Forest Regressor** para estimar el valor de una vivienda en Ames, Iowa.")

# ==========================================================
# 🧾 Entradas del usuario
# ==========================================================
st.header("Ingrese las características de la vivienda:")

OverallQual = st.slider("Calidad general (1-10)", 1, 10, 7)
GrLivArea = st.number_input("Área habitable sobre el suelo (GrLivArea)", value=1800)
GarageCars = st.slider("Capacidad del garaje (GarageCars)", 0, 4, 2)
TotalBsmtSF = st.number_input("Área total del sótano (TotalBsmtSF)", value=900)
YearBuilt = st.number_input("Año de construcción (YearBuilt)", value=2005)
FullBath = st.slider("Número de baños completos (FullBath)", 0, 4, 2)
LotArea = st.number_input("Área del lote (LotArea)", value=8000)
FirstFlrSF = st.number_input("Área del primer piso (1stFlrSF)", value=1200)
Fireplaces = st.slider("Número de chimeneas (Fireplaces)", 0, 3, 1)
TotRmsAbvGrd = st.slider("Habitaciones sobre el nivel del suelo (TotRmsAbvGrd)", 2, 12, 7)

# ==========================================================
# 🧮 Preprocesamiento de entrada
# ==========================================================
nuevos_datos = pd.DataFrame([{
    "OverallQual": OverallQual,
    "GrLivArea": GrLivArea,
    "GarageCars": GarageCars,
    "TotalBsmtSF": TotalBsmtSF,
    "YearBuilt": YearBuilt,
    "FullBath": FullBath,
    "LotArea": LotArea,
    "1stFlrSF": FirstFlrSF,
    "Fireplaces": Fireplaces,
    "TotRmsAbvGrd": TotRmsAbvGrd
}])

# Cargar columnas originales del modelo
encoded_columns = joblib.load("encoded_columns.pkl")

# Crear DataFrame con todas las columnas esperadas por el scaler
# (si falta alguna, se llena con 0)
for col in encoded_columns:
    if col not in nuevos_datos.columns:
        nuevos_datos[col] = 0

# Reordenar las columnas exactamente como el scaler las espera
nuevos_datos = nuevos_datos[encoded_columns]

# Escalar los datos
nuevos_datos_scaled = scaler.transform(nuevos_datos)

# ==========================================================
# 🔮 Predicción
# ==========================================================
if st.button("Predecir Precio de Venta"):
    pred = model.predict(nuevos_datos_scaled)
    st.success(f"💰 Precio estimado de la vivienda (SalePrice): ${pred[0]:,.2f}")
