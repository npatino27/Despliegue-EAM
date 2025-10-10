import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================================
# Cargar modelo
# ==========================================================
model = joblib.load("best_random_forest_model.pkl")

st.title("🏠 Predicción del Valor de Casas (SalePrice) - Modelo Simplificado")
st.markdown("Esta aplicación utiliza un modelo **Random Forest** entrenado con 10 variables numéricas clave para estimar el precio de una vivienda en Iowa.")

# ==========================================================
# Entradas del usuario
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
# Crear DataFrame con nombres EXACTOS del entrenamiento
# ==========================================================
nuevos_datos = pd.DataFrame([{
    "OverallQual": OverallQual,
    "GrLivArea": GrLivArea,
    "GarageCars": GarageCars,
    "TotalBsmtSF": TotalBsmtSF,
    "YearBuilt": YearBuilt,
    "FullBath": FullBath,
    "LotArea": LotArea,
    "1stFlrSF": FirstFlrSF,   # 👈 nombre exacto
    "Fireplaces": Fireplaces,
    "TotRmsAbvGrd": TotRmsAbvGrd
}])

# ==========================================================
# Predicción
# ==========================================================
if st.button("Predecir Precio de Venta"):
    try:
        pred = model.predict(nuevos_datos)[0]
        st.success(f"💰 Precio estimado de la vivienda (SalePrice): ${pred:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Error al realizar la predicción: {e}")

    st.markdown("---")
    st.caption("Modelo: RandomForestRegressor (10 variables numéricas)")
