import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Clasificador ABC de Inventario", page_icon="📦")

st.title("📦 Sistema de Clasificación ABC de Inventario")
st.markdown("""
Esta aplicación utiliza un modelo de **Random Forest** para predecir la Clasificacion de Rotación de una Categoria de Producto (A, B o C) 
basándose en variables logísticas y de ventas.
""")

# Cargar el modelo
@st.cache_resource
def load_model():
    with open('../models/best_rf_model.sav', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Mapeos
ciudades_vendedor = {
    "Sao Paulo": 0,
    "Rio de Janeiro": 1,
    "Belo Horizonte": 2,
    "Curitiba": 3
  ### Falta mapear el top 10 de ciudades
}

categorias = {
    "Camas y Mesa": 0,
    "Belleza y Salud": 1,
    "Informática": 2,
    "Deportes": 3
   
   ### Falta mapear el top 10 de categorias
}

# Formulario de entrada de datos
st.sidebar.header("📥 Datos del Producto")

def user_input_features():
    # 5 variables utilizadas para la predicción ['Ciudad Vendedor', 'Categoria Producto', 'Precio Unitario','Cantidad','Tiempo de Reposicion']
    # Desplegables
    ciudad = st.sidebar.selectbox("Ciudad del Vendedor", list(ciudades_vendedor.keys()))
    cat = st.sidebar.selectbox("Categoría del Producto", list(categorias.keys()))
    
    # Numéricos
    precio_unitario = st.sidebar.number_input("Precio Unitario", min_value=0.0, value=100.0)
    cantidad = st.sidebar.slider("Cantidad", 1, 100, 5)
    tiempo_reposicion = st.sidebar.slider("Tiempo de Reposición (días)", 1, 60, 15)
        
    data = {
        'Ciudad Vendedor': ciudades_vendedor[ciudad],
        'Categoria Producto': categorias[cat],
        'Precio Unitario': precio_unitario,
        'Cantidad': cantidad,
        'Tiempo de Reposicion': tiempo_reposicion,   
    }

# Escalado de variables numéricas con Min Max Scaler (usando los mismos parámetros que el entrenamiento)

    scaler = pickle.load(open('../models/scaler.pkl', 'rb'))
    data_scaled = scaler.fit_transform([[data['Precio Unitario'], data['Cantidad'], data['Tiempo de Reposicion']]])

    return pd.DataFrame(data_scaled, index=[0])

df = user_input_features()



# Predicción
st.subheader("Predicción de Clasificacion")
if st.button("Clasificar Categoria"):
    prediction = model.predict(df)
    proba = model.predict_proba(df)
    
    categories = {0: "Clase A (Alta Rotación)", 1: "Clase B (Media Rotación)", 2: "Clase C (Baja Rotación)"}
    result = categories[prediction[0]]
    
    st.success(f"El producto ha sido clasificado como: **{result}**")
    
    # Mostrar probabilidades en un gráfico
    st.write("Probabilidad por categoría:")
    st.bar_chart(pd.DataFrame(proba, columns=categories.values()).T)
