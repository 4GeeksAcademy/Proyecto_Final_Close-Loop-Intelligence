from utils import db_connect
engine = db_connect()

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Clasificador ABC de Inventario", page_icon="📦")

st.title("📦 Sistema de Clasificación ABC de Inventario")
st.markdown("""
Esta aplicación utiliza un modelo de **Random Forest** para predecir la categoría de un producto (A, B o C) 
basándose en variables logísticas y de ventas.
""")

# 1. Cargar el modelo
@st.cache_resource
def load_model():
    with open('models/best_rf_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# 2. Formulario de entrada de datos
st.sidebar.header("📥 Datos del Producto")

def user_input_features():
    # 10 variables utilizadas para la predicción ['Ciudad Cliente', 'Ciudad Vendedor', 'Categoria Producto', 'Precio Unitario', 
    # 'Cantidad', 'Costo Flete', 'Total Orden', 'Estado Orden', 'Calificacion', 'Tiempo de Reposicion']
    ciudad_cliente = st.sidebar.text_input("Ciudad Cliente")
    ciudad_vendedor = st.sidebar.text_input("Ciudad Vendedor")
    categoria_producto = st.sidebar.selectbox("Categoría del Producto", [])  # Completa con las categorías disponibles
    precio_unitario = st.sidebar.number_input("Precio Unitario", min_value=0.0)
    cantidad = st.sidebar.slider("Cantidad de Orden", 1, 100, 1)
    costo_flete = st.sidebar.number_input("Costo Flete", min_value=0.0)
    total_orden = st.sidebar.number_input("Total de Orden", min_value=0.0)
    estado_orden = st.sidebar.selectbox("Estado de Orden", ["Pendiente", "Completado", "Cancelado"])
    calificacion = st.sidebar.slider("Calificación del Producto (1-5)", 1, 5, 3)
    tiempo_reposicion = st.sidebar.slider("Tiempo de Reposición (días)", 1, 30, 7)
    

    
    data = {
        'Ciudad Cliente': ciudad_cliente,
        'Ciudad Vendedor': ciudad_vendedor,
        'Categoria Producto': categoria_producto,
        'Precio Unitario': precio_unitario,
        'Cantidad': cantidad,
        'Costo Flete': costo_flete,
        'Total Orden': total_orden,
        'Estado Orden': estado_orden,
        'Calificacion': calificacion,
        'Tiempo de Reposicion': tiempo_reposicion,
        
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# 3. Predicción
st.subheader("Predicción de Categoría")
if st.button("Clasificar Producto"):
    prediction = model.predict(df)
    proba = model.predict_proba(df)
    
    categories = {0: "Clase A (Alta Rotación)", 1: "Clase B (Media Rotación)", 2: "Clase C (Baja Rotación)"}
    result = categories[prediction[0]]
    
    st.success(f"El producto ha sido clasificado como: **{result}**")
    
    # Mostrar probabilidades en un gráfico
    st.write("Probabilidad por categoría:")
    st.bar_chart(pd.DataFrame(proba, columns=categories.values()).T)
