import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet # Asegúrate de tenerlo en requirements.txt

# --- CONFIGURACIÓN Y CARGA ---
st.set_page_config(page_title="Close-Loop Intelligence", layout="wide")

@st.cache_resource
def load_classification_assets():
    model = pickle.load(open('models/best_rf_model.sav', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    return model, scaler

@st.cache_resource
def get_prophet_forecast(cat_name):
    """Carga el modelo específico y genera la predicción de 15 días"""
    # Construcción dinámica del nombre según tu estructura
    file_name = f"models/ProphetA_{cat_name.replace(' ', '_')}.pkl"
    try:
        with open(file_name, 'rb') as f:
            m = pickle.load(f)
        future = m.make_future_dataframe(periods=15, freq='D')
        forecast = m.predict(future)
        return m, forecast
    except FileNotFoundError:
        return None, None


@st.cache_data
def cargar_diccionario_categorias():
    try:
        # 1. Leemos el archivo CSV
        TopA = pd.read_csv('/workspaces/Proyecto_Final_Close-Loop-Intelligence/data/interim/TopA.csv')
        
        # 2. Convertimos la columna a una lista y creamos el diccionario
        # Usamos enumerate para asignarles un ID numérico automáticamente
        lista_categorias = TopA['Categoria Producto'].unique().tolist()
        diccionario = {cat: i for i, cat in enumerate(lista_categorias)}
        
        return diccionario
    except Exception as e:
        st.error(f"Error al cargar el archivo de categorías: {e}")
        return {}

# --- DICCIONARIOS ---
categorias = cargar_diccionario_categorias()

ciudades_vendedor = {
    "Sao Paulo": 0, "Rio de Janeiro": 1, "Belo Horizonte": 2, "Curitiba": 3,
    "Porto Alegre": 4, "Salvador": 5, "Guarulhos": 6, "Campinas": 7, "Niteroi": 8, "Osasco": 9
}

# --- INTERFAZ ---
st.title("📊 Dashboard de Inteligencia de Inventario")

st.markdown("Optimización de stock mediante **Random Forest** y **Series de Tiempo (Prophet)**.")

tab1, tab2 = st.tabs(["🎯 Clasificación de Rotacion de Inventario ABC", "📈 Predicción Quincenal"])

# PESTAÑA 1: (Random Forest)
with tab1:
    st.info("Utilice esta sección para clasificar la rotación de inventario por categoría de productos.")
   
    st.header("Clasificación de Rotación de Inventario")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📥 Entrada de Datos")
        ciudad = st.selectbox("Ciudad del Vendedor", list(ciudades_vendedor.keys()))
        cat = st.selectbox("Categoría de Producto", list(categorias.keys()))
        precio_unitario = st.number_input("Precio Unitario (USD)", min_value=0.0, value=100.0)
        cantidad = st.slider("Unidades", 1, 500, 50)
        tiempo_reposicion = st.slider("Días de Reposición", 1, 90, 15)

    with col2:
        st.subheader("📊 Resultado de la Clasificación")
        if st.button("Ejecutar Análisis de Rotación"):
            try:
                model, scaler = load_classification_assets()
                
                # Se preparan las características numericas para el modelo, asegurando el mismo orden y escalado que durante el entrenamiento
                features = pd.DataFrame([[precio_unitario, cantidad, tiempo_reposicion]], 
                                        columns=['Precio Unitario', 'Cantidad', 'Tiempo de Reposicion'])
                
                # Escalado (Usando transform, NO fit_transform)
                features_scaled = scaler.transform(features)
                
                # Para la predicción de RF, necesitamos unir las 5 variables originales:
                # [Ciudad, Categoria, Precio_S, Cantidad_S, Reposicion_S]
                input_final = np.array([[
                    ciudades_vendedor[ciudad], 
                    categorias[cat], 
                    features_scaled[0][0], 
                    features_scaled[0][1], 
                    features_scaled[0][2]
                ]])

                prediction = model.predict(input_final)
                proba = model.predict_proba(input_final)
                
                mapping_abc = {0: "Clase A (Alta Rotación)", 1: "Clase B (Media)", 2: "Clase C (Baja)"}
                clase_result = mapping_abc[prediction[0]]

                # Mostrar resultado con color
                color = "green" if prediction[0] == 0 else "orange" if prediction[0] == 1 else "red"
                st.markdown(f"### El producto es: <span style='color:{color}'>{clase_result}</span>", unsafe_allow_html=True)
                
                # Gráfico de probabilidades
                st.bar_chart(pd.DataFrame(proba, columns=mapping_abc.values()).T)
                
                if prediction[0] == 0:
                    st.info("💡 **Sugerencia:** Este producto es de alta prioridad. Revisa la pestaña de predicción para ajustar tu stock.")
            except Exception as e:
                st.error(f"Error en la predicción: {e}")

# PESTAÑA 2: SERIES DE TIEMPO
with tab2:
    st.header("Pronóstico de Unidades Vendidas (Próximos 15 días), por categoria de productos de Alta Rotación")
    
    seleccionadas = st.multiselect(
        "Seleccione categorías de Alta Rotación para comparar:",
        list(categorias.keys()),
        default= None
    )

    if seleccionadas:
        fig_main = go.Figure()
        
        for cat in seleccionadas:
            model_p, forecast_p = get_prophet_forecast(cat)
            
            if forecast_p is not None:
                # Tomamos los últimos 15 días de la predicción
                ultimo_forecast = forecast_p.tail(15)
                
                fig_main.add_trace(go.Scatter(
                    x=ultimo_forecast['ds'],
                    y=ultimo_forecast['yhat'],
                    mode='lines+markers',
                    name=f"Pred. {cat}",
                    hovertemplate='%{x|%d %b}: %{y:.0f} unidades'
                ))
            else:
                st.error(f"No se encontró el archivo: models/ProphetA_{cat}.pkl")

        fig_main.update_layout(
            title="Comparativa de Demanda Estimada",
            xaxis_title="Fecha",
            yaxis_title="Unidades",
            hovermode="x unified"
        )
        st.plotly_chart(fig_main, use_container_width=True)

        # --- SECCIÓN DE TENDENCIAS Y COMPONENTES ---
        st.divider()
        if len(seleccionadas) == 1:
            col_a, col_b = st.columns(2)
            model_p, forecast_p = get_prophet_forecast(seleccionadas[0])
            
            with col_a:
                st.subheader("📉 Componentes del Modelo")
                fig_comp = model_p.plot_components(forecast_p)
                st.write(fig_comp)
                
            with col_b:
                st.subheader("✅ Validación (Predicción vs Real)")
                # Aquí puedes graficar forecast_p['yhat'] contra tus datos históricos reales
                st.write("Visualizando tendencia histórica y ajuste del modelo...")
                fig_val = model_p.plot(forecast_p)
                st.write(fig_val)
        else:
            st.caption("Seleccione una sola categoría para ver el desglose de componentes y validación detallada.")