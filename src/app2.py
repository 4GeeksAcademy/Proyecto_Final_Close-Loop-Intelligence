import streamlit as st
import os 
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# --- CONFIGURACIÓN Y CARGA ---
st.set_page_config(page_title="Close-Loop Intelligence", layout="wide")

# --- FUNCIONES DE CARGA DE MODELOS Y DATOS ---

@st.cache_resource
def load_classification_assets():
    # Ubicamos la carpeta src y subimos un nivel a la raíz
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, '..', 'models', 'best_rf_model.sav')
    scaler_path = os.path.join(base_path, '..', 'models', 'scaler.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    return model, scaler

@st.cache_resource
def get_prophet_forecast(cat_name):
    """Carga el modelo específico subiendo un nivel desde src hacia models"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Construcción dinámica del nombre
    nombre_archivo = f"ProphetA_{cat_name.replace(' ', '_')}.pkl"
    full_path = os.path.join(base_path, '..', 'models', nombre_archivo)
    
    try:
        if os.path.exists(full_path):
            with open(full_path, 'rb') as f:
                m = pickle.load(f)
            future = m.make_future_dataframe(periods=15, freq='D')
            forecast = m.predict(future)
            return m, forecast
        else:
            st.error(f"No se encontró el modelo en: {full_path}")
            return None, None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None


@st.cache_data
def cargar_diccionario_categorias():
    try:
        path = os.path.dirname(os.path.abspath(__file__))
        
        csv_path = os.path.join(path,'..' , 'data', 'interim', 'TopA.csv')
    

        TopA = pd.read_csv(csv_path)
        
        # 2. Convertimos la columna a una lista y creamos el diccionario
        # Usamos enumerate para asignarles un ID numérico automáticamente
        lista_categorias = TopA['Categoria Producto'].unique().tolist()
        diccionario = {cat: i for i, cat in enumerate(lista_categorias)}
        
        return diccionario
    except Exception as e:
        st.error(f"Error al cargar el archivo de categorías: {e}")
        return {}
    
@st.cache_data
def generar_diccionario_categorias_clasificacion():
    """
    Toma el DataFrame MasterVenta, extrae las categorías únicas 
    y genera un diccionario con IDs numéricos.
    """
    try:

        path = os.path.dirname(os.path.abspath(__file__))
        
        csv_path = os.path.join(path,'..' , 'data', 'interim', 'MasterVentas.csv')
        df_master = pd.read_csv(csv_path)

        lista_categorias = df_master['Categoria Productos'].unique().tolist()
        
        diccionario = {cat: i for i, cat in enumerate(lista_categorias)}
        
        return diccionario
        
    except Exception as e:
        st.error(f"Error al procesar las categorías del DataFrame: {e}")
        return {}


# --- DICCIONARIOS ---
categorias = cargar_diccionario_categorias()
CategoriasClas = generar_diccionario_categorias_clasificacion()

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
        cat = st.selectbox("Categoría de Producto", list(CategoriasClas.keys()))
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
                st.markdown(f"### La Categoria de Producto es: <span style='color:{color}'>{clase_result}</span>", unsafe_allow_html=True)
                
                # Gráfico de probabilidades
                st.bar_chart(pd.DataFrame(proba, columns=mapping_abc.values()).T)
                
                if prediction[0] == 0:
                    st.info("💡 **Sugerencia:** Esta Categoria de Producto es de alta prioridad. Revisa la pestaña de predicción para ajustar tu stock.")
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

    # --- MÉTRICAS DE RESUMEN QUINCENAL (BLINDADO) ---
    st.divider()
        
    # 1. Verificación: ¿Hay categorías seleccionadas?
    if seleccionadas:
        st.subheader("📦 Planificación de Reabastecimiento (Próximos 15 días)")
            
        # Inicializamos el contador global
        total_unidades_global = 0
        datos_metricas = []

        # 2. Procesamos los datos antes de mostrar para obtener el total global
        for cat in seleccionadas:
            model_p, forecast_p = get_prophet_forecast(cat)
            if forecast_p is not None:
            # Sumamos la predicción (yhat) de los últimos 15 días
                suma_cat = forecast_p.tail(15)['yhat'].sum()
                total_unidades_global += suma_cat
                datos_metricas.append((cat, suma_cat))

        # 3. Mostramos el Gran Total primero (Destacado)
        col_total, _ = st.columns([1, 3])
        with col_total:
            st.metric(
                    label="✨ TOTAL GLOBAL DE UNIDADES", 
                    value=f"{int(total_unidades_global)} unids.",
                    delta=f"{len(seleccionadas)} categorías analizadas",
                    delta_color="normal"
                )
            
        st.write("---") # Separador visual sutil

        # 4. Mostramos el desglose por categorías en columnas dinámicas
        # Usamos un máximo de 4 columnas por fila para que no se vea amontonado
        num_cols = min(len(datos_metricas), 4)
        cols = st.columns(num_cols)
            
        for i, (nombre_cat, valor) in enumerate(datos_metricas):
                with cols[i % num_cols]:
                    st.metric(
                        label=f"Demanda {nombre_cat}", 
                        value=f"{int(valor)} unids."
                    )
            
        st.info(f"💡 Se recomienda asegurar un stock de al menos **{int(total_unidades_global)} unidades** para cubrir la demanda proyectada del conjunto seleccionado.")
        
    else:
            # Mensaje amigable si no hay nada seleccionado
        st.warning("⚠️ Por favor, seleccione al menos una categoría en el buscador superior para visualizar las métricas de abastecimiento.")

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
            cat_actual = seleccionadas[0]
            model_p, forecast_p = get_prophet_forecast(cat_actual)
            
            with col_a:
                st.subheader("📉 Componentes del Modelo")
                # Cambio a Plotly interactivo para tendencias
                fig_comp = plot_components_plotly(model_p, forecast_p)
                fig_comp.update_layout(height=1000) # Ajuste de altura para que quepa bien
                st.plotly_chart(fig_comp, use_container_width=True)
                
            with col_b:
                st.subheader("✅ Validación (Predicción vs Real)")
                # Cambio a Plotly interactivo para Predicción vs Realidad
                # Esto incluye los puntos negros (reales) y la línea/sombra azul
                fig_val = plot_plotly(model_p, forecast_p)

                fig_val.data[0].marker.color = '#00f2ff'  # Un cian brillante que resalta en fondo oscuro
                fig_val.data[0].marker.size = 6
                fig_val.data[1].line.color = '#ff7f0e'         
                
                fig_val.update_layout(
                    title=f"Ajuste Histórico: {cat_actual}",
                    height=1000,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_val, use_container_width=True)
        else:
            st.caption("Seleccione una sola categoría para ver el desglose de componentes y validación detallada.")