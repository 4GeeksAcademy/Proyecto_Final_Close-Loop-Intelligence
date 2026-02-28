from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine

#### @st.cache_resource
###def load_classification_assets():
    model = pickle.load(open('models/best_rf_model.sav', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    return model, scaler

##@st.cache_resource
## #def get_prophet_forecast(cat_name):
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