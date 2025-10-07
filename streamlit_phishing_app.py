import os
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- Optional dependencies (handle gracefully if missing) ---
try:
    import xgboost as xgb
except Exception as e:
    xgb = None

from urllib.parse import urlparse
import re
import time

# =============================================================
# 🧩 CONFIGURACIÓN
# =============================================================
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_phishing_model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/feature_order.json")  # lista con el orden exacto de features
APP_TITLE = "Detección de Phishing por URL (XGBoost)"
DESCRIPTION = (
    "Ingresa una URL. Se extraen características, se aplica el modelo XGBoost y se estima la probabilidad de phishing."
)
DEFAULT_THRESHOLD = 0.50  # puedes priorizar Recall subiendo este umbral o bajándolo según tu métrica objetivo
SEED = 13

# =============================================================
# 🧪 UTILIDADES
# =============================================================

def _looks_like_url(s: str) -> bool:
    if not s:
        return False
    # Permite entradas sin esquema ("www.sitio.com")
    pattern = r"^(https?://)?([\w.-]+)\.([a-zA-Z]{2,})(/.*)?$"
    return re.match(pattern, s.strip()) is not None


def _normalize_url(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if not s.startswith(("http://", "https://")):
        s = "http://" + s  # por defecto http; muchos features no dependen del esquema
    return s


# =============================================================
# 📥 CARGA DE MODELO Y METADATOS
# =============================================================
@st.cache_resource(show_spinner=True)
def load_model(model_path: str = MODEL_PATH):
    if xgb is None:
        raise ImportError("xgboost no está instalado en el entorno.")
    import joblib

    if not Path(model_path).exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    model = joblib.load(model_path)
    return model


@st.cache_resource(show_spinner=False)
def load_feature_order(features_path: str = FEATURES_PATH) -> List[str]:
    if Path(features_path).exists():
        with open(features_path, "r", encoding="utf-8") as f:
            order = json.load(f)
        if not isinstance(order, list):
            raise ValueError("El archivo de features debe contener una lista JSON.")
        return order
    # fallback: intentar usar atributo del modelo si existe
    try:
        model = load_model()
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass
    # último recurso: vacía, y más abajo validaremos
    return []


# =============================================================
# 🧱 EXTRACCIÓN / ENRIQUECIMIENTO DE FEATURES
# =============================================================
# 🔧 IMPORTANTE: Ignacio, pega aquí tus funciones de enriquecimiento/web scraping
# y rellena la función `extract_features(url)` para devolver un dict {feature: valor}


def extract_features(url: str) -> Dict[str, float]:
    """
    TODO: Reemplazar por tu pipeline real. Debe devolver un dict {feature_name: valor}.
    Debe ser determinístico y no depender de estado global.
    Ejemplo simple basado sólo en la cadena URL (placeholder):
    """
    parsed = urlparse(url)
    host = parsed.netloc or ""
    path = parsed.path or ""

    features = {
        # === Ejemplos básicos, REEMPLAZA por los tuyos ===
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "count_digits": sum(c.isdigit() for c in url),
        "count_dashes": url.count("-"),
        "count_dots": url.count("."),
        "has_at": int("@" in url),
        "has_https": int(url.startswith("https://")),
        "subdirs": path.count("/"),
    }
    return features


@st.cache_data(show_spinner=False)
def extract_features_cached(url: str) -> Dict[str, float]:
    return extract_features(url)


def _align_features(row_feat: Dict[str, float], feature_order: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Alinea el dict de features al orden esperado por el modelo. Rellena faltantes con 0.
    Retorna: vector numpy (1, n_features), y lista de features utilizados.
    """
    if not feature_order:
        # si no tenemos orden, usar el orden del dict (no recomendado)
        cols = list(row_feat.keys())
        x = np.array([[row_feat.get(k, 0.0) for k in cols]], dtype=float)
        return x, cols
    cols = feature_order
    x = np.array([[row_feat.get(k, 0.0) for k in cols]], dtype=float)
    return x, cols


# =============================================================
# 🔮 INFERENCIA
# =============================================================
@st.cache_data(show_spinner=False)
def predict_proba_single(url: str, threshold: float) -> Dict[str, float]:
    model = load_model()
    feature_order = load_feature_order()

    feats = extract_features_cached(url)
    X, used_cols = _align_features(feats, feature_order)

    # Verificación defensiva de dimensiones
    if hasattr(model, "n_features_in_") and len(used_cols) != int(model.n_features_in_):
        raise ValueError(
            f"Número de features desalineado: modelo espera {getattr(model, 'n_features_in_', '?')} y se recibieron {len(used_cols)}.\n"
            f"Asegúrate de exportar 'feature_order.json' del entrenamiento o usar 'model.feature_names_in_'"
        )

    # predict_proba para XGBClassifier scikit-like
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0, 1])
    else:
        # fallback: usar .predict con output_margin/transformación si fuese Booster
        dmat = xgb.DMatrix(X, feature_names=used_cols) if xgb else X
        proba = float(model.predict(dmat)[0])  # asumir ya es probabilidad

    predicted = 1 if proba >= threshold else 0
    return {
        "proba_phishing": proba,
        "predicted_label": predicted,
        "threshold": threshold,
        "n_features": len(used_cols),
    }


def predict_proba_batch(urls: List[str], threshold: float) -> pd.DataFrame:
    rows = []
    for u in urls:
        try:
            res = predict_proba_single(u, threshold)
            rows.append({"url": u, **res})
        except Exception as e:
            rows.append({"url": u, "error": str(e)})
    df = pd.DataFrame(rows)
    if "proba_phishing" in df.columns:
        df = df.sort_values("proba_phishing", ascending=False)
    return df


# =============================================================
# 🖥️ UI STREAMLIT
# =============================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🕵️", layout="wide")
st.title(APP_TITLE)
st.caption(DESCRIPTION)

with st.sidebar:
    st.header("Parámetros")
    threshold = st.slider(
        "Umbral de decisión (≥ = phishing)", min_value=0.0, max_value=1.0, value=DEFAULT_THRESHOLD, step=0.01
    )
    st.write("\n")
    st.markdown(
        "**Sugerencia:** si tu métrica objetivo prioriza *Recall* para la clase positiva (phishing), considera bajar el umbral para detectar más casos a costa de más falsos positivos."
    )
    st.divider()
    st.caption("Ruta del modelo:")
    st.code(MODEL_PATH)

# --- Pestañas ---
tab1, tab2 = st.tabs(["Predicción única", "Batch por CSV"]) 

# --------------------------
# TAB 1: PREDICCIÓN ÚNICA
# --------------------------
with tab1:
    st.subheader("Clasificar una URL")
    url_input = st.text_input(
        "Ingresa una URL",
        placeholder="p.ej. www.midominio.com/oferta/increible",
    )
    colA, colB = st.columns([1, 3])
    with colA:
        do_predict = st.button("Analizar", type="primary")
    with colB:
        st.write("")

    if do_predict:
        if not _looks_like_url(url_input):
            st.error("La cadena no parece una URL válida. Intenta con 'dominio.com' o 'https://dominio.com'.")
        else:
            url_norm = _normalize_url(url_input)
            with st.spinner("Extrayendo features y aplicando modelo..."):
                try:
                    res = predict_proba_single(url_norm, threshold)
                    proba = res["proba_phishing"]
                    label = res["predicted_label"]

                    st.metric(
                        label="Probabilidad estimada de phishing",
                        value=f"{proba:.3f}",
                        delta=f">= {threshold:.2f} ⇒ {'Phishing' if label==1 else 'Legítima'}",
                        delta_color="inverse" if label == 0 else "normal",
                    )

                    # Mostrar features extraídos
                    with st.expander("Ver features extraídos"):
                        feats = extract_features_cached(url_norm)
                        st.dataframe(pd.DataFrame([feats]).T.rename(columns={0: "valor"}))

                    # Log ligero
                    st.caption(
                        f"Modelo: {Path(MODEL_PATH).name} · Features usados: {res['n_features']} · Umbral: {threshold:.2f}"
                    )
                except Exception as e:
                    st.error(f"Ocurrió un error durante la inferencia: {e}")

# --------------------------
# TAB 2: BATCH CSV
# --------------------------
with tab2:
    st.subheader("Procesar varias URLs por CSV")
    st.caption(
        "Sube un CSV con una columna llamada **url**. Se calculará la probabilidad para cada fila."
    )
    file = st.file_uploader("Subir CSV", type=["csv"])
    if file:
        try:
            df_in = pd.read_csv(file)
            if "url" not in df_in.columns:
                st.error("El CSV debe contener una columna llamada 'url'.")
            else:
                urls = [
                    _normalize_url(u) for u in df_in["url"].astype(str).tolist() if _looks_like_url(str(u))
                ]
                with st.spinner("Ejecutando inferencia batch..."):
                    df_out = predict_proba_batch(urls, threshold)
                st.dataframe(df_out, use_container_width=True)

                # Descargar resultados
                csv_buf = io.StringIO()
                df_out.to_csv(csv_buf, index=False)
                st.download_button(
                    "Descargar resultados (CSV)",
                    csv_buf.getvalue(),
                    file_name="predicciones_phishing.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")

# =============================================================
# 📝 NOTAS PARA INTEGRACIÓN
# =============================================================
with st.expander("Cómo integrar tu pipeline real de features"):
    st.markdown(
        """
        1. **Pega tus funciones** de enriquecimiento / web scraping en este archivo y **reemplaza** el contenido de `extract_features(url)` para que devuelva un `dict` `{nombre_feature: valor}`.
        2. Exporta desde tu entrenamiento el **orden exacto de features** y guárdalo en `models/feature_order.json` (una lista JSON). Alternativamente, si tu modelo tiene `feature_names_in_`, se usará eso.
        3. Asegúrate de que los **nombres y cantidad** de features coincidan con lo que espera el modelo (se valida antes de inferir).
        4. Para correr localmente: `streamlit run streamlit_phishing_app.py`.
        5. Ajusta el **umbral** en la barra lateral para priorizar *Recall* (detección) vs *Precisión* (menos falsos positivos), según tus métricas objetivo (e.g., **PR-AUC** y *Recall* de la clase phishing).
        """
    )

st.sidebar.caption("Hecho con ❤️ para DiploDatos · Ignacio")
