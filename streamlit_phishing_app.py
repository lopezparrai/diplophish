# streamlit_phishing_app.py
# ---------------------------------------------------------------
# App minimal para clasificar una URL/Dominio como phishing (1) o no (0)
# Requisitos: streamlit, joblib (o pickle), xgboost (solo si lo requiere tu modelo), pandas, numpy
# Tu código de features debe exponer:
#   - procesar_dominio_basico(dominio: str) -> dict
#   - enriquecer_dominio_scraping(dominio: str) -> dict
# (los proveíste). Este script los importa desde features.py
# 
# Estructura esperada del proyecto:
#   - streamlit_phishing_app.py  (este archivo)
#   - models/xgb_phishing.pkl    (tu modelo ya entrenado: joblib.dump o pickle)
#   - models/standard_scaler.pkl (opcional, si guardaste el scaler separado)
#   - feature_order.json         (orden de features usado para entrenar, opcional)
#   - features.py                (con tus dos funciones de extracción)
# 
# Ejecutar localmente:
#   streamlit run streamlit_phishing_app.py
# ---------------------------------------------------------------

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# Normalización de URLs/dominos
import re
from urllib.parse import urlparse

# ---------------------------------------------------------------
# Configuración básica de la página
# ---------------------------------------------------------------
st.set_page_config(page_title="Clasificador de Phishing", page_icon="🛡️", layout="centered")

st.title("🛡️ Clasificador de URLs: Phishing vs. Legítimo")
st.caption("Ingresa una URL o dominio, presiona **Analizar** y obtén la predicción del modelo XGBoost.")

# ---------------------------------------------------------------
# Paths (ajústalos si tu estructura difiere)
# ---------------------------------------------------------------
MODEL_PATH = Path("models/xgb_phishing_model.pkl")  # cambia si corresponde
FEATURE_ORDER_PATH = Path("feature_order.json")  # lista con el orden de columnas usado en entrenamiento
SCALER_PATH = Path("models/standard_scaler.pkl")  # opcional

# ---------------------------------------------------------------
# Import del extractor de features provisto por vos
# ---------------------------------------------------------------
try:
    # Debe existir features.py con estas funciones
    from features import procesar_dominio_basico, enriquecer_dominio_scraping
except Exception as e:  # pragma: no cover
    st.warning(
        "No se pudo importar `procesar_dominio_basico`/`enriquecer_dominio_scraping` desde `features.py`."
        "Asegúrate de colocar tu archivo `features.py` junto a este script y que exponga esas funciones."
        f"Detalle: {e}"
    )
    procesar_dominio_basico = None
    enriquecer_dominio_scraping = None

# ---------------------------------------------------------------
# Utilidades: carga de modelo, scaler y orden de features
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    # Intento con joblib; si falla, uso pickle
    try:
        import joblib
        model = joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    return model

@st.cache_resource(show_spinner=False)
def load_scaler(path: Path):
    try:
        import joblib
        if not path.exists():
            return None
        return joblib.load(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_feature_order(path: Path, _model) -> List[str]:
    """Carga feature_order.json (si existe) o intenta inferir del modelo.
    Nota: usamos `_model` (leading underscore) para que Streamlit no intente hashear el objeto.
    """
    if path.exists():
        try:
            order = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(order, list):
                raise ValueError("feature_order.json debe contener una lista de strings")
            return [str(c) for c in order]
        except Exception as e:
            st.warning(f"No se pudo leer feature_order.json: {e}. Se intentará inferir del modelo.")
    if hasattr(_model, "feature_names_in_"):
        return [str(c) for c in _model.feature_names_in_]
    return []

# SIN CACHE: evita UnhashableParamError con scaler/model
# (no es costoso y se evalúa rápido por ejecución)
def get_expected_order(feature_order_json: List[str], _scaler, _model, observed_keys: List[str]) -> List[str]:
    """Determina el orden definitivo de columnas a usar en producción.

    Prioridad:
      1) _scaler.feature_names_in_
      2) feature_order.json
      3) _model.feature_names_in_
      4) observed_keys (orden alfabético)
    """
    if _scaler is not None and hasattr(_scaler, "feature_names_in_"):
        return [str(c) for c in _scaler.feature_names_in_]
    if feature_order_json:
        return [str(c) for c in feature_order_json]
    if hasattr(_model, "feature_names_in_"):
        return [str(c) for c in _model.feature_names_in_]
    return sorted([str(k) for k in observed_keys])

# ---------------------------------------------------------------
# Alineación robusta de features: rellena faltantes con 0 y descarta sobrantes
# ---------------------------------------------------------------
def ensure_feature_vector(feat_map: Dict[str, float], feature_order: List[str]) -> pd.DataFrame:
    """Devuelve un DataFrame con una única fila y columnas en `feature_order`.
    - Completa faltantes con 0.
    - Descarta claves extra que el modelo no usa.
    - Convierte booleanos a 0/1 y None a 0.
    """
    def _cast(v):
        if v is None:
            return 0.0
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, str):
            try:
                return float(v)
            except Exception:
                return 0.0
        try:
            return float(v)
        except Exception:
            return 0.0

    row = {k: _cast(feat_map.get(k, 0.0)) for k in feature_order}
    return pd.DataFrame([row], columns=feature_order)

# ---------------------------------------------------------------
# Normalización de entrada: aceptar URL completa o dominio suelto
# ---------------------------------------------------------------
DOMAIN_REGEX = re.compile(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

def normalize_to_domain(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    # Si no tiene esquema, intenta parsear igual
    if not text.startswith(("http://", "https://")):
        # Si parece dominio simple, devuélvelo
        if DOMAIN_REGEX.match(text):
            return text
        # Si viene con ruta pero sin esquema, añade http:// para parsear
        text = "http://" + text
    parsed = urlparse(text)
    host = parsed.netloc or parsed.path
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host

# ---------------------------------------------------------------
# Carga de modelo, scaler y orden esperado de features
# ---------------------------------------------------------------
model = None
feature_order = []
scaler = None

with st.spinner("Cargando modelo/scaler..."):
    try:
        model = load_model(MODEL_PATH)
        scaler = load_scaler(SCALER_PATH)
        feature_order = load_feature_order(FEATURE_ORDER_PATH, model)
    except Exception as e:  # pragma: no cover
        st.error(f"Error cargando el modelo/scaler: {e}")

# ---------------------------------------------------------------
# UI mínima
# ---------------------------------------------------------------
entrada = st.text_input("URL o dominio a analizar", placeholder="https://ejemplo.com/…")
analizar = st.button("Analizar", type="primary", use_container_width=True)

# Área de resultado
resultado_placeholder = st.empty()

def render_diffs(calculadas: Dict[str, float], esperadas: List[str]):
    calculadas_cols = set(calculadas.keys())
    esperadas_cols = set(esperadas)
    faltantes = list(esperadas_cols - calculadas_cols)
    sobrantes = list(calculadas_cols - esperadas_cols)

    st.caption(
        f"**Features presentes (merge de básicas+scraping):** {len(calculadas_cols)}  |  **Esperadas por el modelo:** {len(esperadas)}"
    )
    if faltantes:
        with st.expander("Ver features faltantes (se imputan 0)"):
            st.write(", ".join(sorted(faltantes)))
    if sobrantes:
        with st.expander("Ver features sobrantes (no usadas por el modelo)"):
            st.write(", ".join(sorted(sobrantes)))

if analizar:
    if not entrada.strip():
        st.warning("Ingresá una URL o dominio válido.")
    elif procesar_dominio_basico is None or enriquecer_dominio_scraping is None:
        st.error("No se pudieron importar las funciones de features. Revisa `features.py`.")
    elif model is None:
        st.error("No se pudo cargar el modelo.")
    else:
        try:
            dominio = normalize_to_domain(entrada)
            if not dominio:
                st.warning("No se pudo interpretar la entrada como dominio/URL.")
            else:
                with st.spinner("Extrayendo features (WHOIS + scraping)…"):
                    base_feats = procesar_dominio_basico(dominio)  # dict
                    dyn_feats = enriquecer_dominio_scraping(dominio)  # dict
                    feats = {**(base_feats or {}), **(dyn_feats or {})}

                # Determinar el orden definitivo de columnas (prioriza scaler)
                expected_order = get_expected_order(feature_order, scaler, model, list(feats.keys()))

                # Mostrar diferencias presentes vs. esperadas
                render_diffs(feats, expected_order)

                # Vector alineado **solo** con las columnas esperadas
                X = ensure_feature_vector(feats, expected_order)

                # Escalado opcional (si existe scaler externo)
                X_in = X
                if scaler is not None:
                    # Valida columnas idénticas al scaler
                    if hasattr(scaler, "feature_names_in_"):
                        scaler_cols = list(scaler.feature_names_in_)
                        # Validación fuerte: mismas columnas y mismo orden
                        missing = [c for c in scaler_cols if c not in X.columns]
                        if missing:
                            st.error("Faltan columnas requeridas por el scaler: " + ", ".join(missing))
                            st.stop()
                        # Reordenar y descartar extra (ya están descartadas por ensure_feature_vector)
                        X = X[scaler_cols]
                    X_in = scaler.transform(X)

                # Predicción
                with st.spinner("Prediciendo…"):
                    proba = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_in)  # shape (1, 2)
                    y_pred = model.predict(X_in)

                label = int(y_pred[0]) if hasattr(y_pred, "__iter__") else int(y_pred)
                if proba is not None and np.ndim(proba) == 2 and proba.shape[1] >= 2:
                    p_phishing = float(proba[0, 1])
                else:
                    p_phishing = 1.0 if label == 1 else 0.0

                # Render resultado
                if label == 1:
                    resultado_placeholder.error(
                        f"**Posible PHISHING (1)** — probabilidad clase 1: {p_phishing:.3f}"
                    )
                else:
                    resultado_placeholder.success(
                        f"**No phishing (0)** — probabilidad clase 1: {p_phishing:.3f}"
                    )

                # Expander con el vector definitivo y, si aplica, columnas del scaler
                with st.expander("Ver vector de features alineado"):
                    st.dataframe(X.T.rename(columns={0: "valor"}))
                    if scaler is not None and hasattr(scaler, "feature_names_in_"):
                        st.caption("Columnas esperadas por el scaler:")
                        st.code(", ".join(list(scaler.feature_names_in_)))

        except Exception as e:  # pragma: no cover
            st.exception(e)

# Footer sutil
st.markdown("")
st.caption(
    "Modelo: XGBoost ya entrenado. Front minimalista."
    "Ajusta MODEL_PATH/FEATURE_ORDER_PATH según tu proyecto y asegúrate de que `features.py` exponga las funciones solicitadas."
)
