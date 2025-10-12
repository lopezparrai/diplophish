# streamlit_phishing_app.py
# ---------------------------------------------------------------
# App para clasificar una URL/Dominio como phishing (1) o no (0)
# Requisitos: streamlit, joblib (o pickle), xgboost (si tu modelo lo requiere),
#             pandas, numpy, plotly
# Tu archivo features.py debe exponer:
#   - procesar_dominio_basico(dominio: str) -> dict
#   - enriquecer_dominio_scraping(dominio: str) -> dict
#
# Estructura esperada del proyecto:
#   - streamlit_phishing_app.py  (este archivo)
#   - models/xgb_phishing.pkl    (tu modelo ya entrenado: joblib.dump o pickle)
#   - models/standard_scaler.pkl (requerido para este deployment)
#   - feature_order.json         (orden de features usado para entrenar, opcional)
#   - features.py                (con tus funciones de extracción)
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
import plotly.graph_objects as go

# Normalización de URLs/dominos
import re
from urllib.parse import urlparse

# ---------------------------------------------------------------
# Configuración básica de la página
# ---------------------------------------------------------------
st.set_page_config(page_title="Clasificador de Phishing", page_icon="🛡️", layout="centered")

st.title("🛡️ Clasificador de URLs: Phishing vs. Legítimo")
st.caption("Ingresá una URL o dominio, presioná **Analizar** y obtené la predicción del modelo.")

# ==== Estilos de dashboard (card con sombra + degradado) ====
st.markdown("""
<style>
.gauge-card {
  background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 60%, #eef2f7 100%);
  border-radius: 18px;
  box-shadow: 0 8px 24px rgba(30, 42, 70, 0.15), 0 2px 6px rgba(30, 42, 70, 0.08);
  padding: 18px 18px 8px 18px;
  border: 1px solid rgba(0,0,0,0.05);
}
.gauge-title {
  font-size: 0.95rem;
  color: #5b6b7f;
  margin: 0 0 6px 2px;
  letter-spacing: .3px;
}
.gauge-subtitle {
  font-size: 0.85rem;
  color: #7a8a9f;
  margin: -6px 0 10px 2px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Paths (ajústalos si tu estructura difiere)
# ---------------------------------------------------------------
MODEL_PATH = Path("models/xgb_phishing.pkl")
FEATURE_ORDER_PATH = Path("feature_order.json")
SCALER_PATH = Path("models/standard_scaler.pkl")

# ---------------------------------------------------------------
# Import del extractor de features provisto por vos
# ---------------------------------------------------------------
try:
    from features import procesar_dominio_basico, enriquecer_dominio_scraping
except Exception as e:  # pragma: no cover
    st.warning(
        "No se pudo importar `procesar_dominio_basico`/`enriquecer_dominio_scraping` desde `features.py`.\n"
        "Asegurate de colocar tu archivo `features.py` junto a este script y que exponga esas funciones.\n\n"
        f"Detalle: {e}"
    )
    procesar_dominio_basico = None
    enriquecer_dominio_scraping = None

# ---------------------------------------------------------------
# Definiciones de columnas de entrenamiento y escalado
# ---------------------------------------------------------------
TRAIN_FEATURES_FALLBACK = [
    'url_length','num_dashes','num_digits','num_special_chars','path_segments','num_dots',
    'hostname_length','path_length','query_length','num_underscores','num_dashes_in_hostname',
    'title_length','http_status_code','has_https','has_ssl_cert','iframe_present','insecure_forms',
    'submit_info_to_email','abnormal_form_action','double_slash_in_path','is_registered_in_ar','responds'
]

NUMERIC_FEATURES = [
    'url_length','num_dashes','num_digits','num_special_chars','path_segments','num_dots',
    'num_underscores','num_dashes_in_hostname','hostname_length','path_length','query_length',
    'site_age_years','time_to_expire_years','response_time','http_status_code','title_length',
    'sensitive_words_count'
]

# ---------------------------------------------------------------
# Utilidades: carga de modelo, scaler y orden de features
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
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

def get_expected_order(feature_order_json: List[str], _scaler, _model, observed_keys: List[str]) -> List[str]:
    if hasattr(_model, "feature_names_in_") and getattr(_model, "feature_names_in_", None) is not None:
        fins = [str(c) for c in _model.feature_names_in_]
        if len(fins) > 0:
            return fins
    if feature_order_json:
        return [str(c) for c in feature_order_json]
    if TRAIN_FEATURES_FALLBACK:
        return list(TRAIN_FEATURES_FALLBACK)
    if _scaler is not None and hasattr(_scaler, "feature_names_in_") and getattr(_scaler, "feature_names_in_", None) is not None:
        sins = [str(c) for c in _scaler.feature_names_in_]
        if len(sins) > 0:
            return sins
    return sorted([str(k) for k in observed_keys])

# ---------------------------------------------------------------
# Alineación robusta de features: rellena faltantes con 0 y descarta sobrantes
# ---------------------------------------------------------------
def ensure_feature_vector(feat_map: Dict[str, float], feature_order: List[str]) -> pd.DataFrame:
    def _cast(v):
        if v is None:
            return 0.0
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float, np.number)):
            try:
                return float(v)
            except Exception:
                return 0.0
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
    if not text.startswith(("http://", "https://")):
        if DOMAIN_REGEX.match(text):
            return text
        text = "http://" + text
    parsed = urlparse(text)
    host = parsed.netloc or parsed.path
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host

# ---------------------------------------------------------------
# UI: entrada
# ---------------------------------------------------------------
entrada = st.text_input("URL o dominio a analizar", placeholder="https://ejemplo.com/…")
analizar = st.button("Analizar", type="primary", use_container_width=True)

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

if scaler is None:
    st.error("No se encontró `models/standard_scaler.pkl`. Este deployment **requiere** el scaler entrenado para transformar las variables numéricas.")
    st.stop()

# ---------------------------------------------------------------
# Visuales: difs de features y tacómetro (dashboard)
# ---------------------------------------------------------------
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

def riesgo_bucket(p: float):
    """Etiqueta y recomendación según prob. de phishing."""
    if p < 0.20:
        return "Bajo", "Parece legítimo, pero mantené precaución."
    elif p < 0.50:
        return "Medio", "Verificá la URL y el remitente antes de interactuar."
    elif p < 0.80:
        return "Alto", "Evitá ingresar datos y comprobá por canales oficiales."
    else:
        return "Muy alto", "No hagas clic ni ingreses credenciales; reportalo."

# --- Helper: genera pasos de color con degradado suave verde→amarillo→rojo
def make_gradient_steps(n: int = 80, vmin: float = 0.0, vmax: float = 100.0):
    """
    Devuelve una lista de steps para el gauge de Plotly, usando un degradado
    continuo de verde (#21c55d) → amarillo (#facc15) → rojo (#ef4444).
    """
    # Stops en espacio RGB
    stops = [
        (0.00, (33, 197, 93)),   # #21c55d  verde
        (0.50, (250, 204, 21)),  # #facc15  amarillo
        (1.00, (239, 68, 68)),   # #ef4444  rojo
    ]

    def lerp(a, b, t):  # interpolación lineal
        return a + (b - a) * t

    def interp_color(p: float):
        # p en [0,1]
        for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
            if p <= p1:
                t = (p - p0) / (p1 - p0)
                r = int(lerp(c0[0], c1[0], t))
                g = int(lerp(c0[1], c1[1], t))
                b = int(lerp(c0[2], c1[2], t))
                return f"rgb({r},{g},{b})"
        r, g, b = stops[-1][1]
        return f"rgb({r},{g},{b})"

    steps = []
    span = vmax - vmin
    for i in range(n):
        a = vmin + (i * span) / n
        b = vmin + ((i + 1) * span) / n
        p_mid = (i + 0.5) / n  # punto medio del tramo
        steps.append({"range": [a, b], "color": interp_color(p_mid)})
    return steps


def render_tacometro_dashboard(prob: float, title: str = "Phishing Risk"):
    """
    Tacómetro semicircular estilo dashboard con degradado suave verde→rojo.
    Usa 'threshold' como aguja y oculta la barra para un look similar al velocímetro.
    """
    import plotly.graph_objects as go

    pct = round(prob * 100, 1)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={
                "suffix": "%",
                "font": {"size": 44, "color": "#101418", "family": "Arial Black"},
            },
            title={"text": "", "font": {"size": 1}},
            gauge={
                "shape": "angular",  # semicircular
                "axis": {
                    "range": [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 20, 40, 60, 80, 100],
                    "ticktext": ["0", "20", "40", "60", "80", "100"],
                    "tickwidth": 0,
                    "ticks": "",
                },
                "bar": {"color": "rgba(0,0,0,0)"},  # sin barra: solo aguja
                "threshold": {
                    "line": {"color": "#111", "width": 6},  # “aguja”
                    "thickness": 0.9,
                    "value": pct,
                },
                "borderwidth": 0,
                "bgcolor": "rgba(0,0,0,0)",
                # <<< Degradado real con muchos pasos >>>
                "steps": make_gradient_steps(n=80, vmin=0, vmax=100),
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=0),
        height=300,
        font=dict(color="#101418", family="Arial"),
        transition={"duration": 500, "easing": "cubic-in-out"},
    )

    # Card con tu CSS existente (gauge-card)
    st.markdown(f"""
    <div class="gauge-card">
      <div class="gauge-title">{title}</div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("""
      <div class="gauge-subtitle">Probabilidad estimada de phishing</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# Lógica principal
# ---------------------------------------------------------------
def predict_and_show(dominio: str):
    # Extracción de features
    with st.spinner("Extrayendo features (WHOIS + scraping)…"):
        base_feats = procesar_dominio_basico(dominio)
        dyn_feats = enriquecer_dominio_scraping(dominio)
        feats = {**(base_feats or {}), **(dyn_feats or {})}

    # Orden esperado
    expected_order = get_expected_order(feature_order, scaler, model, list(feats.keys()))
    scaler_in_use = scaler
    expected_numeric = [c for c in expected_order if c in NUMERIC_FEATURES]

    # Columnas del scaler
    if hasattr(scaler_in_use, "feature_names_in_") and getattr(scaler_in_use, "feature_names_in_", None) is not None:
        scaler_cols = [str(c) for c in scaler_in_use.feature_names_in_]
    else:
        scaler_cols = expected_numeric

    # Info de debug ligera
    extra_in_scaler = [c for c in scaler_cols if c not in expected_numeric]
    if extra_in_scaler:
        st.info("Se omiten en la salida del scaler columnas no usadas por el modelo: " + ", ".join(extra_in_scaler))

    # Mostrar diferencias de columnas
    render_diffs(feats, expected_order)

    # Alinear vector
    X = ensure_feature_vector(feats, expected_order)

    # Aplicar scaler SOLO a columnas que coinciden
    X_in = X.copy()
    scaler_input = pd.DataFrame(columns=scaler_cols)
    for c in scaler_cols:
        if c in X.columns:
            scaler_input[c] = X[c].values
        else:
            scaler_input[c] = 0.0
    try:
        scaled_array = scaler_in_use.transform(scaler_input[scaler_cols])
    except Exception as se:
        st.error(f"Error aplicando el scaler sobre columnas {scaler_cols}: {se}")
        st.stop()

    overlap_cols = [c for c in scaler_cols if c in expected_numeric]
    if overlap_cols:
        idx_map = [scaler_cols.index(c) for c in overlap_cols]
        X_in.loc[:, overlap_cols] = scaled_array[:, idx_map]
    else:
        st.warning("No hay columnas numéricas en común entre el scaler y el modelo; se continúa sin modificar X.")

    # Predicción
    with st.spinner("Prediciendo…"):
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_in)
        y_pred = model.predict(X_in)

    label = int(y_pred[0]) if hasattr(y_pred, "__iter__") else int(y_pred)
    if proba is not None and np.ndim(proba) == 2 and proba.shape[1] >= 2:
        p_phishing = float(proba[0, 1])
    else:
        p_phishing = 1.0 if label == 1 else 0.0

    # === Mostrar resultado con tacómetro "dashboard" + meta ===
    nivel, recomendacion = riesgo_bucket(p_phishing)

    render_tacometro_dashboard(p_phishing, title="Phishing Risk")

    colA, colB = st.columns([1, 1])
    with colA:
        if label == 1:
            st.error(f"**Resultado:** Posible PHISHING")
        else:
            st.success(f"**Resultado:** No phishing")
        st.markdown(f"**Probabilidad (clase 1):** {p_phishing:.3f}")
    with colB:
        st.markdown(f"**Nivel de riesgo:** {nivel}")
        st.caption(recomendacion)

    # (Opcional) detalles técnicos
    with st.expander("Ver vector de features alineado"):
        st.dataframe(X.T.rename(columns={0: "valor"}))
        if hasattr(scaler_in_use, "feature_names_in_") and getattr(scaler_in_use, "feature_names_in_", None) is not None:
            st.caption("Columnas esperadas por el scaler (deberían coincidir con las numéricas):")
            st.code(", ".join(list(scaler_in_use.feature_names_in_)))

# ---------------------------------------------------------------
# Evento de análisis
# ---------------------------------------------------------------
if analizar:
    if not entrada.strip():
        st.warning("Ingresá una URL o dominio válido.")
    elif procesar_dominio_basico is None or enriquecer_dominio_scraping is None:
        st.error("No se pudieron importar las funciones de features. Revisá `features.py`.")
    elif model is None:
        st.error("No se pudo cargar el modelo.")
    else:
        try:
            dominio = normalize_to_domain(entrada)
            if not dominio:
                st.warning("No se pudo interpretar la entrada como dominio/URL.")
            else:
                predict_and_show(dominio)
        except Exception as e:  # pragma: no cover
            st.exception(e)

# ---------------------------------------------------------------
# Descargo de responsabilidad permanente
# ---------------------------------------------------------------
st.divider()
st.markdown(
    """
**Descargo de responsabilidad**  
Esta herramienta ofrece una **estimación automática** basada en un modelo de Machine Learning y **no garantiza** la legitimidad o ilicitud de un sitio.  
No reemplaza tu criterio ni verifica la identidad de ninguna organización.  
Ante dudas, **no ingreses datos sensibles** (contraseñas, tarjetas) y verificá por **canales oficiales**.
"""
)

# Pie informativo
st.caption(
    "Modelo: XGBoost ya entrenado. Ajustá rutas y asegurate de que `features.py` exponga las funciones requeridas."
)
