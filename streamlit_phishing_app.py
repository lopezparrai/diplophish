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
except Exception:
    xgb = None

from urllib.parse import urlparse
import re
import time
import ssl
import socket
import requests
from bs4 import BeautifulSoup
from dateutil import parser
from datetime import datetime
import tldextract
import whois

# =============================================================
# 🧩 CONFIGURACIÓN
# =============================================================
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_phishing_model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/feature_order.json")  # orden exacto de features del modelo
APP_TITLE = "Detección de Phishing por URL (XGBoost)"
DESCRIPTION = (
    "Ingresa una URL. Se extraen características, se aplica el modelo XGBoost y se estima la probabilidad de phishing."
)
DEFAULT_THRESHOLD = 0.50

# =============================================================
# 🧪 UTILIDADES
# =============================================================
def _looks_like_url(s: str) -> bool:
    if not s:
        return False
    pattern = r"^(https?://)?([\w.-]+)\.([a-zA-Z]{2,})(/.*)?$"
    return re.match(pattern, s.strip()) is not None

def _normalize_url(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if not s.startswith(("http://", "https://")):
        s = "http://" + s
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
    try:
        model = load_model()
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass
    return []

# =============================================================
# 🧱 EXTRACCIÓN / ENRIQUECIMIENTO DE FEATURES (tus funciones integradas)
# =============================================================

# Stub simple por si no tenés un clasificador de categorías del título
def clasificar_categoria(texto: str) -> str:
    texto = (texto or "").lower()
    if any(k in texto for k in ["bank", "banco", "login", "account", "verify", "update"]):
        return "credenciales/pagos"
    if any(k in texto for k in ["shop", "store", "oferta", "sale", "promo"]):
        return "ecommerce"
    return "general"

def procesar_dominio_basico(dominio: str) -> dict:
    """
    Procesa información básica y WHOIS de un dominio, generando las features correspondientes.
    (Tu función, con guardas menores)
    """
    url = f"http://{dominio}"
    parsed = urlparse(url)
    hostname = parsed.hostname or dominio

    # --- Métricas estáticas de la URL ---
    url_length = len(url)
    num_dashes = dominio.count('-')
    num_digits = sum(c.isdigit() for c in dominio)
    num_special_chars = len(re.findall(r'[^\w\s:/.-]', dominio))
    num_dots = dominio.count('.')
    num_underscores = dominio.count('_')
    num_dashes_in_hostname = hostname.count('-')
    double_slash_in_path = '//' in (parsed.path or '')

    path_segments = len((parsed.path or '').strip('/').split('/')) if parsed.path else 0
    hostname_length = len(hostname)
    path_length = len(parsed.path or "")
    query_length = len(parsed.query or "")

    # --- Extraer TLD ---
    try:
        ext = tldextract.extract(dominio)
        tld = ext.suffix
    except Exception:
        tld = None

    # --- WHOIS ---
    creation_date_iso = None
    expiration_date_iso = None
    registrar = None
    country_registered = None
    site_age_years = None
    time_to_expire_years = None
    registration_time = None

    try:
        socket.setdefaulttimeout(5)
        info_whois = whois.whois(dominio)

        creation = info_whois.creation_date
        expiration = info_whois.expiration_date

        if isinstance(creation, list):
            creation = creation[0]
        if isinstance(expiration, list):
            expiration = expiration[0]

        if isinstance(creation, str):
            try:
                creation = parser.parse(creation)
            except Exception:
                creation = None
        if isinstance(expiration, str):
            try:
                expiration = parser.parse(expiration)
            except Exception:
                expiration = None

        creation_date_iso = creation.isoformat() if creation else None
        expiration_date_iso = expiration.isoformat() if expiration else None

        if creation:
            site_age_years = round((datetime.now() - creation).days / 365, 2)
        if expiration:
            time_to_expire_years = round((expiration - datetime.now()).days / 365, 2)
        if creation and expiration:
            registration_time = round((expiration - creation).days / 365, 2)

        registrar = getattr(info_whois, 'registrar', None)
        country_registered = (
            getattr(info_whois, "country", None)
            or getattr(info_whois, "registrant_country", None)
            or "Desconocido"
        )
    except Exception:
        pass

    is_registered_in_ar = bool(
        country_registered and isinstance(country_registered, str) and "argentina" in country_registered.lower()
    )

    return {
        # Identificación
        "url": url,
        "tld": tld,
        "is_phishing": None,

        # Estructura URL
        "url_length": url_length,
        "num_dashes": num_dashes,
        "num_digits": num_digits,
        "num_special_chars": num_special_chars,
        "path_segments": path_segments,
        "num_dots": num_dots,
        "num_underscores": num_underscores,
        "num_dashes_in_hostname": num_dashes_in_hostname,
        "double_slash_in_path": double_slash_in_path,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "query_length": query_length,

        # WHOIS y temporalidad
        "registration_time": registration_time,
        "creation_date": creation_date_iso,
        "expiration_date": expiration_date_iso,
        "site_age_years": site_age_years,
        "time_to_expire_years": time_to_expire_years,
        "registrar": registrar,
        "country_registered": country_registered,
        "is_registered_in_ar": is_registered_in_ar,
    }

def enriquecer_dominio_scraping(dominio: str) -> dict:
    """
    Obtiene datos dinámicos del sitio mediante requests y análisis HTML.
    (Tu función, con guardas menores)
    """
    esquemas = ["https", "http"]
    titulo = ""
    tiempo_respuesta = None
    responde = False
    codigo_estado = None
    url_redireccionada = None
    tiene_https = False
    tiene_ssl = False
    meta_keywords = ""
    iframe_present = False
    insecure_forms = False
    submit_info_to_email = False
    abnormal_form_action = False
    html_text = ""

    url = None
    for esquema in esquemas:
        try:
            url = f"{esquema}://{dominio}"
            inicio = time.time()
            r = requests.get(url, timeout=6, allow_redirects=True, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/117.0 Safari/537.36"
            })
            fin = time.time()

            tiempo_respuesta = round(fin - inicio, 3)
            responde = True
            codigo_estado = getattr(r, 'status_code', None)
            url_redireccionada = getattr(r, 'url', None)
            html_text = getattr(r, 'text', '')

            soup = BeautifulSoup(html_text, "html.parser")
            titulo_tag = soup.find("title")
            if titulo_tag:
                titulo = titulo_tag.text.strip()
            if esquema == "https":
                tiene_https = True
            break
        except requests.exceptions.RequestException:
            continue

    soup = BeautifulSoup(html_text, "html.parser") if html_text else None

    # --- Title y meta keywords ---
    longitud_titulo = len(titulo or "")
    if soup:
        meta_tag = soup.find("meta", attrs={"name": "keywords"})
        if meta_tag and "content" in meta_tag.attrs:
            meta_keywords = str(meta_tag["content"]).lower()

    # --- SSL check ---
    if tiene_https:
        try:
            hostname = urlparse(url).hostname
            context = ssl.create_default_context()
            with socket.create_connection((hostname, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    tiene_ssl = bool(ssock.getpeercert())
        except Exception:
            pass

    # --- Analizar HTML para iframes y forms ---
    if soup:
        iframe_present = bool(soup.find("iframe"))
        forms = soup.find_all("form")
        for form in forms:
            action = (form.get("action") or "").lower()
            if not action or action.startswith("http://"):
                insecure_forms = True
            if "mailto:" in action:
                submit_info_to_email = True
            if action.startswith("http"):
                try:
                    action_host = urlparse(action).hostname
                    if action_host and not action_host.endswith(dominio):
                        abnormal_form_action = True
                except Exception:
                    pass

    # --- Indicadores engañosos ---
    sensitive_words = ["login", "secure", "account", "bank", "verify", "update"]
    sensitive_words_count = sum((titulo or "").lower().count(word) for word in sensitive_words)
    https_in_hostname = "https" in (dominio or "").lower()
    random_string = bool(re.findall(r"[a-z]{5,}\d{3,}|[0-9]{5,}", (dominio or "").lower()))

    bancos = [
        "santander", "bbva", "galicia", "banco nación", "bna", "hipotecario", "provincia",
        "macro", "comafi", "brubank", "itau", "supervielle", "patagonia"
    ]
    entidades_publicas = [
        "afip", "anses", "anmat", "ministerio", "gobierno", "municipio", "secretaria", "renaper",
        "dgr", "dine", "senado", "diputados", "presidencia"
    ]
    redes_sociales = [
        "facebook", "instagram", "twitter", "whatsapp", "tiktok", "linkedin", "telegram", "snapchat"
    ]
    empresas_y_servicios = [
        "mercadolibre", "mercadopago", "despegar", "globant", "tenaris", "ypf", "rappi", "pedidosya",
        "todopago", "naranja", "ripley", "uala", "plin", "cencosud"
    ]
    marcas_populares = [*bancos, *entidades_publicas, *redes_sociales, *empresas_y_servicios]
    embedded_brand_name = any(brand in (dominio or '').lower() for brand in marcas_populares)

    parsed_url = urlparse(url_redireccionada or url or "")
    base_domain = (dominio or "").split(".")[0]
    domain_in_subdomains = bool(parsed_url.hostname and base_domain in parsed_url.hostname and parsed_url.hostname != dominio)
    domain_in_paths = bool(base_domain and base_domain in (parsed_url.path or ""))

    categoria = clasificar_categoria(titulo or dominio)

    return {
        # Seguridad
        "has_https": tiene_https,
        "has_ssl_cert": tiene_ssl,
        "iframe_present": iframe_present,
        "insecure_forms": insecure_forms,
        "submit_info_to_email": submit_info_to_email,
        "abnormal_form_action": abnormal_form_action,

        # Respuesta servidor
        "response_time": tiempo_respuesta,
        "responds": responde,
        "http_status_code": codigo_estado,
        "redirected_url": url_redireccionada,

        # Contenido
        "title": titulo,
        "title_length": longitud_titulo,
        "meta_keywords": meta_keywords,
        "category": categoria,

        # Indicadores engañosos
        "random_string": random_string,
        "sensitive_words_count": sensitive_words_count,
        "embedded_brand_name": embedded_brand_name,
        "https_in_hostname": https_in_hostname,
        "domain_in_subdomains": domain_in_subdomains,
        "domain_in_paths": domain_in_paths,
    }

# ---- Helpers para ensamblar vector numérico final ----
def _domain_from_url(url: str) -> str:
    p = urlparse(url)
    host = p.hostname or p.netloc
    if not host:
        host = url.replace("http://", "").replace("https://", "").split("/")[0]
    return host

def _to_numeric_features(d: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        if v is None:
            continue
        if isinstance(v, bool):
            out[k] = 1.0 if v else 0.0
        elif isinstance(v, (int, float, np.number)):
            out[k] = float(v)
    return out

@st.cache_data(show_spinner=False)
def extract_features(url: str) -> Dict[str, float]:
    """Usa tus dos funciones y devuelve SOLO numéricas (bool→0/1) con los nombres esperados por el modelo."""
    if not url:
        return {}
    dominio = _domain_from_url(url)
    base = procesar_dominio_basico(dominio)
    dyn  = enriquecer_dominio_scraping(dominio)

    merged  = {**(base or {}), **(dyn or {})}
    feats   = _to_numeric_features(merged)

    # Señales derivadas útiles
    if "has_https" in feats and "has_ssl_cert" in feats:
        feats.setdefault("https_no_cert_flag", float(feats["has_https"] == 1.0 and feats["has_ssl_cert"] == 0.0))
    if "response_time" in feats and feats["response_time"] is not None:
        feats.setdefault("slow_response_flag", float(feats["response_time"] > 2.0))

    # Eliminar placeholders antiguos
    for p in ["host_length", "count_digits", "count_dashes", "count_dots", "has_at", "subdirs"]:
        feats.pop(p, None)

    # Asegurar claves esperadas (si faltan, quedan en 0.0)
    expected_like = [
        # Estructura URL
        "url_length", "num_dashes", "num_digits", "num_special_chars", "path_segments",
        "num_dots", "hostname_length", "query_length", "num_underscores", "num_dashes_in_hostname",
        "double_slash_in_path", "path_length",
        # Dinámica/HTML/seguridad
        "title_length", "http_status_code", "has_ssl_cert", "iframe_present", "insecure_forms",
        "submit_info_to_email", "abnormal_form_action", "has_https", "responds",
        # (agrego dos señales derivadas comunes; si tu modelo no las usa, no estarán en feature_order.json)
        "response_time", "https_no_cert_flag"
    ]
    for k in expected_like:
        feats.setdefault(k, 0.0)

    return feats

@st.cache_data(show_spinner=False)
def extract_features_cached(url: str) -> Dict[str, float]:
    return extract_features(url)

def _align_features(row_feat: Dict[str, float], feature_order: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Alinea el dict de features al orden esperado por el modelo. Rellena faltantes con 0."""
    if not feature_order:
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

    if hasattr(model, "n_features_in_") and len(used_cols) != int(model.n_features_in_):
        raise ValueError(
            f"Número de features desalineado: modelo espera {getattr(model, 'n_features_in_', '?')} y se recibieron {len(used_cols)}.\n"
            f"Revisa 'feature_order.json' y que 'extract_features' esté generando esos nombres."
        )

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0, 1])
    else:
        dmat = xgb.DMatrix(X, feature_names=used_cols) if xgb else X
        proba = float(model.predict(dmat)[0])

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

                    # Diferencias vs modelo
                    with st.expander("Ver diferencias de columnas (esperadas vs presentes)"):
                        feats = extract_features_cached(url_norm)
                        expected = load_feature_order()
                        present  = list(feats.keys())
                        missing  = [c for c in expected if c not in present]
                        extra    = [c for c in present  if c not in expected]
                        st.write(f"Features presentes: {len(present)} / Esperadas por el modelo: {len(expected)}")
                        st.markdown("**Faltan en la URL (se imputan 0):** " + (", ".join(missing) if missing else "—"))
                        st.markdown("**Sobran (no usadas por el modelo):** " + (", ".join(extra) if extra else "—"))

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
# 📝 NOTAS
# =============================================================
with st.expander("Notas de integración"):
    st.markdown(
        """
        1. `extract_features(url)` ahora usa tus dos funciones y devuelve **solo numéricas** con nombres esperados.
        2. Exportá a `models/feature_order.json` el **orden exacto** de columnas del entrenamiento (22 en tu caso).
        3. Las faltantes se imputan a 0 en el alineado; el expander muestra qué falta/sobra para depurar.
        4. Ejecutá: `streamlit run streamlit_phishing_app.py`.
        """
    )

st.sidebar.caption("Hecho con ❤️ para DiploDatos · Ignacio")
