# features.py
from __future__ import annotations

import re
import ssl
import time
import socket
import requests
import tldextract
import whois
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import parser
from urllib.parse import urlparse
from functools import lru_cache
from typing import Dict, Tuple, Optional

# -------- Config de red --------
DEFAULT_TIMEOUT = 6.0
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (phishing-checker)"}


def _safe_get(url: str, timeout: float = DEFAULT_TIMEOUT) -> Optional[requests.Response]:
    """GET con headers y timeouts; None si hay error."""
    try:
        return requests.get(url, timeout=timeout, allow_redirects=True, headers=DEFAULT_HEADERS)
    except requests.exceptions.RequestException:
        return None


@lru_cache(maxsize=256)
def resolve_canonical_url(dominio: str) -> Tuple[Optional[str], int, bool, bool]:
    """
    Devuelve (final_url, status_code, has_https, responds)
    Prueba variantes comunes (prioriza HTTPS) y sigue redirects.
    Unifica 'apex' y 'www' para evitar divergencias de features.
    """
    d = (dominio or "").strip().lower()
    if d.startswith("www."):
        d = d[4:]

    candidates = [
        f"https://{d}",
        f"http://{d}",
        f"https://www.{d}",
        f"http://www.{d}",
    ]

    # 1) Devolver la primera 2xx/3xx
    for u in candidates:
        r = _safe_get(u)
        if r and (200 <= r.status_code < 400):
            final_url = str(r.url)
            return final_url, int(r.status_code), final_url.startswith("https://"), True

    # 2) Si nada 2xx/3xx, quedarnos con "lo mejor" que respondió
    best = None
    for u in candidates:
        r = _safe_get(u)
        if r and (best is None or r.status_code < best.status_code):
            best = r
    if best:
        final_url = str(best.url)
        return final_url, int(best.status_code), final_url.startswith("https://"), False

    return None, 0, False, False


# ============================================================
# ===============  Features BASE (sin red)  ==================
# ============================================================
def procesar_dominio_basico(dominio: str) -> Dict[str, float]:
    """
    Procesa información estática (sintaxis de URL) y WHOIS del dominio.
    No hace requests HTML (para eso está enriquecer_dominio_scraping).
    """
    # Normalizar a host
    url = f"http://{dominio}"
    parsed = urlparse(url)
    hostname = parsed.hostname or dominio

    # --- Métricas sintácticas de la URL/host ---
    url_length = len(url)
    num_dashes = dominio.count('-')
    num_digits = sum(c.isdigit() for c in dominio)
    num_special_chars = len(re.findall(r'[^\w\s:/.-]', dominio))
    num_dots = dominio.count('.')
    num_underscores = dominio.count('_')
    num_dashes_in_hostname = hostname.count('-')
    hostname_length = len(hostname)

    # Como no estamos resolviendo aquí, path/query vendrán vacíos
    path = parsed.path or ""
    query = parsed.query or ""
    path_segments = len([p for p in path.strip('/').split('/') if p]) if path else 0
    path_length = len(path)
    query_length = len(query)
    double_slash_in_path = bool(re.search(r"//", path)) if path else False

    # --- TLD ---
    ext = tldextract.extract(dominio)
    tld = ext.suffix

    # --- WHOIS (con timeouts) ---
    creation_date_iso = None
    expiration_date_iso = None
    registrar = None
    country_registered = None
    site_age_years = None
    time_to_expire_years = None
    registration_time = None

    try:
        socket.setdefaulttimeout(5)  # evitar bloqueos en WHOIS
        info_whois = whois.whois(dominio)

        creation = info_whois.creation_date
        expiration = info_whois.expiration_date

        if isinstance(creation, list) and creation:
            creation = creation[0]
        if isinstance(expiration, list) and expiration:
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

        registrar = getattr(info_whois, "registrar", None)
        # Algunos whois devuelven dict-like
        try:
            country_registered = (
                info_whois.get("country")
                or info_whois.get("registrant_country")
                or getattr(info_whois, "country", None)
                or "Desconocido"
            )
        except Exception:
            country_registered = getattr(info_whois, "country", "Desconocido")
    except Exception:
        pass

    is_registered_in_ar = bool(country_registered and "argentina" in str(country_registered).lower())

    return {
        # Identificación (pueden no estar en feature_order; no molesta)
        "url": url,
        "tld": tld,

        # Estructura URL/host
        "url_length": float(url_length),
        "num_dashes": float(num_dashes),
        "num_digits": float(num_digits),
        "num_special_chars": float(num_special_chars),
        "path_segments": float(path_segments),
        "num_dots": float(num_dots),
        "num_underscores": float(num_underscores),
        "num_dashes_in_hostname": float(num_dashes_in_hostname),
        "double_slash_in_path": 1.0 if double_slash_in_path else 0.0,
        "hostname_length": float(hostname_length),
        "path_length": float(path_length),
        "query_length": float(query_length),

        # WHOIS y temporalidad (si tu scaler/modelo no los usa, no pasa nada)
        "registration_time": float(registration_time) if registration_time is not None else None,
        "creation_date": creation_date_iso,
        "expiration_date": expiration_date_iso,
        "site_age_years": float(site_age_years) if site_age_years is not None else None,
        "time_to_expire_years": float(time_to_expire_years) if time_to_expire_years is not None else None,
        "registrar": registrar,
        "country_registered": country_registered,
        "is_registered_in_ar": 1.0 if is_registered_in_ar else 0.0,

        # Placeholders que se pisan con scraping
        "title_length": 0.0,
        "http_status_code": 0.0,
        "has_https": 0.0,
        "has_ssl_cert": 0.0,
        "iframe_present": 0.0,
        "insecure_forms": 0.0,
        "submit_info_to_email": 0.0,
        "abnormal_form_action": 0.0,
        "responds": 0.0,
        "response_time": None,
        "sensitive_words_count": 0.0,
    }


# ============================================================
# ===========  Features DINÁMICAS (con red)  =================
# ============================================================
def enriquecer_dominio_scraping(dominio: str) -> Dict[str, float]:
    """
    Usa una URL canónica (apex/www unificados) para medir señales de red/HTML.
    Con esto, 'dominio' y 'www.dominio' producirán features consistentes.
    """
    # 1) Resolver canónica (apex/www) y seguir redirects
    final_url, status_code, has_https, responds = resolve_canonical_url(dominio)

    titulo = ""
    tiempo_respuesta = None
    url_redireccionada = final_url
    tiene_ssl = False
    html_text = ""

    # 2) Descargar el HTML de la URL canónica (si existe)
    if final_url:
        t0 = time.time()
        r = _safe_get(final_url)
        t1 = time.time()
        if r is not None:
            tiempo_respuesta = round(t1 - t0, 3)
            html_text = r.text or ""
            status_code = int(r.status_code)
            url_redireccionada = str(r.url)

    # 3) Chequeo de SSL (sólo si la canónica es HTTPS)
    if final_url and has_https:
        try:
            host = urlparse(final_url).hostname
            if host:
                context = ssl.create_default_context()
                with socket.create_connection((host, 443), timeout=5) as sock:
                    with context.wrap_socket(sock, server_hostname=host) as ssock:
                        tiene_ssl = bool(ssock.getpeercert())
        except Exception:
            pass

    # 4) Parse HTML
    soup = BeautifulSoup(html_text, "html.parser") if html_text else None
    if soup:
        t = soup.find("title")
        if t:
            titulo = (t.text or "").strip()

    longitud_titulo = float(len(titulo))
    parsed_final = urlparse(url_redireccionada or (final_url or ""))
    path = parsed_final.path or ""
    query = parsed_final.query or ""

    # --- Cálculo robusto de subdominios y presencia del dominio en el path ---
    # Usamos el registrable final (no el input) y marcamos subdominio real distinto de "" y "www"
    final_ext = tldextract.extract(parsed_final.netloc or "")
    final_reg = final_ext.registered_domain  # p.ej. "arca.gob.ar"
    final_sub = final_ext.subdomain          # p.ej. "", "www", "mi"
    domain_in_subdomains = 1.0 if final_sub not in ("", None, "www") else 0.0

    # Dominio "base" (label antes del TLD registrable), ej. "arca" en "arca.gob.ar"
    base_label = final_reg.split(".")[0] if final_reg else ""
    domain_in_paths = 1.0 if (base_label and base_label in path) else 0.0

    # 5) Señales HTML y de formularios
    iframe_present = False
    insecure_forms = False
    submit_info_to_email = False
    abnormal_form_action = False

    if soup:
        iframe_present = bool(soup.find("iframe"))
        forms = soup.find_all("form")

        # comparar contra el registrable del HOST FINAL (evita falsos con www/apex)
        base_reg = final_reg
        for form in forms:
            action = (form.get("action") or "").strip().lower()

            # action vacío/fragmento → inseguro
            if not action or action.startswith("#") or action.startswith("?"):
                insecure_forms = True
                continue

            if "mailto:" in action:
                submit_info_to_email = True

            if not action.startswith(("http://", "https://")):
                # relativo: inseguro si el sitio NO es https
                if not has_https:
                    insecure_forms = True
            else:
                # absoluto http en sitio https → inseguro
                if has_https and action.startswith("http://"):
                    insecure_forms = True
                # acción a otro registrable → abnormal
                try:
                    a_host = urlparse(action).hostname or ""
                    a_reg = tldextract.extract(a_host).registered_domain
                    if a_reg and base_reg and a_reg != base_reg:
                        abnormal_form_action = True
                except Exception:
                    abnormal_form_action = True

    # 6) Indicadores engañosos (en el título / dominio)
    sensitive_words = [
        "login", "secure", "account", "bank", "verify", "update",
        "contraseña", "tarjeta", "seguridad", "verificación"
    ]
    sensitive_words_count = sum((titulo or "").lower().count(w) for w in sensitive_words)

    dominio_lower = (dominio or "").lower()
    bancos = [
        "santander", "bbva", "galicia", "banco nación", "bna", "hipotecario", "provincia",
        "macro", "comafi", "brubank", "itau", "supervielle", "patagonia"
    ]
    entidades_publicas = [
        "afip", "anses", "anmat", "ministerio", "gobierno", "municipio", "secretaria", "renaper",
        "dgr", "dine", "senado", "diputados", "presidencia", "arca"
    ]
    redes_sociales = ["facebook", "instagram", "twitter", "whatsapp", "tiktok", "linkedin", "telegram", "snapchat"]
    empresas_y_servicios = [
        "mercadolibre", "mercadopago", "despegar", "globant", "tenaris", "ypf",
        "rappi", "pedidosya", "todopago", "naranja", "uala", "plin", "cencosud"
    ]
    marcas_populares = bancos + entidades_publicas + redes_sociales + empresas_y_servicios
    embedded_brand_name = any(brand in dominio_lower for brand in marcas_populares)

    # evitar flags engañosos
    https_in_hostname = 0.0
    random_string = 1.0 if re.search(r"[a-z]{5,}\d{3,}|[0-9]{5,}", dominio_lower) else 0.0

    categoria = clasificar_categoria(titulo or dominio)

    return {
        # Seguridad
        "has_https": 1.0 if has_https else 0.0,
        "has_ssl_cert": 1.0 if tiene_ssl else 0.0,
        "iframe_present": 1.0 if iframe_present else 0.0,
        "insecure_forms": 1.0 if insecure_forms else 0.0,
        "submit_info_to_email": 1.0 if submit_info_to_email else 0.0,
        "abnormal_form_action": 1.0 if abnormal_form_action else 0.0,

        # Respuesta servidor
        "response_time": float(tiempo_respuesta) if tiempo_respuesta is not None else None,
        "responds": 1.0 if responds else 0.0,
        "http_status_code": float(status_code or 0),
        "redirected_url": url_redireccionada,

        # Contenido
        "title": titulo,
        "title_length": longitud_titulo,
        "meta_keywords": "",  # opcional
        "category": categoria,

        # Indicadores engañosos
        "random_string": random_string,
        "sensitive_words_count": float(sensitive_words_count),
        "embedded_brand_name": 1.0 if embedded_brand_name else 0.0,
        "https_in_hostname": https_in_hostname,
        "domain_in_subdomains": domain_in_subdomains,
        "domain_in_paths": domain_in_paths,

        # Longitudes derivadas de canónica
        "path_segments": float(len([p for p in path.split('/') if p])),
        "path_length": float(len(path)),
        "query_length": float(len(query)),
        "double_slash_in_path": 1.0 if ("//" in path and not path.startswith("//")) else 0.0,
    }

# ============================================================
# =============  Clasificación temática simple  ==============
# ============================================================
def clasificar_categoria(texto_o_dominio: str) -> str:
    """
    Clasifica en categorías amplias según palabras clave en título/dominio.
    """
    d = (texto_o_dominio or "").lower()

    if any(x in d for x in ["news", "noticia", "diario", "prensa", "periodico", "press"]):
        return "noticias"
    if any(x in d for x in ["gob", "gov", "municipio", "ministerio", "provincia", ".gob.", ".gov."]):
        return "gobierno"
    if any(x in d for x in ["banco", "bank", "finance", "finanzas", "credito", "loan", "tarjeta"]):
        return "banca"
    if any(x in d for x in ["shop", "store", "tienda", "ecommerce", "comprar", "venta", "oferta"]):
        return "e-commerce"
    if any(x in d for x in ["edu", "universidad", "facultad", "campus", "colegio", "escuela", "instituto"]):
        return "educacion"
    return "otro"
