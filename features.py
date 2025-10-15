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

def procesar_dominio_basico(dominio: str) -> dict:
    """
    Procesa información básica y WHOIS de un dominio, generando las features correspondientes.
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
    double_slash_in_path = '//' in parsed.path if parsed.path else False

    path_segments = len(parsed.path.strip('/').split('/')) if parsed.path else 0
    hostname_length = len(hostname)
    path_length = len(parsed.path)
    query_length = len(parsed.query)

    # --- Extraer TLD ---
    ext = tldextract.extract(dominio)
    tld = ext.suffix

    # --- WHOIS ---
    creation_date_iso = None
    expiration_date_iso = None
    registrar = None
    country_registered = None
    site_age_years = None
    time_to_expire_years = None
    registration_time = None

    try:
        socket.setdefaulttimeout(5)  # evitar bloqueos WHOIS
        info_whois = whois.whois(dominio)

        # Normalizar fechas
        creation = info_whois.creation_date
        expiration = info_whois.expiration_date

        if isinstance(creation, list):
            creation = creation[0]
        if isinstance(expiration, list):
            expiration = expiration[0]

        # Convertir strings a datetime
        if isinstance(creation, str):
            try:
                creation = parser.parse(creation)
            except:
                creation = None
        if isinstance(expiration, str):
            try:
                expiration = parser.parse(expiration)
            except:
                expiration = None

        # Guardar en ISO si disponibles
        creation_date_iso = creation.isoformat() if creation else None
        expiration_date_iso = expiration.isoformat() if expiration else None

        # Calcular métricas temporales
        if creation:
            site_age_years = round((datetime.now() - creation).days / 365, 2)
        if expiration:
            time_to_expire_years = round((expiration - datetime.now()).days / 365, 2)
        if creation and expiration:
            registration_time = round((expiration - creation).days / 365, 2)

        registrar = info_whois.registrar
        country_registered = (
            info_whois.get("country") or
            info_whois.get("registrant_country") or
            "Desconocido"
        )
    except Exception:
        pass

    # --- Flag si está registrado en Argentina ---
    is_registered_in_ar = bool(country_registered and "argentina" in country_registered.lower())

    return {
        # Identificación
        "url": url,
        "tld": tld,
        "is_phishing": None,  # Placeholder para etiquetado posterior

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
        "is_registered_in_ar": is_registered_in_ar
    }



def enriquecer_dominio_scraping(dominio: str) -> dict:
    """
    Obtiene datos dinámicos del sitio mediante requests y análisis HTML.
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
            codigo_estado = r.status_code
            url_redireccionada = r.url
            html_text = r.text

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
    longitud_titulo = len(titulo)
    if soup:
        meta_tag = soup.find("meta", attrs={"name": "keywords"})
        if meta_tag and "content" in meta_tag.attrs:
            meta_keywords = meta_tag["content"].lower()

    # --- SSL check ---
    if tiene_https:
        try:
            hostname = urlparse(url).hostname
            context = ssl.create_default_context()
            with socket.create_connection((hostname, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    tiene_ssl = bool(ssock.getpeercert())
        except:
            pass

    # --- Analizar HTML para iframes y forms ---
    if soup:
        iframe_present = bool(soup.find("iframe"))
        forms = soup.find_all("form")
        for form in forms:
            action = form.get("action", "").lower()
            if not action or action.startswith("http://"):
                insecure_forms = True
            if "mailto:" in action:
                submit_info_to_email = True
            # Acción que apunta a otro dominio
            if action.startswith("http"):
                action_host = urlparse(action).hostname
                if action_host and not action_host.endswith(dominio):
                    abnormal_form_action = True

    # --- Indicadores engañosos ---
    sensitive_words = ["login", "secure", "account", "bank", "verify", "update"]
    sensitive_words_count = sum(titulo.lower().count(word) for word in sensitive_words)
    https_in_hostname = "https" in dominio
    random_string = bool(re.search(r"[a-z]{5,}\d{3,}|[0-9]{5,}", dominio))

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
        "todoPago", "naranja", "ripley", "uala", "plin", "cencosud"
    ]
    # Unificar listas y normalizar dominio
    marcas_populares = bancos + entidades_publicas + redes_sociales + empresas_y_servicios
    dominio_lower = dominio.lower()

    embedded_brand_name = any(brand.lower() in dominio_lower for brand in marcas_populares)

    domain_in_subdomains = False
    domain_in_paths = False
    parsed_url = urlparse(url_redireccionada or url)
    base_domain = dominio.split(".")[0]
    if base_domain in parsed_url.netloc and parsed_url.netloc != dominio:
        domain_in_subdomains = True
    if base_domain in parsed_url.path:
        domain_in_paths = True

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
        "domain_in_paths": domain_in_paths
    }

def clasificar_categoria(dominio):
    """
    Clasifica un dominio web en una categoría temática general según palabras clave presentes.

    Esta función busca términos genéricos en el nombre del dominio para asignarle una categoría
    temática amplia como "noticias", "gobierno", "banca", "e-commerce" o "educacion". Si no
    encuentra coincidencias, clasifica el dominio como "otro".

    Args:
        dominio (str): Nombre de dominio (por ejemplo, "noticiasargentinas.com.ar").

    Returns:
        str: Categoría general a la que pertenece el dominio. Puede ser:
            - "noticias"
            - "gobierno"
            - "banca"
            - "e-commerce"
            - "educacion"
            - "otro"
    """
    dominio = dominio.lower()

    if any(x in dominio for x in ["news", "noticia", "diario", "prensa", "periodico", "press"]):
        return "noticias"
    elif any(x in dominio for x in ["gob", "gov", "municipio", "ministerio", "provincia", ".gob.", ".gov."]):
        return "gobierno"
    elif any(x in dominio for x in ["banco", "bank", "finance", "finanzas", "credito", "loan", "tarjeta"]):
        return "banca"
    elif any(x in dominio for x in ["shop", "store", "tienda", "ecommerce", "comprar", "venta", "oferta"]):
        return "e-commerce"
    elif any(x in dominio for x in ["edu", "universidad", "facultad", "campus", "colegio", "escuela", "instituto"]):
        return "educacion"
    else:
        return "otro"
