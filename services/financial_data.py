import pandas as pd
import yfinance as yf




# ============================================================
# BÚSQUEDA DE EMPRESAS
# ============================================================

# FINSCOPE · UNIVERSAL SEARCH RANKING V4

def buscar_empresas(consulta, max_resultados=8):
    """
    FINSCOPE · UNIVERSAL SEARCH FINAL V7

    Estrategia:
    - Yahoo Search aporta descubrimiento y relevancia.
    - Un ticker exacto escrito por el usuario se respeta.
    - Se distingue la identidad de una empresa de compañías cuyo
      nombre solamente contiene la marca buscada.
    - Se penalizan mercados claramente secundarios/OTC.
    - No se fuerza artificialmente una bolsa doméstica cuando la
      consulta es también un ticker válido.
    """

    import re
    import unicodedata
    import yfinance as yf

    consulta = str(consulta or "").strip()

    if not consulta:
        return []

    try:
        max_resultados = max(1, int(max_resultados or 8))
    except Exception:
        max_resultados = 8

    def normalizar(valor):
        valor = str(valor or "").strip().lower()

        valor = "".join(
            c for c in unicodedata.normalize("NFKD", valor)
            if not unicodedata.combining(c)
        )

        valor = re.sub(r"[^a-z0-9]+", " ", valor)

        return " ".join(valor.split())

    consulta_norm = normalizar(consulta)
    consulta_upper = consulta.upper()

    # ---------------------------------------------------------
    # ALIAS EMPRESARIAL YA EXISTENTE
    # ---------------------------------------------------------

    aliases = {
        "inditex": "ITX.MC",
    }

    ticker_alias = aliases.get(consulta_norm)

    # ---------------------------------------------------------
    # SUFIJOS JURÍDICOS
    # ---------------------------------------------------------

    sufijos = {
        "ag",
        "aktiengesellschaft",
        "se",
        "sa",
        "nv",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "ltd",
        "limited",
        "plc",
        "spa",
        "group",
        "holding",
        "holdings",
        "llc",
        "lp",
        "sas",
        "a",
        "s",
    }

    def tokens_identidad(nombre):
        tokens = normalizar(nombre).split()

        return [
            token
            for token in tokens
            if token not in sufijos
        ]

    query_tokens = tokens_identidad(consulta)

    # ---------------------------------------------------------
    # MERCADOS SECUNDARIOS
    # ---------------------------------------------------------

    penalizacion_exchange = {
        "PNK": -4500,
        "OID": -4500,
        "OQX": -3500,
        "NEO": -1800,

        "FRA": -1000,
        "MUN": -1000,
        "STU": -1000,
        "DUS": -1000,
        "BER": -1000,
        "HAM": -1000,
        "HAN": -1000,
        "TLO": -1000,
        "LTS": -1000,

        "BUD": -900,
    }

    # ---------------------------------------------------------
    # YAHOO
    # ---------------------------------------------------------

    try:
        search = yf.Search(
            consulta,
            max_results=max(20, max_resultados + 12),
            news_count=0,
        )

        quotes = list(search.quotes or [])

    except Exception:
        return []

    candidatos = []

    for posicion, item in enumerate(quotes):

        quote_type = str(
            item.get("quoteType") or ""
        ).upper()

        if quote_type != "EQUITY":
            continue

        ticker = str(
            item.get("symbol") or ""
        ).strip()

        if not ticker:
            continue

        nombre = (
            item.get("longname")
            or item.get("shortname")
            or ticker
        )

        exchange = str(
            item.get("exchange") or ""
        ).upper()

        try:
            yahoo_score = float(
                item.get("score") or 0
            )
        except Exception:
            yahoo_score = 0.0

        nombre_tokens = tokens_identidad(nombre)

        # =====================================================
        # 1. BASE: RELEVANCIA YAHOO
        # =====================================================

        # Para búsquedas normales no necesitamos magnitudes
        # enormes. Conservamos el orden relativo de Yahoo.
        if yahoo_score >= 1_000_000:
            base_yahoo = 5000
        else:
            base_yahoo = min(
                5000,
                yahoo_score / 5.0,
            )

        score = base_yahoo

        # La posición original sigue siendo una señal fuerte.
        score += max(
            0,
            5000 - posicion * 500,
        )

        # =====================================================
        # 2. TICKER EXACTO
        # =====================================================

        if ticker.upper() == consulta_upper:
            score += 1_000_000

        # =====================================================
        # 3. ALIAS EXPLÍCITO
        # =====================================================

        if (
            ticker_alias
            and ticker.upper() == ticker_alias.upper()
        ):
            score += 2_000_000

        # =====================================================
        # 4. IDENTIDAD SEMÁNTICA
        # =====================================================

        if query_tokens and nombre_tokens:

            query_set = set(query_tokens)
            nombre_set = set(nombre_tokens)

            comunes = query_set & nombre_set

            cobertura = (
                len(comunes) / len(query_set)
                if query_set
                else 0
            )

            score += cobertura * 3500

            # Identidad limpia:
            #
            # Puma -> PUMA SE
            # Siemens -> Siemens AG
            # Adidas -> adidas AG
            #
            # frente a:
            #
            # Puma Biotechnology
            # Siemens Energy
            # Siemens Healthineers

            if nombre_tokens == query_tokens:
                score += 6500

            elif (
                len(query_tokens) == 1
                and query_tokens[0] in nombre_tokens
            ):
                extras = [
                    token
                    for token in nombre_tokens
                    if token != query_tokens[0]
                ]

                if not extras:
                    score += 6000
                else:
                    score -= min(
                        5000,
                        len(extras) * 1800,
                    )

            # Coincidencia inicial de nombre.
            nombre_norm = normalizar(nombre)

            if nombre_norm.startswith(consulta_norm + " "):
                score += 800

        # =====================================================
        # 5. RAÍZ DEL TICKER
        # =====================================================

        ticker_root = re.split(
            r"[.\-=]",
            ticker.upper(),
            maxsplit=1,
        )[0]

        if (
            consulta_upper == ticker_root
            and ticker.upper() != consulta_upper
        ):
            score += 1800

        # =====================================================
        # 6. CALIDAD DE MERCADO
        # =====================================================

        score += penalizacion_exchange.get(
            exchange,
            0,
        )

        texto = " ".join(
            [
                str(item.get("shortname") or ""),
                str(item.get("longname") or ""),
            ]
        ).lower()

        if (
            " adr" in f" {texto}"
            or "(adr" in texto
            or "depositary receipt" in texto
        ):
            score -= 2500

        candidatos.append(
            {
                "ticker": ticker,
                "nombre": nombre,
                "exchange": exchange,
                "_score": score,
                "_posicion": posicion,
            }
        )

    candidatos.sort(
        key=lambda x: (
            x["_score"],
            -x["_posicion"],
        ),
        reverse=True,
    )

    vistos = set()
    resultado = []

    for item in candidatos:

        ticker = item["ticker"]

        if ticker in vistos:
            continue

        vistos.add(ticker)

        resultado.append(
            {
                "ticker": ticker,
                "nombre": item["nombre"],
                "exchange": item["exchange"],
            }
        )

        if len(resultado) >= max_resultados:
            break

    return resultado


def buscar_activos(consulta, max_resultados=8):

    # FINSCOPE · resolución de marcas inequívocas
    _consulta_marca = str(consulta or "").strip().lower()
    _marcas_principales = {
        "adidas": "ADS.DE",
        "bmw": "BMW.DE",
    }
    if _consulta_marca in _marcas_principales:
        consulta = _marcas_principales[_consulta_marca]

    """
    Buscador universal de FinScope.

    Combina:
    - buscador especializado de empresas;
    - búsqueda multi-activo de Yahoo Finance;
    - aliases de instrumentos de referencia;
    - aliases de cotizaciones principales verificadas.

    Las alternativas continúan disponibles en el selector.
    """

    consulta = str(consulta or "").strip()

    if not consulta:
        return []

    def normalizar(texto):
        texto = str(texto or "").casefold()

        reemplazos = {
            "&": " and ",
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "ü": "u",
            "ñ": "n",
            "-": " ",
            "_": " ",
            "/": " ",
            ".": " ",
        }

        for origen, destino in reemplazos.items():
            texto = texto.replace(origen, destino)

        return " ".join(texto.split())

    consulta_norm = normalizar(consulta)
    consulta_upper = consulta.upper()

    # ------------------------------------------------------------
    # INSTRUMENTOS DE REFERENCIA
    # ------------------------------------------------------------

    aliases_referencia = {
        "s&p 500": "^GSPC",
        "s&p500": "^GSPC",
        "sp 500": "^GSPC",
        "sp500": "^GSPC",
        "s and p 500": "^GSPC",

        "nasdaq 100": "^NDX",
        "nasdaq100": "^NDX",

        "msci world": "^990100-USD-STRD",

        "bitcoin": "BTC-USD",
        "btc": "BTC-USD",

        "ethereum": "ETH-USD",
        "ether": "ETH-USD",
        "eth": "ETH-USD",

        "gold": "GC=F",
        "oro": "GC=F",

        "silver": "SI=F",
        "plata": "SI=F",
    }

    # Cotizaciones principales que FinScope ya ha verificado
    # durante las pruebas del buscador empresarial.
    aliases_empresa_principal = {
        "inditex": "ITX.MC",
    }

    alias_referencia = aliases_referencia.get(
        consulta_norm
    )

    alias_empresa = aliases_empresa_principal.get(
        consulta_norm
    )

    ticker_preferido = (
        alias_referencia
        or alias_empresa
    )

    mapa_tipos = {
        "EQUITY": "EMPRESA",
        "ETF": "ETF",
        "MUTUALFUND": "FONDO",
        "INDEX": "INDICE",
        "CRYPTOCURRENCY": "CRIPTOMONEDA",
        "FUTURE": "FUTURO",
    }

    nombres_tipos = {
        "EMPRESA": "Empresa",
        "ETF": "ETF",
        "FONDO": "Fondo",
        "INDICE": "Índice",
        "CRIPTOMONEDA": "Criptomoneda",
        "FUTURO": "Futuro",
    }

    try:
        candidatos = []
        vistos = set()

        # ========================================================
        # EMPRESAS
        # ========================================================

        empresas = buscar_empresas(
            consulta,
            max_resultados=max(12, max_resultados),
        )

        for posicion, empresa in enumerate(empresas):
            simbolo = empresa["ticker"]
            clave = simbolo.upper()

            if clave in vistos:
                continue

            nombre = empresa["nombre"]
            nombre_norm = normalizar(nombre)

            score = max(
                0,
                500 - posicion,
            )

            if clave == consulta_upper:
                score += 20000

            if nombre_norm == consulta_norm:
                score += 2500

            elif nombre_norm.startswith(
                consulta_norm
            ):
                score += 800

            elif consulta_norm in nombre_norm:
                score += 350

            # Cotización principal verificada.
            if (
                alias_empresa
                and clave == alias_empresa.upper()
            ):
                score += 30000

            # Si existe un instrumento de referencia no empresarial,
            # las empresas homónimas quedan por detrás.
            if (
                alias_referencia
                and clave != alias_referencia.upper()
            ):
                score -= 3000

            candidatos.append(
                {
                    "ticker": simbolo,
                    "nombre": nombre,
                    "exchange": empresa["exchange"],
                    "tipo": "EMPRESA",
                    "tipo_nombre": "Empresa",
                    "quote_type_yahoo": "EQUITY",
                    "_score": score,
                    "_posicion": posicion,
                }
            )

            vistos.add(clave)

        # ========================================================
        # YAHOO MULTI-ACTIVO
        # ========================================================

        consultas_yahoo = []

        if ticker_preferido:
            consultas_yahoo.append(
                ticker_preferido
            )

        consultas_yahoo.append(
            consulta
        )

        posicion_global = 0

        for consulta_yahoo in consultas_yahoo:
            busqueda = yf.Search(
                consulta_yahoo,
                max_results=max(
                    40,
                    max_resultados * 6,
                ),
                news_count=0,
            )

            for item in busqueda.quotes or []:
                simbolo = str(
                    item.get("symbol") or ""
                ).strip()

                if not simbolo:
                    continue

                clave = simbolo.upper()

                quote_type = str(
                    item.get("quoteType") or ""
                ).upper()

                tipo = mapa_tipos.get(
                    quote_type
                )

                if not tipo:
                    continue

                # Las empresas encontradas por Yahoo también participan
                # en el ranking universal. Esto es importante para
                # búsquedas internacionales por nombre (por ejemplo,
                # cuando Yahoo conoce la empresa pero el buscador
                # empresarial no devuelve su cotización principal).
                #
                # Los duplicados siguen eliminándose mediante `vistos`
                # y el ranking posterior decide qué resultado priorizar.
                if clave in vistos:
                    continue

                nombre = (
                    item.get("longname")
                    or item.get("shortname")
                    or simbolo
                )

                exchange = (
                    item.get("exchange")
                    or item.get("exchDisp")
                    or "Mercado no disponible"
                )

                nombre_norm = normalizar(nombre)

                score = max(
                    0,
                    500 - posicion_global,
                )

                if clave == consulta_upper:
                    score += 20000

                if (
                    ticker_preferido
                    and clave == ticker_preferido.upper()
                ):
                    score += 30000

                if nombre_norm == consulta_norm:
                    score += 2500

                elif nombre_norm.startswith(
                    consulta_norm + " "
                ):
                    score += 800

                elif consulta_norm in nombre_norm:
                    score += 350

                palabras = set(
                    nombre_norm.split()
                )

                if palabras & {
                    "ultra",
                    "ultrapro",
                    "short",
                    "inverse",
                    "leveraged",
                    "bear",
                    "bull",
                    "2x",
                    "3x",
                }:
                    score -= 1000

                candidatos.append(
                    {
                        "ticker": simbolo,
                        "nombre": nombre,
                        "exchange": exchange,
                        "tipo": tipo,
                        "tipo_nombre": nombres_tipos[tipo],
                        "quote_type_yahoo": quote_type,
                        "_score": score,
                        "_posicion": posicion_global,
                    }
                )

                vistos.add(clave)
                posicion_global += 1

        candidatos.sort(
            key=lambda x: (
                x["_score"],
                -x["_posicion"],
            ),
            reverse=True,
        )

        salida = []

        for item in candidatos[:max_resultados]:
            limpio = dict(item)
            limpio.pop("_score", None)
            limpio.pop("_posicion", None)
            salida.append(limpio)

        return salida

    except Exception:
        return []


# ============================================================
# METADATOS DE MÉTRICAS
# ============================================================

METRICAS_INFO = {

    # ─────────────────────────────
    # MERCADO
    # ─────────────────────────────

    "precio": {
        "nombre": "Precio",
        "campo_fuente": "currentPrice",
        "tipo": "mercado",
        "descripcion": (
            "Precio de mercado disponible para la acción."
        ),
        "limitacion": (
            "Puede no representar una cotización en tiempo real "
            "y su disponibilidad depende del mercado."
        ),
    },

    "capitalizacion": {
        "nombre": "Capitalización bursátil",
        "campo_fuente": "marketCap",
        "tipo": "mercado",
        "descripcion": (
            "Valor de mercado aproximado de las acciones "
            "de la compañía."
        ),
        "limitacion": (
            "Cambia con el precio de mercado y con el número "
            "de acciones utilizado por la fuente."
        ),
    },

    "minimo_52_semanas": {
        "nombre": "Mínimo 52 semanas",
        "campo_fuente": "fiftyTwoWeekLow",
        "tipo": "mercado",
        "descripcion": (
            "Precio mínimo registrado durante aproximadamente "
            "las últimas 52 semanas."
        ),
        "limitacion": (
            "Es una referencia histórica y no determina "
            "por sí misma el valor de la empresa."
        ),
    },

    "maximo_52_semanas": {
        "nombre": "Máximo 52 semanas",
        "campo_fuente": "fiftyTwoWeekHigh",
        "tipo": "mercado",
        "descripcion": (
            "Precio máximo registrado durante aproximadamente "
            "las últimas 52 semanas."
        ),
        "limitacion": (
            "Es una referencia histórica y no determina "
            "por sí misma el valor de la empresa."
        ),
    },


    # ─────────────────────────────
    # RESULTADOS FINANCIEROS
    # ─────────────────────────────

    "ingresos": {
        "nombre": "Ingresos",
        "campo_fuente": "totalRevenue",
        "tipo": "fundamental",
        "descripcion": (
            "Ingresos totales disponibles en la fuente."
        ),
        "limitacion": (
            "Debe comprobarse el periodo contable al que "
            "corresponde antes de realizar comparaciones."
        ),
    },

    "beneficio_neto": {
        "nombre": "Beneficio neto",
        "campo_fuente": "netIncomeToCommon",
        "tipo": "fundamental",
        "descripcion": (
            "Beneficio neto atribuible a los accionistas "
            "ordinarios según la fuente."
        ),
        "limitacion": (
            "Puede verse afectado por partidas extraordinarias "
            "y otros efectos contables."
        ),
    },

    "ebitda": {
        "nombre": "EBITDA",
        "campo_fuente": "ebitda",
        "tipo": "fundamental",
        "descripcion": (
            "Medida de resultados antes de intereses, "
            "impuestos, depreciación y amortización."
        ),
        "limitacion": (
            "No equivale a flujo de caja y su definición "
            "puede variar entre fuentes o compañías."
        ),
    },

    "free_cash_flow": {
        "nombre": "Free Cash Flow",
        "campo_fuente": "freeCashflow",
        "tipo": "fundamental",
        "descripcion": (
            "Flujo de caja libre proporcionado por "
            "la fuente de datos."
        ),
        "limitacion": (
            "Un único periodo positivo o negativo no permite "
            "evaluar por sí solo la salud financiera."
        ),
    },


    # ─────────────────────────────
    # MÁRGENES
    # ─────────────────────────────

    "margen_bruto": {
        "nombre": "Margen bruto",
        "campo_fuente": "grossMargins",
        "tipo": "ratio",
        "descripcion": (
            "Proporción de ingresos restante después "
            "del coste directo asociado a las ventas."
        ),
        "limitacion": (
            "Su nivel normal varía considerablemente "
            "entre sectores y modelos de negocio."
        ),
    },

    "margen_operativo": {
        "nombre": "Margen operativo",
        "campo_fuente": "operatingMargins",
        "tipo": "ratio",
        "descripcion": (
            "Proporción de los ingresos representada "
            "por el resultado operativo."
        ),
        "limitacion": (
            "Debe compararse con compañías y periodos "
            "económicamente comparables."
        ),
    },

    "margen_neto": {
        "nombre": "Margen neto",
        "campo_fuente": "profitMargins",
        "tipo": "ratio",
        "descripcion": (
            "Proporción aproximada de ingresos que termina "
            "como beneficio neto."
        ),
        "limitacion": (
            "Puede estar afectado por impuestos, financiación "
            "y partidas no recurrentes."
        ),
    },


    # ─────────────────────────────
    # RENTABILIDAD
    # ─────────────────────────────

    "roe": {
        "nombre": "ROE",
        "campo_fuente": "returnOnEquity",
        "tipo": "ratio",
        "descripcion": (
            "Relación entre el beneficio y el patrimonio "
            "de los accionistas."
        ),
        "limitacion": (
            "Un ROE elevado puede estar influido por deuda, "
            "recompras o un patrimonio contable reducido."
        ),
    },

    "roa": {
        "nombre": "ROA",
        "campo_fuente": "returnOnAssets",
        "tipo": "ratio",
        "descripcion": (
            "Relaciona los resultados obtenidos con "
            "los activos utilizados por la empresa."
        ),
        "limitacion": (
            "Las diferencias de intensidad de activos hacen "
            "que la comparación entre sectores sea limitada."
        ),
    },


    # ─────────────────────────────
    # VALORACIÓN
    # ─────────────────────────────

    "per": {
        "nombre": "PER",
        "campo_fuente": "trailingPE",
        "tipo": "ratio_valoracion",
        "descripcion": (
            "Relación entre el precio de mercado y "
            "el beneficio por acción utilizado por la fuente."
        ),
        "limitacion": (
            "No determina por sí solo si una acción está "
            "barata o cara y puede perder utilidad con "
            "beneficios muy bajos o negativos."
        ),
    },

    "price_to_book": {
        "nombre": "Price / Book",
        "campo_fuente": "priceToBook",
        "tipo": "ratio_valoracion",
        "descripcion": (
            "Relación entre el valor de mercado y "
            "el valor contable del patrimonio."
        ),
        "limitacion": (
            "Su utilidad depende mucho del sector y puede "
            "ser limitada en negocios con muchos intangibles."
        ),
    },

    "ev_ebitda": {
        "nombre": "EV / EBITDA",
        "campo_fuente": "enterpriseToEbitda",
        "tipo": "ratio_valoracion",
        "descripcion": (
            "Relaciona el valor empresarial con el EBITDA."
        ),
        "limitacion": (
            "No incorpora directamente todas las necesidades "
            "de inversión ni sustituye un análisis de caja."
        ),
    },


    # ─────────────────────────────
    # CRECIMIENTO
    # ─────────────────────────────

    "crecimiento_ingresos": {
        "nombre": "Crecimiento de ingresos",
        "campo_fuente": "revenueGrowth",
        "tipo": "crecimiento",
        "descripcion": (
            "Variación de ingresos proporcionada por "
            "la fuente respecto a su periodo de comparación."
        ),
        "limitacion": (
            "Una sola variación no permite establecer una "
            "tendencia; debe analizarse una serie de periodos."
        ),
    },

    "crecimiento_beneficios": {
        "nombre": "Crecimiento de beneficios",
        "campo_fuente": "earningsGrowth",
        "tipo": "crecimiento",
        "descripcion": (
            "Variación de beneficios proporcionada por "
            "la fuente respecto a su periodo de comparación."
        ),
        "limitacion": (
            "Puede ser muy volátil cuando el beneficio del "
            "periodo de referencia es pequeño o excepcional."
        ),
    },

}

def _fecha_desde_timestamp(valor):
    """
    Convierte un timestamp Unix proporcionado por la fuente a YYYY-MM-DD.
    Si el valor no existe o no puede interpretarse, devuelve None.
    """
    if valor is None:
        return None

    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(
            float(valor),
            tz=timezone.utc
        ).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def obtener_datos_empresa(ticker):
    ticker = ticker.upper().strip()

    empresa = yf.Ticker(ticker)

    # Yahoo Finance puede bloquear parcialmente `.info` en algunos
    # entornos cloud (401 / Invalid Crumb). Ese fallo no debe convertir
    # un ticker válido en un activo inexistente.
    try:
        info = empresa.info
        if not isinstance(info, dict):
            info = {}
    except Exception:
        info = {}

    # Identidad de respaldo mediante el buscador.
    identidad_fallback = {}
    if not info.get("longName"):
        try:
            candidatos = buscar_empresas(ticker, max_resultados=6)
            identidad_fallback = next(
                (
                    item for item in candidatos
                    if str(item.get("ticker") or "").strip().upper() == ticker
                ),
                {},
            )
        except Exception:
            identidad_fallback = {}

    # Mercado de respaldo mediante fast_info.
    fast_fallback = {}
    try:
        fast = empresa.fast_info

        def _fast(nombre):
            try:
                valor = getattr(fast, nombre, None)
                return valor() if callable(valor) else valor
            except Exception:
                return None

        fast_fallback = {
            "last_price": _fast("last_price"),
            "market_cap": _fast("market_cap"),
            "year_high": _fast("year_high"),
            "year_low": _fast("year_low"),
            "currency": _fast("currency"),
        }
    except Exception:
        fast_fallback = {}

    # ---------------------------------------------------------
    # FUNDAMENTALES DE RESPALDO PARA ENTORNOS CLOUD
    # ---------------------------------------------------------
    #
    # Yahoo puede bloquear `.info` en determinados servidores.
    # En ese caso FinScope intenta reconstruir únicamente métricas
    # verificables a partir de los estados financieros publicados
    # por Yahoo Finance. No se estiman valores ausentes.
    #
    # Este bloque también completa campos concretos que falten en
    # `.info`, aunque el resto de `.info` sí esté disponible.

    fundamentales_fallback = {}

    def _numero_finito(valor):
        try:
            import math
            numero = float(valor)
            return numero if math.isfinite(numero) else None
        except Exception:
            return None

    def _fila_reciente(df, nombres):
        if df is None or getattr(df, "empty", True):
            return None

        for nombre in nombres:
            if nombre not in df.index:
                continue

            try:
                serie = df.loc[nombre]
            except Exception:
                continue

            try:
                for valor in serie:
                    numero = _numero_finito(valor)
                    if numero is not None:
                        return numero
            except Exception:
                numero = _numero_finito(serie)
                if numero is not None:
                    return numero

        return None

    def _suma_ultimos_cuatro(df, nombres):
        if df is None or getattr(df, "empty", True):
            return None

        for nombre in nombres:
            if nombre not in df.index:
                continue

            try:
                serie = df.loc[nombre].dropna().head(4)
            except Exception:
                continue

            if len(serie) != 4:
                continue

            valores = [_numero_finito(v) for v in serie]

            if all(v is not None for v in valores):
                return sum(valores)

        return None

    try:
        income_q = empresa.quarterly_income_stmt
    except Exception:
        income_q = None

    try:
        balance_q = empresa.quarterly_balance_sheet
    except Exception:
        balance_q = None

    try:
        cashflow_q = empresa.quarterly_cashflow
    except Exception:
        cashflow_q = None

    # TTM: solo se calcula cuando existen exactamente cuatro
    # observaciones trimestrales válidas.
    ingresos_ttm_fb = _suma_ultimos_cuatro(
        income_q,
        ["Total Revenue", "Operating Revenue"],
    )

    beneficio_ttm_fb = _suma_ultimos_cuatro(
        income_q,
        [
            "Net Income Common Stockholders",
            "Net Income",
        ],
    )

    ebitda_ttm_fb = _suma_ultimos_cuatro(
        income_q,
        ["EBITDA", "Normalized EBITDA"],
    )

    beneficio_operativo_ttm_fb = _suma_ultimos_cuatro(
        income_q,
        ["Operating Income"],
    )

    beneficio_bruto_ttm_fb = _suma_ultimos_cuatro(
        income_q,
        ["Gross Profit"],
    )

    flujo_operativo_ttm_fb = _suma_ultimos_cuatro(
        cashflow_q,
        [
            "Operating Cash Flow",
            "Total Cash From Operating Activities",
        ],
    )

    deuda_fb = _fila_reciente(
        balance_q,
        ["Total Debt"],
    )

    efectivo_fb = _fila_reciente(
        balance_q,
        [
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents",
            "Cash",
        ],
    )

    patrimonio_fb = _fila_reciente(
        balance_q,
        [
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Stockholder Equity",
        ],
    )

    activos_fb = _fila_reciente(
        balance_q,
        ["Total Assets"],
    )

    pasivo_corriente_fb = _fila_reciente(
        balance_q,
        ["Current Liabilities", "Total Current Liabilities"],
    )

    activo_corriente_fb = _fila_reciente(
        balance_q,
        ["Current Assets", "Total Current Assets"],
    )

    acciones_fb = _fila_reciente(
        balance_q,
        [
            "Ordinary Shares Number",
            "Share Issued",
        ],
    )

    precio_fb = (
        _numero_finito(info.get("currentPrice"))
        or _numero_finito(fast_fallback.get("last_price"))
    )

    capitalizacion_fb = (
        _numero_finito(info.get("marketCap"))
        or _numero_finito(fast_fallback.get("market_cap"))
    )

    if ingresos_ttm_fb is not None:
        fundamentales_fallback["totalRevenue"] = ingresos_ttm_fb

    if beneficio_ttm_fb is not None:
        fundamentales_fallback["netIncomeToCommon"] = beneficio_ttm_fb

    if ebitda_ttm_fb is not None:
        fundamentales_fallback["ebitda"] = ebitda_ttm_fb

    if deuda_fb is not None:
        fundamentales_fallback["totalDebt"] = deuda_fb

    if efectivo_fb is not None:
        fundamentales_fallback["totalCash"] = efectivo_fb

    if (
        activo_corriente_fb is not None
        and pasivo_corriente_fb is not None
        and pasivo_corriente_fb != 0
    ):
        fundamentales_fallback["currentRatio"] = (
            activo_corriente_fb / pasivo_corriente_fb
        )

    if (
        ingresos_ttm_fb is not None
        and ingresos_ttm_fb != 0
    ):
        if beneficio_bruto_ttm_fb is not None:
            fundamentales_fallback["grossMargins"] = (
                beneficio_bruto_ttm_fb / ingresos_ttm_fb
            )

        if beneficio_operativo_ttm_fb is not None:
            fundamentales_fallback["operatingMargins"] = (
                beneficio_operativo_ttm_fb / ingresos_ttm_fb
            )

        if beneficio_ttm_fb is not None:
            fundamentales_fallback["profitMargins"] = (
                beneficio_ttm_fb / ingresos_ttm_fb
            )

    if (
        beneficio_ttm_fb is not None
        and patrimonio_fb is not None
        and patrimonio_fb != 0
    ):
        fundamentales_fallback["returnOnEquity"] = (
            beneficio_ttm_fb / patrimonio_fb
        )

    if (
        beneficio_ttm_fb is not None
        and activos_fb is not None
        and activos_fb != 0
    ):
        fundamentales_fallback["returnOnAssets"] = (
            beneficio_ttm_fb / activos_fb
        )

    eps_fb = None

    if (
        beneficio_ttm_fb is not None
        and acciones_fb is not None
        and acciones_fb > 0
    ):
        eps_fb = beneficio_ttm_fb / acciones_fb
        fundamentales_fallback["trailingEps"] = eps_fb

    if (
        precio_fb is not None
        and eps_fb is not None
        and eps_fb > 0
    ):
        fundamentales_fallback["trailingPE"] = (
            precio_fb / eps_fb
        )

    if (
        capitalizacion_fb is not None
        and patrimonio_fb is not None
        and patrimonio_fb > 0
    ):
        fundamentales_fallback["priceToBook"] = (
            capitalizacion_fb / patrimonio_fb
        )

    if (
        deuda_fb is not None
        and patrimonio_fb is not None
        and patrimonio_fb != 0
    ):
        # Yahoo expresa debtToEquity habitualmente como porcentaje.
        fundamentales_fallback["debtToEquity"] = (
            deuda_fb / patrimonio_fb * 100.0
        )

    if flujo_operativo_ttm_fb is not None:
        fundamentales_fallback[
            "operatingCashflow"
        ] = flujo_operativo_ttm_fb

    # Crecimiento interanual TTM: cuatro trimestres actuales frente
    # a los cuatro inmediatamente anteriores. Solo se publica cuando
    # existen ocho trimestres válidos.
    def _crecimiento_ttm(df, nombres):
        if df is None or getattr(df, "empty", True):
            return None

        for nombre in nombres:
            if nombre not in df.index:
                continue

            try:
                serie = df.loc[nombre].dropna().head(8)
            except Exception:
                continue

            if len(serie) != 8:
                continue

            valores = [_numero_finito(v) for v in serie]

            if not all(v is not None for v in valores):
                continue

            actual = sum(valores[:4])
            anterior = sum(valores[4:8])

            if anterior == 0:
                return None

            return (actual / anterior) - 1.0

        return None

    crecimiento_ingresos_fb = _crecimiento_ttm(
        income_q,
        ["Total Revenue", "Operating Revenue"],
    )

    crecimiento_beneficios_fb = _crecimiento_ttm(
        income_q,
        [
            "Net Income Common Stockholders",
            "Net Income",
        ],
    )

    if crecimiento_ingresos_fb is not None:
        fundamentales_fallback[
            "revenueGrowth"
        ] = crecimiento_ingresos_fb

    if crecimiento_beneficios_fb is not None:
        fundamentales_fallback[
            "earningsGrowth"
        ] = crecimiento_beneficios_fb

    # `.info` sigue siendo la fuente preferida. El respaldo únicamente
    # completa claves ausentes o nulas.
    for clave, valor in fundamentales_fallback.items():
        if info.get(clave) is None and valor is not None:
            info[clave] = valor

    # ---------------------------------------------------------
    # PERIODOS Y FCF VERIFICABLE
    # ---------------------------------------------------------
    # totalRevenue, netIncomeToCommon y EBITDA se mantienen como
    # datos proporcionados por Yahoo Finance. FinScope conserva
    # además información del periodo para no presentarlos como
    # pertenecientes a un ejercicio concreto sin indicarlo.
    #
    # Para el Free Cash Flow TTM usamos, cuando están disponibles,
    # los cuatro últimos trimestres del estado de flujos de caja.

    ultimo_ejercicio_fiscal_timestamp = info.get("lastFiscalYearEnd")
    trimestre_mas_reciente_timestamp = info.get("mostRecentQuarter")

    ultimo_ejercicio_fiscal = _fecha_desde_timestamp(
        ultimo_ejercicio_fiscal_timestamp
    )
    trimestre_mas_reciente = _fecha_desde_timestamp(
        trimestre_mas_reciente_timestamp
    )

    es_financiera = info.get("sector") == "Financial Services"

    free_cash_flow_ttm = None
    free_cash_flow_ttm_periodos = []

    # El FCF convencional no se utiliza como métrica analítica para
    # entidades financieras. La estructura de caja de bancos y otras
    # financieras no es comparable con la de una empresa industrial.
    if not es_financiera:
        try:
            cashflow_trimestral = empresa.quarterly_cashflow

            if (
                cashflow_trimestral is not None
                and not cashflow_trimestral.empty
                and "Free Cash Flow" in cashflow_trimestral.index
            ):
                fcf_trimestres = (
                    cashflow_trimestral.loc["Free Cash Flow"]
                    .dropna()
                    .head(4)
                )

                if len(fcf_trimestres) == 4:
                    free_cash_flow_ttm = float(fcf_trimestres.sum())
                    free_cash_flow_ttm_periodos = [
                        str(fecha.date())
                        for fecha in fcf_trimestres.index
                    ]

        except Exception:
            # Si la fuente no proporciona los estados trimestrales,
            # FinScope no inventa ni estima el FCF TTM.
            free_cash_flow_ttm = None
            free_cash_flow_ttm_periodos = []

    # ---------------------------------------------------------
    # DATOS ESPECÍFICOS PARA ENTIDADES FINANCIERAS
    # ---------------------------------------------------------
    # Solo se recuperan de los estados trimestrales cuando Yahoo
    # Finance clasifica la entidad dentro de Financial Services.
    # FinScope no estima valores ausentes.

    datos_bancarios = {
        "net_interest_income": None,
        "interest_income": None,
        "interest_expense": None,
        "net_loans": None,
        "common_stock_equity": None,
        "tangible_book_value": None,
        "total_assets": None,

        # Fechas generales conservadas por compatibilidad.
        "fecha_resultados_bancarios": None,
        "fecha_balance_bancario": None,

        # Fecha real del dato utilizado para cada métrica.
        "fecha_net_interest_income": None,
        "fecha_interest_income": None,
        "fecha_interest_expense": None,
        "fecha_net_loans": None,
        "fecha_common_stock_equity": None,
        "fecha_tangible_book_value": None,
        "fecha_total_assets": None,
    }

    if info.get("sector") == "Financial Services":

        def ultimo_valor_valido(df, fila):
            """
            Devuelve el valor más reciente no nulo de una fila
            y la fecha exacta a la que pertenece.

            No rellena, estima ni sustituye datos ausentes.
            """
            if (
                df is None
                or df.empty
                or fila not in df.index
            ):
                return None, None

            serie = df.loc[fila]

            for fecha, valor in serie.items():
                try:
                    if valor == valor:
                        return float(valor), str(fecha.date())
                except Exception:
                    continue

            return None, None

        # -----------------------------------------------------
        # CUENTA DE RESULTADOS BANCARIA
        # -----------------------------------------------------

        try:
            income_banco = empresa.quarterly_income_stmt

            if (
                income_banco is not None
                and not income_banco.empty
            ):

                fechas_resultados = []

                for campo_yahoo, campo_finscope in [
                    (
                        "Net Interest Income",
                        "net_interest_income",
                    ),
                    (
                        "Interest Income",
                        "interest_income",
                    ),
                    (
                        "Interest Expense",
                        "interest_expense",
                    ),
                ]:

                    valor, fecha = ultimo_valor_valido(
                        income_banco,
                        campo_yahoo,
                    )

                    if valor is not None:
                        datos_bancarios[campo_finscope] = valor

                        clave_fecha = (
                            "fecha_" + campo_finscope
                        )

                        datos_bancarios[clave_fecha] = fecha

                        if fecha is not None:
                            fechas_resultados.append(fecha)

                # Fecha general = periodo más reciente realmente
                # utilizado por alguna métrica de resultados.
                if fechas_resultados:
                    datos_bancarios[
                        "fecha_resultados_bancarios"
                    ] = max(fechas_resultados)

        except Exception:
            pass

        # -----------------------------------------------------
        # BALANCE BANCARIO
        # -----------------------------------------------------

        try:
            balance_banco = empresa.quarterly_balance_sheet

            if (
                balance_banco is not None
                and not balance_banco.empty
            ):

                fechas_balance = []

                for campo_yahoo, campo_finscope in [
                    (
                        "Net Loan",
                        "net_loans",
                    ),
                    (
                        "Common Stock Equity",
                        "common_stock_equity",
                    ),
                    (
                        "Tangible Book Value",
                        "tangible_book_value",
                    ),
                    (
                        "Total Assets",
                        "total_assets",
                    ),
                ]:

                    valor, fecha = ultimo_valor_valido(
                        balance_banco,
                        campo_yahoo,
                    )

                    if valor is not None:
                        datos_bancarios[campo_finscope] = valor

                        clave_fecha = (
                            "fecha_" + campo_finscope
                        )

                        datos_bancarios[clave_fecha] = fecha

                        if fecha is not None:
                            fechas_balance.append(fecha)

                # Fecha general = periodo más reciente realmente
                # utilizado por alguna métrica de balance.
                if fechas_balance:
                    datos_bancarios[
                        "fecha_balance_bancario"
                    ] = max(fechas_balance)

        except Exception:
            pass

    datos = {
        # Identificación
        "nombre": info.get("longName") or identidad_fallback.get("nombre") or ticker,
        "ticker": ticker,
        "sector": info.get("sector"),
        "industria": info.get("industry"),
        "pais": info.get("country"),
        "moneda": info.get("currency") or fast_fallback.get("currency"),

        # Clasificación para adaptar posteriormente el análisis.
        # No cambia los datos financieros: únicamente identifica
        # si Yahoo Finance clasifica la empresa en Financial Services.
        "es_financiera": es_financiera,

        # Mercado
        "precio": info.get("currentPrice") or fast_fallback.get("last_price"),
        "capitalizacion": info.get("marketCap") or fast_fallback.get("market_cap"),
        "maximo_52_semanas": info.get("fiftyTwoWeekHigh") or fast_fallback.get("year_high"),
        "minimo_52_semanas": info.get("fiftyTwoWeekLow") or fast_fallback.get("year_low"),

        # Cuenta de resultados
        "ingresos": info.get("totalRevenue"),
        "beneficio_neto": info.get("netIncomeToCommon"),
        "ebitda": info.get("ebitda"),
        "eps": info.get("trailingEps"),
        "eps_forward": info.get("forwardEps"),

        # Flujo de caja.
        #
        # Guardamos por separado el valor bruto distribuido por Yahoo
        # Finance y la métrica que FinScope permite utilizar en análisis.
        # Para entidades financieras el FCF convencional se marca como
        # no aplicable para evitar interpretaciones engañosas.
        "free_cash_flow_fuente": info.get("freeCashflow"),
        "free_cash_flow": (
            None if es_financiera else info.get("freeCashflow")
        ),
        "flujo_caja_operativo_fuente": info.get("operatingCashflow"),
        "flujo_caja_operativo": (
            None if es_financiera else info.get("operatingCashflow")
        ),
        "fcf_aplicable": not es_financiera,

        # FCF TTM calculado directamente desde los 4 últimos
        # estados trimestrales cuando existen los cuatro periodos.
        "free_cash_flow_ttm": free_cash_flow_ttm,
        "free_cash_flow_ttm_periodos": free_cash_flow_ttm_periodos,

        # Información temporal para transparencia.
        # Se conserva tanto la fecha legible como el timestamp bruto
        # recibido de la fuente.
        "ultimo_ejercicio_fiscal": ultimo_ejercicio_fiscal,
        "ultimo_ejercicio_fiscal_timestamp": ultimo_ejercicio_fiscal_timestamp,
        "trimestre_mas_reciente": trimestre_mas_reciente,
        "trimestre_mas_reciente_timestamp": trimestre_mas_reciente_timestamp,

        # Trazabilidad general.
        "fuente_datos": "Yahoo Finance vía yfinance",
        "fuente_primaria": False,

        # Etiquetas de periodo.
        # Los campos trailing de Yahoo se identifican como TTM cuando
        # corresponden. Los crecimientos conservan la definición de la
        # fuente porque FinScope no presupone un periodo no verificado.
        "periodo_ingresos": "TTM",
        "periodo_beneficio_neto": "TTM",
        "periodo_ebitda": (
            "TTM" if info.get("ebitda") is not None else None
        ),
        "periodo_free_cash_flow_fuente": "Fuente Yahoo Finance",
        "periodo_free_cash_flow_ttm": (
            "TTM calculado desde 4 trimestres"
            if free_cash_flow_ttm is not None
            else None
        ),
        "periodo_crecimiento_ingresos": (
            "TTM interanual calculado desde 8 trimestres"
            if "revenueGrowth" in fundamentales_fallback
            and fundamentales_fallback.get("revenueGrowth") is not None
            and info.get("revenueGrowth") == fundamentales_fallback.get("revenueGrowth")
            else "Periodo definido por la fuente"
        ),
        "periodo_crecimiento_beneficios": (
            "TTM interanual calculado desde 8 trimestres"
            if "earningsGrowth" in fundamentales_fallback
            and fundamentales_fallback.get("earningsGrowth") is not None
            and info.get("earningsGrowth") == fundamentales_fallback.get("earningsGrowth")
            else "Periodo definido por la fuente"
        ),

        # Balance
        "deuda_total": info.get("totalDebt"),
        "efectivo": info.get("totalCash"),
        "current_ratio": info.get("currentRatio"),
        "quick_ratio": info.get("quickRatio"),

        # Márgenes
        "margen_bruto": info.get("grossMargins"),
        "margen_operativo": info.get("operatingMargins"),
        "margen_neto": info.get("profitMargins"),

        # Rentabilidad
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),

        # Valoración
        "per": info.get("trailingPE"),
        "per_forward": info.get("forwardPE"),
        "price_to_book": info.get("priceToBook"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "ev_ventas": info.get("enterpriseToRevenue"),
        "enterprise_value": info.get("enterpriseValue"),

        # Endeudamiento
        "debt_to_equity": info.get("debtToEquity"),

        # Crecimiento
        "crecimiento_ingresos": info.get("revenueGrowth"),
        "crecimiento_beneficios": info.get("earningsGrowth"),
    }

    datos.update(datos_bancarios)

    # ---------------------------------------------------------
    # FINSCOPE · NORMALIZACIÓN MONETARIA GLOBAL A EUR
    # ---------------------------------------------------------
    #
    # FinScope utiliza EUR como divisa base de presentación.
    # Conservamos la moneda y los valores originales para mantener
    # trazabilidad completa del dato recibido desde la fuente.
    #
    # Solo se convierten magnitudes monetarias. Ratios, porcentajes,
    # múltiplos y tasas permanecen sin cambios.

    moneda_original = _normalizar_moneda_finscope(
        info.get("currency") or fast_fallback.get("currency")
    )

    datos["moneda_original"] = moneda_original
    datos["moneda_base"] = "EUR"

    campos_monetarios_eur = [
        # Mercado
        "precio",
        "capitalizacion",
        "maximo_52_semanas",
        "minimo_52_semanas",

        # Cuenta de resultados
        "ingresos",
        "beneficio_neto",
        "ebitda",
        "eps",
        "eps_forward",

        # Flujo de caja
        "free_cash_flow_fuente",
        "free_cash_flow",
        "flujo_caja_operativo_fuente",
        "flujo_caja_operativo",
        "free_cash_flow_ttm",

        # Balance
        "deuda_total",
        "efectivo",

        # Enterprise value
        "enterprise_value",

        # Entidades financieras
        "net_interest_income",
        "interest_income",
        "interest_expense",
        "net_loans",
        "common_stock_equity",
        "tangible_book_value",
        "total_assets",
    ]

    # Copia exacta de los valores monetarios antes de convertirlos.
    datos["valores_monetarios_originales"] = {
        campo: datos.get(campo)
        for campo in campos_monetarios_eur
    }

    fx_eur = obtener_factor_moneda_a_eur(
        moneda_original
    )

    factor_eur = fx_eur.get("factor")

    if (
        fx_eur.get("error") is None
        and factor_eur is not None
    ):
        try:
            factor_eur = float(factor_eur)
        except Exception:
            factor_eur = None

    if factor_eur is not None:
        import math

        if (
            not math.isfinite(factor_eur)
            or factor_eur <= 0
        ):
            factor_eur = None

    datos["factor_moneda_a_eur"] = factor_eur
    datos["ticker_fx_eur"] = fx_eur.get("ticker_fx")
    datos["fecha_fx_eur"] = fx_eur.get("fecha_fx")
    datos["error_conversion_eur"] = fx_eur.get("error")

    if factor_eur is not None:

        for campo in campos_monetarios_eur:
            valor = datos.get(campo)

            if valor is None:
                continue

            try:
                valor = float(valor)
            except Exception:
                continue

            if not math.isfinite(valor):
                continue

            datos[campo] = valor * factor_eur

        # Los campos monetarios expuestos por FinScope pasan a EUR.
        datos["moneda"] = "EUR"
        datos["conversion_eur_aplicada"] = True

    else:
        # Si no existe un FX fiable, FinScope NO etiqueta como EUR
        # un importe que no ha podido convertir.
        datos["conversion_eur_aplicada"] = False

    return datos


def obtener_historico_fundamental(ticker):
    """
    Recupera históricos fundamentales anuales disponibles.

    Empresas no financieras:
    ingresos, beneficio neto, EBITDA, FCF, deuda, caja,
    patrimonio y activos.

    Entidades financieras:
    ingresos, beneficio neto, deuda, caja, patrimonio y activos.
    No utiliza FCF ni EBITDA como métricas analíticas bancarias.

    FinScope no rellena ni estima periodos ausentes.
    """
    ticker = ticker.upper().strip()
    empresa = yf.Ticker(ticker)

    try:
        info = empresa.info
        es_financiera = info.get("sector") == "Financial Services"

        income = empresa.financials
        cashflow = empresa.cashflow
        balance = empresa.balance_sheet

        registros = {}

        def guardar(df, fila, clave):
            if df is None or df.empty or fila not in df.index:
                return

            serie = df.loc[fila]

            for fecha, valor in serie.items():
                try:
                    if valor != valor:
                        continue

                    fecha_txt = str(fecha.date())

                    if fecha_txt not in registros:
                        registros[fecha_txt] = {
                            "fecha": fecha_txt,
                            "anio": int(fecha.year),
                        }

                    registros[fecha_txt][clave] = float(valor)

                except Exception:
                    continue

        # Resultados
        guardar(income, "Total Revenue", "ingresos")

        if "Net Income Common Stockholders" in income.index:
            guardar(
                income,
                "Net Income Common Stockholders",
                "beneficio_neto",
            )
        else:
            guardar(income, "Net Income", "beneficio_neto")

        # EBITDA y FCF solo son métricas analíticas convencionales
        # para compañías no financieras.
        if not es_financiera:
            if "EBITDA" in income.index:
                guardar(income, "EBITDA", "ebitda")
            elif "Normalized EBITDA" in income.index:
                guardar(
                    income,
                    "Normalized EBITDA",
                    "ebitda",
                )

            guardar(
                cashflow,
                "Free Cash Flow",
                "free_cash_flow",
            )

            guardar(
                cashflow,
                "Operating Cash Flow",
                "flujo_caja_operativo",
            )

        # Balance
        guardar(balance, "Total Debt", "deuda_total")

        if (
            "Cash Cash Equivalents And Short Term Investments"
            in balance.index
        ):
            guardar(
                balance,
                "Cash Cash Equivalents And Short Term Investments",
                "efectivo",
            )
        else:
            guardar(
                balance,
                "Cash And Cash Equivalents",
                "efectivo",
            )

        if "Common Stock Equity" in balance.index:
            guardar(
                balance,
                "Common Stock Equity",
                "patrimonio",
            )
        else:
            guardar(
                balance,
                "Stockholders Equity",
                "patrimonio",
            )

        guardar(
            balance,
            "Total Assets",
            "activos_totales",
        )

        # ---------------------------------------------------------
        # PRESENTACIÓN DE FUNDAMENTALES HISTÓRICOS EN EUR
        # ---------------------------------------------------------

        moneda_original = _normalizar_moneda_finscope(
            info.get("financialCurrency")
            or info.get("currency")
        )

        campos_monetarios = {
            "ingresos",
            "beneficio_neto",
            "ebitda",
            "free_cash_flow",
            "flujo_caja_operativo",
            "deuda_total",
            "efectivo",
            "patrimonio",
            "activos_totales",
        }

        resultado = []

        for fecha in sorted(registros):
            fila = registros[fecha].copy()

            if len(fila) <= 2:
                continue

            fila["valores_originales"] = {
                campo: fila.get(campo)
                for campo in campos_monetarios
                if campo in fila
            }

            if moneda_original == "EUR":
                factor_fecha = 1.0

            elif moneda_original:
                serie_base = pd.Series(
                    [1.0],
                    index=pd.DatetimeIndex([
                        pd.Timestamp(fecha)
                    ]),
                    dtype=float,
                )

                serie_fx = convertir_serie_historica_a_eur(
                    serie_base,
                    moneda_original,
                )

                if (
                    serie_fx is None
                    or serie_fx.empty
                ):
                    factor_fecha = None
                else:
                    factor_fecha = float(
                        serie_fx.iloc[0]
                    )

            else:
                factor_fecha = None

            if factor_fecha is not None:
                for campo in campos_monetarios:
                    if fila.get(campo) is None:
                        continue

                    try:
                        fila[campo] = (
                            float(fila[campo])
                            * factor_fecha
                        )
                    except Exception:
                        pass

            fila["factor_eur"] = factor_fecha
            resultado.append(fila)

        return {
            "ticker": ticker,
            "es_financiera": es_financiera,
            "fuente": "Yahoo Finance vía yfinance",
            "frecuencia": "anual",
            "moneda_original": moneda_original,
            "moneda": "EUR",
            "moneda_base": "EUR",
            "conversion_eur_aplicada": bool(
                moneda_original
                and all(
                    fila.get("factor_eur") is not None
                    for fila in resultado
                )
            ),
            "datos": resultado,
        }

    except Exception:
        return {
            "ticker": ticker,
            "es_financiera": None,
            "fuente": "Yahoo Finance vía yfinance",
            "frecuencia": "anual",
            "datos": [],
        }


def _normalizar_indice_fechas(df):
    """
    Devuelve una copia con índice cronológico ordenado.
    Conserva la zona horaria proporcionada por Yahoo Finance.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


# FINSCOPE · MONEDA DE MERCADO DEL ACTIVO
def _obtener_moneda_mercado(ticker):
    """
    Obtiene la moneda de cotización distribuida por Yahoo Finance.

    Se utiliza únicamente para decidir cómo expresar en EUR las
    magnitudes monetarias de mercado de FinScope.
    """
    ticker = str(ticker or "").strip().upper()

    if not ticker:
        return None

    try:
        tk = yf.Ticker(ticker)

        try:
            fast = tk.fast_info
            moneda = getattr(fast, "currency", None)

            if moneda:
                return _normalizar_moneda_finscope(moneda)
        except Exception:
            pass

        try:
            moneda = tk.info.get("currency")

            if moneda:
                return _normalizar_moneda_finscope(moneda)
        except Exception:
            pass

    except Exception:
        pass

    moneda_ticker = _inferir_moneda_desde_ticker(ticker)

    if moneda_ticker:
        return _normalizar_moneda_finscope(moneda_ticker)

    return None


def obtener_historico(ticker, periodo="1mo"):
    """
    Histórico utilizado para representar el gráfico de Cotización.

    Close representa el precio de cierre SIN ajustar por dividendos.
    Por tanto, los cambios calculados a partir de esta serie son
    variaciones del precio y no rentabilidad total para el accionista.
    """
    ticker = ticker.upper().strip()
    empresa = yf.Ticker(ticker)

    # Esta función alimenta exclusivamente el gráfico.
    # El cálculo de variación utiliza su propio histórico ampliado
    # dentro de calcular_variacion_precio().
    configuraciones = {
        "1h": {"period": "1d", "interval": "1m"},
        "1d": {"period": "1d", "interval": "5m"},
        "5d": {"period": "5d", "interval": "30m"},
        "1mo": {"period": "1mo", "interval": "1d"},
        "3mo": {"period": "3mo", "interval": "1d"},
        "6mo": {"period": "6mo", "interval": "1d"},
        "1y": {"period": "1y", "interval": "1d"},
        "2y": {"period": "2y", "interval": "1d"},
        "5y": {"period": "5y", "interval": "1wk"},
        "max": {"period": "max", "interval": "1mo"},
    }

    config = configuraciones.get(periodo, configuraciones["1mo"])

    try:
        historico = empresa.history(
            period=config["period"],
            interval=config["interval"],
            auto_adjust=False,
            actions=False,
        )
    except Exception:
        return None

    historico = _normalizar_indice_fechas(historico)

    if historico is None or historico.empty:
        return historico

    if "Close" not in historico.columns:
        return None

    historico = historico.dropna(subset=["Close"])

    if periodo == "1h":
        historico = historico.tail(60)

    # ---------------------------------------------------------
    # FINSCOPE · COTIZACIÓN VISIBLE EN EUR
    # ---------------------------------------------------------
    moneda_original = _obtener_moneda_mercado(ticker)

    if not moneda_original:
        return None

    close_original = pd.to_numeric(
        historico["Close"],
        errors="coerce",
    ).dropna()

    if close_original.empty:
        return None

    if moneda_original != "EUR":
        close_eur = convertir_serie_historica_a_eur(
            close_original,
            moneda_original,
        )

        if close_eur is None or close_eur.empty:
            return None

        indice = historico.index

        if getattr(indice, "tz", None) is not None:
            indice = indice.tz_localize(None)

        historico = historico.copy()
        historico.index = indice

        historico["Close"] = close_eur.reindex(
            historico.index
        )

        historico = historico.dropna(
            subset=["Close"]
        )

    historico.attrs["moneda_original"] = moneda_original
    historico.attrs["moneda_base"] = "EUR"
    historico.attrs["conversion_eur_aplicada"] = True

    return historico


def calcular_variacion_precio(ticker, periodo):
    """
    Calcula una VARIACIÓN DEL PRECIO, no rentabilidad total.

    Reglas FinScope:
    - 1h: primer y último precio disponible de los últimos 60 minutos.
    - 1d: cierre de la sesión anterior -> último precio disponible.
    - 5d: cierre de 5 sesiones bursátiles atrás -> último precio.
    - 1mo/3mo/6mo/1y/2y/5y:
      último precio -> fecha calendario objetivo; para el inicio se usa
      el último cierre disponible EN O ANTES de esa fecha.
    - max: primer cierre histórico disponible -> último precio.

    Se usa Close sin ajustar por dividendos.
    """
    import pandas as pd

    ticker = ticker.upper().strip()
    empresa = yf.Ticker(ticker)

    periodo = periodo if periodo in {
        "1h", "1d", "5d", "1mo", "3mo",
        "6mo", "1y", "2y", "5y", "max"
    } else "1mo"

    try:
        # ----------------------------------------------------
        # 1 HORA
        # ----------------------------------------------------
        if periodo == "1h":
            df = empresa.history(
                period="1d",
                interval="1m",
                auto_adjust=False,
                actions=False,
            )
            df = _normalizar_indice_fechas(df)

            if df is None or df.empty or "Close" not in df.columns:
                return None

            df = df.dropna(subset=["Close"]).tail(60)

            if len(df) < 2:
                return None

            inicio = df.iloc[0]
            final = df.iloc[-1]
            fecha_inicio = df.index[0]
            fecha_fin = df.index[-1]

        # ----------------------------------------------------
        # PERIODOS BASADOS EN SESIONES
        # ----------------------------------------------------
        elif periodo in {"1d", "5d"}:
            df = empresa.history(
                period="1mo",
                interval="1d",
                auto_adjust=False,
                actions=False,
            )
            df = _normalizar_indice_fechas(df)

            if df is None or df.empty or "Close" not in df.columns:
                return None

            df = df.dropna(subset=["Close"])

            sesiones_atras = 1 if periodo == "1d" else 5

            if len(df) <= sesiones_atras:
                return None

            final = df.iloc[-1]
            inicio = df.iloc[-(sesiones_atras + 1)]
            fecha_fin = df.index[-1]
            fecha_inicio = df.index[-(sesiones_atras + 1)]

        # ----------------------------------------------------
        # MAX
        # ----------------------------------------------------
        elif periodo == "max":
            df = empresa.history(
                period="max",
                interval="1d",
                auto_adjust=False,
                actions=False,
            )
            df = _normalizar_indice_fechas(df)

            if df is None or df.empty or "Close" not in df.columns:
                return None

            df = df.dropna(subset=["Close"])

            if len(df) < 2:
                return None

            inicio = df.iloc[0]
            final = df.iloc[-1]
            fecha_inicio = df.index[0]
            fecha_fin = df.index[-1]

        # ----------------------------------------------------
        # PERIODOS CALENDARIO
        # ----------------------------------------------------
        else:
            meses = {
                "1mo": 1,
                "3mo": 3,
                "6mo": 6,
                "1y": 12,
                "2y": 24,
                "5y": 60,
            }[periodo]

            # Pedimos margen suficiente para poder localizar
            # un cierre anterior a la fecha calendario objetivo.
            periodos_descarga = {
                "1mo": "3mo",
                "3mo": "6mo",
                "6mo": "1y",
                "1y": "2y",
                "2y": "5y",
                "5y": "10y",
            }

            df = empresa.history(
                period=periodos_descarga[periodo],
                interval="1d",
                auto_adjust=False,
                actions=False,
            )
            df = _normalizar_indice_fechas(df)

            if df is None or df.empty or "Close" not in df.columns:
                return None

            df = df.dropna(subset=["Close"])

            if len(df) < 2:
                return None

            fecha_fin = df.index[-1]

            # Timestamp sin timezone para hacer la comparación
            # calendario de forma robusta.
            fecha_fin_naive = pd.Timestamp(fecha_fin)
            if fecha_fin_naive.tzinfo is not None:
                fecha_fin_naive = fecha_fin_naive.tz_localize(None)

            fecha_objetivo = fecha_fin_naive - pd.DateOffset(months=meses)

            indices_naive = pd.DatetimeIndex(df.index)
            if indices_naive.tz is not None:
                indices_naive = indices_naive.tz_localize(None)

            posiciones = [
                i
                for i, fecha in enumerate(indices_naive)
                if fecha <= fecha_objetivo
            ]

            if not posiciones:
                return None

            pos_inicio = posiciones[-1]

            inicio = df.iloc[pos_inicio]
            final = df.iloc[-1]
            fecha_inicio = df.index[pos_inicio]

        precio_inicial = float(inicio["Close"])
        precio_final = float(final["Close"])

        moneda_original = _obtener_moneda_mercado(ticker)

        if not moneda_original:
            return None

        if moneda_original != "EUR":

            if periodo == "1h":
                fx_actual = obtener_factor_moneda_a_eur(
                    moneda_original
                )

                factor = fx_actual.get("factor")

                if factor is None:
                    return None

                factor = float(factor)

                precio_inicial *= factor
                precio_final *= factor

            else:
                serie_conversion = pd.Series(
                    [precio_inicial, precio_final],
                    index=pd.DatetimeIndex([
                        pd.Timestamp(fecha_inicio),
                        pd.Timestamp(fecha_fin),
                    ]),
                    dtype=float,
                )

                if serie_conversion.index.tz is not None:
                    serie_conversion.index = (
                        serie_conversion.index.tz_localize(None)
                    )

                serie_eur = convertir_serie_historica_a_eur(
                    serie_conversion,
                    moneda_original,
                )

                if (
                    serie_eur is None
                    or len(serie_eur) < 2
                ):
                    return None

                precio_inicial = float(
                    serie_eur.iloc[0]
                )
                precio_final = float(
                    serie_eur.iloc[-1]
                )

        if precio_inicial == 0:
            return None

        variacion_abs = precio_final - precio_inicial
        variacion_pct = (
            (precio_final / precio_inicial) - 1
        ) * 100

        def fecha_legible(fecha, incluir_hora=False):
            ts = pd.Timestamp(fecha)

            if incluir_hora:
                return ts.strftime("%d/%m/%Y %H:%M")

            return ts.strftime("%d/%m/%Y")

        return {
            "ticker": ticker,
            "periodo": periodo,
            "precio_inicial": precio_inicial,
            "precio_final": precio_final,
            "variacion_abs": variacion_abs,
            "variacion_pct": variacion_pct,
            "fecha_inicio": fecha_legible(
                fecha_inicio,
                incluir_hora=(periodo == "1h"),
            ),
            "fecha_fin": fecha_legible(
                fecha_fin,
                incluir_hora=(periodo == "1h"),
            ),
            "tipo": "Variación del precio",
            "precio": "Close sin ajustar",
            "incluye_dividendos": False,
            "fuente": "Yahoo Finance vía yfinance",
            "moneda_original": moneda_original,
            "moneda": "EUR",
            "moneda_base": "EUR",
            "conversion_eur_aplicada": True,
        }

    except Exception:
        return None



# ============================================================
# DIVIDENDOS HISTÓRICOS
# ============================================================

def obtener_dividendos_historicos(ticker, anios=10):
    """
    Obtiene y resume el historial de dividendos distribuidos por
    Yahoo Finance vía yfinance.

    Los importes son dividendos por acción en la moneda en la que
    Yahoo distribuye el activo. No se estiman pagos ausentes.
    """

    import pandas as pd

    resultado_vacio = {
        "tiene_dividendos": False,
        "pagos": None,
        "anual": None,
        "dividendo_ttm": None,
        "ultimo_dividendo": None,
        "fecha_ultimo_dividendo": None,
        "pagos_ultimos_12m": 0,
        "frecuencia_aprox": None,
        "crecimiento_1a_pct": None,
        "cagr_3a_pct": None,
        "cagr_5a_pct": None,
        "anios_con_pagos": 0,
        "anios_consecutivos": 0,
        "fuente": "Yahoo Finance vía yfinance",
    }

    try:
        tk = yf.Ticker(ticker)

        dividendos = tk.dividends

        if dividendos is None or dividendos.empty:
            return resultado_vacio

        dividendos = dividendos.copy()
        dividendos = dividendos.dropna()
        dividendos = dividendos[dividendos > 0]

        if dividendos.empty:
            return resultado_vacio

        indice = pd.to_datetime(dividendos.index)

        if getattr(indice, "tz", None) is not None:
            indice = indice.tz_localize(None)

        dividendos.index = indice
        dividendos = dividendos.sort_index()

        moneda_original = _obtener_moneda_mercado(ticker)

        if not moneda_original:
            return resultado_vacio

        if moneda_original != "EUR":
            dividendos_eur = convertir_serie_historica_a_eur(
                dividendos,
                moneda_original,
            )

            if (
                dividendos_eur is None
                or dividendos_eur.empty
            ):
                return resultado_vacio

            dividendos = dividendos_eur

        fecha_fin = dividendos.index.max()
        fecha_inicio = fecha_fin - pd.DateOffset(years=anios)

        pagos = dividendos[
            dividendos.index >= fecha_inicio
        ].copy()

        if pagos.empty:
            pagos = dividendos.copy()

        anual = (
            pagos.groupby(pagos.index.year)
            .sum()
            .astype(float)
        )

        anual.index = anual.index.astype(int)

        fecha_ttm_inicio = fecha_fin - pd.DateOffset(years=1)

        pagos_ttm = dividendos[
            dividendos.index > fecha_ttm_inicio
        ]

        dividendo_ttm = (
            float(pagos_ttm.sum())
            if not pagos_ttm.empty
            else None
        )

        pagos_ultimos_12m = int(len(pagos_ttm))

        frecuencia_aprox = None

        if pagos_ultimos_12m >= 10:
            frecuencia_aprox = "Mensual"
        elif pagos_ultimos_12m >= 4:
            frecuencia_aprox = "Trimestral"
        elif pagos_ultimos_12m == 3:
            frecuencia_aprox = "Aprox. cuatrimestral"
        elif pagos_ultimos_12m == 2:
            frecuencia_aprox = "Semestral"
        elif pagos_ultimos_12m == 1:
            frecuencia_aprox = "Anual"

        # Para medir crecimiento no utilizamos el año natural
        # actual, porque todavía puede estar incompleto.
        anio_actual = pd.Timestamp.now().year

        anual_completo = anual[
            anual.index < anio_actual
        ].copy()

        crecimiento_1a = None

        if len(anual_completo) >= 2:
            actual = float(anual_completo.iloc[-1])
            anterior = float(anual_completo.iloc[-2])

            if anterior > 0:
                crecimiento_1a = (
                    (actual / anterior) - 1
                ) * 100

        def calcular_cagr(serie, periodos):
            if len(serie) < periodos + 1:
                return None

            fin = float(serie.iloc[-1])
            inicio = float(
                serie.iloc[-(periodos + 1)]
            )

            if inicio <= 0 or fin <= 0:
                return None

            return (
                (fin / inicio) ** (1 / periodos) - 1
            ) * 100

        cagr_3a = calcular_cagr(
            anual_completo,
            3,
        )

        cagr_5a = calcular_cagr(
            anual_completo,
            5,
        )

        anios_con_pagos = int(len(anual))

        anios_disponibles = sorted(
            int(x) for x in anual.index
        )

        anios_consecutivos = 0

        if anios_disponibles:
            ultimo = anios_disponibles[-1]
            esperado = ultimo

            for year in reversed(anios_disponibles):
                if year == esperado:
                    anios_consecutivos += 1
                    esperado -= 1
                elif year < esperado:
                    break

        pagos_df = pagos.rename("Dividendo").reset_index()
        pagos_df.columns = ["Fecha", "Dividendo"]

        anual_df = anual.rename(
            "Dividendo anual"
        ).reset_index()

        anual_df.columns = [
            "Año",
            "Dividendo anual",
        ]

        return {
            "tiene_dividendos": True,
            "pagos": pagos_df,
            "anual": anual_df,
            "dividendo_ttm": dividendo_ttm,
            "ultimo_dividendo": (
                float(dividendos.iloc[-1])
                if dividendos is not None and not dividendos.empty
                else None
            ),
            "fecha_ultimo_dividendo":
                dividendos.index[-1],
            "pagos_ultimos_12m":
                pagos_ultimos_12m,
            "frecuencia_aprox":
                frecuencia_aprox,
            "crecimiento_1a_pct":
                crecimiento_1a,
            "cagr_3a_pct": cagr_3a,
            "cagr_5a_pct": cagr_5a,
            "anios_con_pagos":
                anios_con_pagos,
            "anios_consecutivos":
                anios_consecutivos,
            "fuente":
                "Yahoo Finance vía yfinance",
            "moneda_original": moneda_original,
            "moneda": "EUR",
            "moneda_base": "EUR",
            "conversion_eur_aplicada": True,
        }

    except Exception:
        return resultado_vacio


# =============================================================================
# FINSCOPE · MOTOR CENTRAL DE DIVISAS EUR
# =============================================================================
#
# EUR es la divisa base de FinScope.
#
# Convención:
#   factor_moneda_a_eur("USD") = euros que vale 1 USD.
#
# Ejemplo:
#   precio_eur = precio_usd * factor_usd_eur
#
# Para históricos se utiliza un factor FX por fecha, evitando convertir
# toda una serie con el cambio vigente hoy.
# =============================================================================

def _normalizar_moneda_finscope(moneda):
    """Normaliza códigos de moneda usados por FinScope."""
    moneda = str(moneda or "").strip().upper()

    aliases = {
        "GBX": "GBp",
        "GBPENCE": "GBp",
        "PENCE": "GBp",
    }

    return aliases.get(moneda, moneda)


def _inferir_moneda_desde_ticker(ticker):
    """
    Inferencia conservadora de moneda únicamente cuando el propio
    símbolo contiene explícitamente un código monetario.

    Ejemplo:
        ^990100-USD-STRD -> USD

    No se asigna ninguna moneda cuando el ticker no aporta una
    evidencia explícita.
    """
    import re

    ticker = str(ticker or "").strip().upper()

    if not ticker:
        return None

    monedas = (
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CHF",
        "CAD",
        "AUD",
        "NZD",
        "CNY",
        "HKD",
        "SEK",
        "NOK",
        "DKK",
        "SGD",
    )

    partes = [
        parte
        for parte in re.split(r"[^A-Z]+", ticker)
        if parte
    ]

    for moneda in monedas:
        if moneda in partes:
            return moneda

    return None


def _ticker_fx_a_eur(moneda):
    """
    Devuelve el ticker Yahoo necesario para convertir una unidad de
    `moneda` a EUR y si la cotización debe invertirse.

    Yahoo suele expresar pares como:
        EURUSD=X -> USD por 1 EUR
        EURGBP=X -> GBP por 1 EUR

    Por tanto, para USD -> EUR usamos 1 / EURUSD.
    """
    moneda = _normalizar_moneda_finscope(moneda)

    if not moneda:
        return None, False

    if moneda == "EUR":
        return None, False

    if moneda == "GBp":
        # Penique británico: primero se pasa a GBP.
        return "EURGBP=X", True

    return f"EUR{moneda}=X", True


def obtener_factor_moneda_a_eur(moneda):
    """
    Obtiene el último factor disponible para convertir una unidad de
    `moneda` a EUR.

    No se promete tiempo real: se utiliza el último dato disponible
    suministrado por la fuente de mercado.
    """
    import numpy as np
    import yfinance as yf

    moneda = _normalizar_moneda_finscope(moneda)

    if not moneda:
        return {
            "error": "Moneda no especificada.",
            "moneda_origen": moneda,
            "moneda_destino": "EUR",
            "factor": np.nan,
            "ticker_fx": None,
        }

    if moneda == "EUR":
        return {
            "error": None,
            "moneda_origen": "EUR",
            "moneda_destino": "EUR",
            "factor": 1.0,
            "ticker_fx": None,
        }

    ticker_fx, invertir = _ticker_fx_a_eur(moneda)

    try:
        fx = yf.Ticker(ticker_fx)

        hist = fx.history(
            period="5d",
            interval="1d",
            auto_adjust=False,
        )

        if hist is None or hist.empty or "Close" not in hist:
            raise ValueError(
                f"No hay cotización disponible para {ticker_fx}."
            )

        serie = hist["Close"].dropna()

        if serie.empty:
            raise ValueError(
                f"No hay cotización válida para {ticker_fx}."
            )

        cotizacion = float(serie.iloc[-1])

        if not np.isfinite(cotizacion) or cotizacion <= 0:
            raise ValueError(
                f"Cotización FX inválida para {ticker_fx}."
            )

        if invertir:
            factor = 1.0 / cotizacion
        else:
            factor = cotizacion

        # GBp representa peniques, no libras.
        if moneda == "GBp":
            factor = factor / 100.0

        return {
            "error": None,
            "moneda_origen": moneda,
            "moneda_destino": "EUR",
            "factor": float(factor),
            "ticker_fx": ticker_fx,
            "cotizacion_fx": cotizacion,
            "fecha_fx": serie.index[-1],
        }

    except Exception as exc:
        return {
            "error": str(exc),
            "moneda_origen": moneda,
            "moneda_destino": "EUR",
            "factor": np.nan,
            "ticker_fx": ticker_fx,
        }


def obtener_fx_historico_a_eur(
    moneda,
    inicio=None,
    fin=None,
    periodo="max",
):
    """
    Serie histórica del factor moneda -> EUR.

    Cada observación representa cuántos EUR vale una unidad de la
    moneda de origen en esa fecha.
    """
    import numpy as np
    import pandas as pd
    import yfinance as yf

    moneda = _normalizar_moneda_finscope(moneda)

    if moneda == "EUR":
        return pd.Series(
            dtype=float,
            name="EUR",
        )

    ticker_fx, invertir = _ticker_fx_a_eur(moneda)

    try:
        ticker = yf.Ticker(ticker_fx)

        kwargs = {
            "interval": "1d",
            "auto_adjust": False,
        }

        if inicio is not None:
            kwargs["start"] = inicio

            if fin is not None:
                # yfinance trata end como límite exclusivo.
                kwargs["end"] = (
                    pd.Timestamp(fin)
                    + pd.Timedelta(days=1)
                )
        else:
            kwargs["period"] = periodo

        hist = ticker.history(**kwargs)

        if hist is None or hist.empty or "Close" not in hist:
            return pd.Series(
                dtype=float,
                name=f"{moneda}_EUR",
            )

        fx = hist["Close"].dropna().astype(float)

        fx = fx[
            np.isfinite(fx)
            & (fx > 0)
        ]

        if invertir:
            fx = 1.0 / fx

        if moneda == "GBp":
            fx = fx / 100.0

        fx.name = f"{moneda}_EUR"

        return fx

    except Exception:
        return pd.Series(
            dtype=float,
            name=f"{moneda}_EUR",
        )


def convertir_valor_a_eur(valor, moneda):
    """
    Convierte un valor monetario actual a EUR.
    """
    import numpy as np

    try:
        valor = float(valor)
    except Exception:
        return np.nan

    if not np.isfinite(valor):
        return np.nan

    resultado = obtener_factor_moneda_a_eur(moneda)

    if resultado.get("error"):
        return np.nan

    factor = resultado.get("factor")

    if factor is None or not np.isfinite(factor):
        return np.nan

    return float(valor * factor)


def convertir_serie_historica_a_eur(serie, moneda):
    """
    Convierte una serie monetaria histórica a EUR utilizando el
    tipo de cambio disponible para cada fecha.

    Política FinScope:
    - EUR permanece sin cambios.
    - Para otras monedas se descarga FX con margen anterior.
    - Nunca se utiliza un FX futuro para valorar una fecha pasada.
    - Festivos y fines de semana utilizan el último FX anterior
      disponible mediante forward-fill.
    """

    if serie is None:
        return None

    try:
        serie = serie.copy()

        if isinstance(serie, pd.DataFrame):
            if serie.shape[1] != 1:
                return None
            serie = serie.iloc[:, 0]

        serie = pd.to_numeric(
            serie,
            errors="coerce",
        ).dropna()

        if serie.empty:
            return serie

        indice = pd.to_datetime(
            serie.index,
            errors="coerce",
        )

        mascara = ~pd.isna(indice)

        serie = serie.loc[mascara].copy()
        indice = pd.DatetimeIndex(indice[mascara])

        if indice.tz is not None:
            indice = indice.tz_localize(None)

        serie.index = indice

        serie = serie[
            ~serie.index.duplicated(keep="last")
        ].sort_index()

        moneda = _normalizar_moneda_finscope(moneda)

        if not moneda:
            return pd.Series(dtype=float)

        if moneda == "EUR":
            resultado = serie.astype(float)
            resultado.attrs["moneda_original"] = "EUR"
            resultado.attrs["moneda_base"] = "EUR"
            resultado.attrs["conversion_eur_aplicada"] = True
            return resultado

        # Margen anterior para cubrir:
        # - fines de semana
        # - festivos
        # - fechas contables sin sesión FX
        inicio_fx = (
            pd.Timestamp(serie.index.min())
            - pd.Timedelta(days=10)
        )

        fin_fx = (
            pd.Timestamp(serie.index.max())
            + pd.Timedelta(days=1)
        )

        fx = obtener_fx_historico_a_eur(
            moneda,
            inicio=inicio_fx,
            fin=fin_fx,
        )

        if fx is None or len(fx) == 0:
            return pd.Series(dtype=float)

        if isinstance(fx, pd.DataFrame):
            if "factor" in fx.columns:
                fx = fx["factor"]
            elif fx.shape[1] == 1:
                fx = fx.iloc[:, 0]
            else:
                return pd.Series(dtype=float)

        fx = pd.to_numeric(
            fx,
            errors="coerce",
        ).dropna()

        if fx.empty:
            return pd.Series(dtype=float)

        fx.index = pd.to_datetime(
            fx.index,
            errors="coerce",
        )

        fx = fx[
            ~fx.index.isna()
        ]

        if isinstance(fx.index, pd.DatetimeIndex):
            if fx.index.tz is not None:
                fx.index = fx.index.tz_localize(None)

        fx = fx[
            ~fx.index.duplicated(keep="last")
        ].sort_index()

        # Construimos un calendario combinado y solo propagamos
        # información hacia delante: jamás usamos FX futuro.
        indice_combinado = (
            fx.index
            .union(serie.index)
            .sort_values()
        )

        fx_alineado = (
            fx
            .reindex(indice_combinado)
            .ffill()
            .reindex(serie.index)
        )

        valido = (
            serie.notna()
            & fx_alineado.notna()
            & (fx_alineado > 0)
        )

        resultado = (
            serie.loc[valido].astype(float)
            * fx_alineado.loc[valido].astype(float)
        )

        resultado.attrs["moneda_original"] = moneda
        resultado.attrs["moneda_base"] = "EUR"
        resultado.attrs["conversion_eur_aplicada"] = True

        return resultado

    except Exception:
        return pd.Series(dtype=float)
