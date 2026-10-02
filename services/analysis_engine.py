"""
Motor de análisis fundamental de FinScope.

Principios:
- Los datos y su interpretación permanecen separados.
- No genera recomendaciones de compra o venta.
- No clasifica automáticamente una empresa como buena o mala.
- No inventa métricas ausentes.
- Adapta el análisis al tipo de empresa cuando es necesario.
"""


def _porcentaje(valor):
    if valor is None:
        return None
    return valor * 100


def _dato(nombre, valor, texto, advertencia=None):
    return {
        "nombre": nombre,
        "valor": valor,
        "texto": texto,
        "advertencia": advertencia,
    }


def analizar_crecimiento(datos):
    resultados = []

    ingresos = datos.get("crecimiento_ingresos")
    beneficios = datos.get("crecimiento_beneficios")

    if ingresos is not None:
        pct = _porcentaje(ingresos)

        resultados.append(
            _dato(
                "Crecimiento de ingresos",
                ingresos,
                f"La fuente registra una variación de ingresos del {pct:+.1f}%.",
                (
                    "Una única tasa de crecimiento no permite determinar "
                    "si existe una tendencia estructural."
                ),
            )
        )

    if beneficios is not None:
        pct = _porcentaje(beneficios)

        resultados.append(
            _dato(
                "Crecimiento de beneficios",
                beneficios,
                f"La fuente registra una variación de beneficios del {pct:+.1f}%.",
                (
                    "El beneficio puede verse afectado por elementos "
                    "extraordinarios, impuestos y otras partidas."
                ),
            )
        )

    if ingresos is not None and beneficios is not None:
        if beneficios > ingresos:
            texto = (
                "En el periodo de comparación utilizado por la fuente, "
                "el beneficio crece a una tasa superior a los ingresos."
            )
        elif beneficios < ingresos:
            texto = (
                "En el periodo de comparación utilizado por la fuente, "
                "los ingresos crecen a una tasa superior al beneficio."
            )
        else:
            texto = (
                "En el periodo de comparación utilizado por la fuente, "
                "ingresos y beneficios presentan la misma tasa de variación."
            )

        resultados.append(
            _dato(
                "Relación crecimiento beneficio/ingresos",
                None,
                texto,
                (
                    "Esta comparación describe únicamente los datos disponibles "
                    "y no determina por sí sola la calidad del crecimiento."
                ),
            )
        )

    return resultados


def analizar_rentabilidad(datos):
    resultados = []

    metricas = [
        ("Margen bruto", "margen_bruto"),
        ("Margen operativo", "margen_operativo"),
        ("Margen neto", "margen_neto"),
        ("ROE", "roe"),
        ("ROA", "roa"),
    ]

    for nombre, clave in metricas:
        valor = datos.get(clave)

        if valor is None:
            continue

        pct = _porcentaje(valor)

        if clave == "roe":
            advertencia = (
                "El ROE puede estar influido por el endeudamiento, "
                "recompras de acciones y un patrimonio reducido."
            )
        elif clave == "roa":
            advertencia = (
                "El ROA depende de la intensidad de activos del negocio, "
                "por lo que su comparación entre sectores es limitada."
            )
        else:
            advertencia = (
                "Los márgenes deben interpretarse teniendo en cuenta "
                "el sector y el modelo de negocio."
            )

        resultados.append(
            _dato(
                nombre,
                valor,
                f"{nombre}: {pct:.1f}%.",
                advertencia,
            )
        )

    return resultados


def analizar_caja(datos):
    # Para entidades financieras evitamos interpretar el FCF
    # como si fuera una empresa industrial o tecnológica.
    if datos.get("es_financiera"):
        return [
            _dato(
                "Flujo de caja",
                None,
                (
                    "FinScope no aplica el análisis convencional de Free Cash "
                    "Flow a esta empresa porque está clasificada dentro de "
                    "Financial Services."
                ),
                (
                    "En entidades financieras, movimientos de depósitos, "
                    "préstamos y otros activos y pasivos financieros hacen que "
                    "el FCF convencional no sea directamente comparable con "
                    "el de una empresa no financiera."
                ),
            )
        ]

    resultados = []

    fcf = datos.get("free_cash_flow_ttm")
    ocf = datos.get("flujo_caja_operativo")
    beneficio = datos.get("beneficio_neto")

    if fcf is not None:
        resultados.append(
            _dato(
                "Free Cash Flow TTM",
                fcf,
                (
                    f"El Free Cash Flow TTM calculado por FinScope es "
                    f"{fcf / 1_000_000_000:,.2f} B."
                ),
                (
                    "Se calcula sumando los cuatro últimos estados "
                    "trimestrales disponibles."
                ),
            )
        )

    if ocf is not None:
        resultados.append(
            _dato(
                "Flujo de caja operativo",
                ocf,
                (
                    f"El flujo de caja operativo registrado por la fuente es "
                    f"{ocf / 1_000_000_000:,.2f} B."
                ),
            )
        )

    if fcf is not None and beneficio not in (None, 0):
        conversion = fcf / beneficio

        resultados.append(
            _dato(
                "Conversión beneficio a FCF",
                conversion,
                (
                    f"El FCF TTM equivale aproximadamente al "
                    f"{conversion * 100:.1f}% del beneficio neto TTM."
                ),
                (
                    "Esta relación puede variar significativamente por "
                    "capital circulante, inversión y elementos extraordinarios."
                ),
            )
        )

    return resultados


def analizar_balance(datos):
    if datos.get("es_financiera"):
        return [
            _dato(
                "Balance",
                None,
                (
                    "FinScope no aplica las reglas convencionales de deuda, "
                    "liquidez y deuda/patrimonio a esta entidad financiera."
                ),
                (
                    "El balance de bancos y otras entidades financieras "
                    "requiere métricas específicas."
                ),
            )
        ]

    resultados = []

    efectivo = datos.get("efectivo")
    deuda = datos.get("deuda_total")
    current = datos.get("current_ratio")
    quick = datos.get("quick_ratio")
    deuda_patrimonio = datos.get("debt_to_equity")

    if efectivo is not None and deuda is not None:
        deuda_neta = deuda - efectivo

        resultados.append(
            _dato(
                "Deuda neta",
                deuda_neta,
                (
                    f"Deuda total menos efectivo: "
                    f"{deuda_neta / 1_000_000_000:,.2f} B."
                ),
                (
                    "Esta medida simplificada no incluye todos los posibles "
                    "activos líquidos ni ajustes de deuda."
                ),
            )
        )

    if current is not None:
        resultados.append(
            _dato(
                "Current ratio",
                current,
                f"Current ratio registrado: {current:.2f}x.",
                (
                    "Su interpretación depende del sector y de la estructura "
                    "del capital circulante."
                ),
            )
        )

    if quick is not None:
        resultados.append(
            _dato(
                "Quick ratio",
                quick,
                f"Quick ratio registrado: {quick:.2f}x.",
                (
                    "No debe interpretarse de forma aislada como prueba "
                    "de solvencia."
                ),
            )
        )

    if deuda_patrimonio is not None:
        resultados.append(
            _dato(
                "Deuda / patrimonio",
                deuda_patrimonio,
                f"Deuda/patrimonio registrado por la fuente: {deuda_patrimonio:.2f}.",
                (
                    "La definición y escala de este ratio dependen de la "
                    "fuente y debe contextualizarse antes de compararlo."
                ),
            )
        )

    return resultados


def analizar_valoracion(datos):
    resultados = []

    metricas = [
        ("PER", "per", "x"),
        ("PER forward", "per_forward", "x"),
        ("Price / Book", "price_to_book", "x"),
        ("EV / Ventas", "ev_ventas", "x"),
    ]

    # EV/EBITDA no se utiliza para financieras.
    if not datos.get("es_financiera"):
        metricas.append(("EV / EBITDA", "ev_ebitda", "x"))

    for nombre, clave, unidad in metricas:
        valor = datos.get(clave)

        if valor is None:
            continue

        resultados.append(
            _dato(
                nombre,
                valor,
                f"{nombre}: {valor:.2f}{unidad}.",
                (
                    "Este múltiplo no permite determinar por sí solo si "
                    "una acción está cara o barata. Debe compararse con "
                    "su historia, sector, crecimiento y características "
                    "del negocio."
                ),
            )
        )

    return resultados


def analizar_empresa(datos):
    """
    Punto de entrada del motor de análisis.

    Devuelve información estructurada para que app.py pueda decidir
    posteriormente cómo mostrarla.
    """

    if not datos:
        return {
            "tipo_empresa": "desconocida",
            "bloques": {},
            "advertencias": ["No hay datos suficientes para analizar la empresa."],
        }

    es_financiera = bool(datos.get("es_financiera"))

    return {
        "tipo_empresa": "financiera" if es_financiera else "no_financiera",
        "sector": datos.get("sector"),
        "industria": datos.get("industria"),
        "bloques": {
            "crecimiento": analizar_crecimiento(datos),
            "rentabilidad": analizar_rentabilidad(datos),
            "caja": analizar_caja(datos),
            "balance": analizar_balance(datos),
            "valoracion": analizar_valoracion(datos),
        },
        "advertencias": [
            (
                "El análisis describe e interpreta los datos disponibles. "
                "No constituye una recomendación de compra o venta."
            ),
            (
                "Las métricas deben contextualizarse con el sector, "
                "los estados financieros y la evolución histórica."
            ),
        ],
    }
