from pathlib import Path
import sys
import py_compile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

errores = []
avisos = []


def ok(texto):
    print(f"✅ {texto}")


def error(texto):
    print(f"❌ {texto}")
    errores.append(texto)


def aviso(texto):
    print(f"⚠️  {texto}")
    avisos.append(texto)


print()
print("=" * 72)
print("FINSCOPE · AUDITORÍA AUTOMÁTICA")
print("=" * 72)


# ============================================================
# 1. ARCHIVOS PRINCIPALES
# ============================================================

print("\n[1/7] ESTRUCTURA")

archivos = [
    ROOT / "app.py",
    ROOT / "services" / "financial_data.py",
    ROOT / "services" / "analysis_engine.py",
]

for archivo in archivos:
    if archivo.exists():
        ok(str(archivo.relative_to(ROOT)))
    else:
        error(f"Falta {archivo.relative_to(ROOT)}")


# ============================================================
# 2. SINTAXIS
# ============================================================

print("\n[2/7] SINTAXIS")

for archivo in archivos:
    if not archivo.exists():
        continue

    try:
        py_compile.compile(
            str(archivo),
            doraise=True
        )
        ok(f"{archivo.name} compila")
    except Exception as e:
        error(f"{archivo.name}: {e}")


# ============================================================
# 3. IMPORTS
# ============================================================

print("\n[3/7] IMPORTS")

try:
    from services.financial_data import (
        obtener_datos_empresa,
        obtener_historico,
    )
    ok("financial_data importado")
except Exception as e:
    error(f"financial_data no importa: {e}")
    obtener_datos_empresa = None
    obtener_historico = None

try:
    from services.analysis_engine import analizar_empresa
    ok("analysis_engine importado")
except Exception as e:
    error(f"analysis_engine no importa: {e}")
    analizar_empresa = None


# ============================================================
# 4. EMPRESA NO FINANCIERA
# ============================================================

print("\n[4/7] AAPL · EMPRESA NO FINANCIERA")

aapl = None

if obtener_datos_empresa:
    try:
        aapl = obtener_datos_empresa("AAPL")

        if aapl:
            ok("AAPL devuelve datos")
        else:
            error("AAPL no devuelve datos")

        if aapl and aapl.get("es_financiera") is False:
            ok("AAPL detectada como no financiera")
        else:
            error("Clasificación incorrecta de AAPL")

        claves_aapl = [
            "precio",
            "capitalizacion",
            "ingresos",
            "beneficio_neto",
            "roe",
            "per",
        ]

        for clave in claves_aapl:
            if aapl and aapl.get(clave) is not None:
                ok(f"AAPL · {clave}")
            else:
                aviso(f"AAPL · {clave} no disponible")

        if aapl and aapl.get("free_cash_flow_ttm") is not None:
            ok("AAPL · FCF TTM disponible")
        else:
            aviso("AAPL · FCF TTM no disponible")

    except Exception as e:
        error(f"Error consultando AAPL: {e}")


# ============================================================
# 5. EMPRESA FINANCIERA
# ============================================================

print("\n[5/7] JPM · ENTIDAD FINANCIERA")

jpm = None

if obtener_datos_empresa:
    try:
        jpm = obtener_datos_empresa("JPM")

        if jpm:
            ok("JPM devuelve datos")
        else:
            error("JPM no devuelve datos")

        if jpm and jpm.get("es_financiera") is True:
            ok("JPM detectada como financiera")
        else:
            error("Clasificación incorrecta de JPM")

        claves_banco = [
            "net_interest_income",
            "interest_income",
            "interest_expense",
            "net_loans",
            "common_stock_equity",
            "tangible_book_value",
            "total_assets",
        ]

        for clave in claves_banco:
            if jpm and jpm.get(clave) is not None:
                ok(f"JPM · {clave}")
            else:
                aviso(f"JPM · {clave} no disponible")

        if jpm and jpm.get("fecha_resultados_bancarios"):
            ok(
                "JPM · fecha resultados "
                + str(jpm["fecha_resultados_bancarios"])
            )
        else:
            aviso("JPM · fecha de resultados bancarios ausente")

    except Exception as e:
        error(f"Error consultando JPM: {e}")


# ============================================================
# 6. MOTOR DE ANÁLISIS
# ============================================================

print("\n[6/7] MOTOR DE ANÁLISIS")

if analizar_empresa and aapl:
    try:
        analisis_aapl = analizar_empresa(aapl)

        if analisis_aapl.get("tipo_empresa") == "no_financiera":
            ok("Motor trata AAPL como no financiera")
        else:
            error("Motor clasifica incorrectamente AAPL")

    except Exception as e:
        error(f"Motor AAPL: {e}")


if analizar_empresa and jpm:
    try:
        analisis_jpm = analizar_empresa(jpm)

        if analisis_jpm.get("tipo_empresa") == "financiera":
            ok("Motor trata JPM como financiera")
        else:
            error("Motor clasifica incorrectamente JPM")

        caja = analisis_jpm.get("bloques", {}).get("caja", [])
        texto_caja = " ".join(
            str(x.get("texto", ""))
            for x in caja
        ).lower()

        if "financial services" in texto_caja or "financier" in texto_caja:
            ok("Motor evita análisis convencional de FCF en JPM")
        else:
            aviso(
                "Revisar manualmente el tratamiento del FCF de JPM"
            )

    except Exception as e:
        error(f"Motor JPM: {e}")


# ============================================================
# 7. HISTÓRICO Y DEUDA TÉCNICA
# ============================================================

print("\n[7/7] HISTÓRICO Y LIMPIEZA")

if obtener_historico:
    try:
        hist = obtener_historico("AAPL", "1y")

        if hist is not None and not hist.empty:
            ok(f"Histórico AAPL 1y · {len(hist)} registros")
        else:
            error("Histórico AAPL 1y vacío")

    except Exception as e:
        error(f"Histórico AAPL: {e}")


app_path = ROOT / "app.py"

if app_path.exists():
    texto_app = app_path.read_text()

    usos_antiguos = texto_app.count("use_container_width")

    if usos_antiguos:
        aviso(
            f"Streamlit: quedan {usos_antiguos} usos de "
            "use_container_width por migrar"
        )
    else:
        ok("Sin use_container_width obsoleto")


# ============================================================
# RESUMEN
# ============================================================

print()
print("=" * 72)
print("RESULTADO")
print("=" * 72)

print(f"Errores:      {len(errores)}")
print(f"Advertencias: {len(avisos)}")

if errores:
    print("\n❌ AUDITORÍA FALLIDA")
    print("No conviene continuar con un bloque nuevo hasta revisar los errores.")
    sys.exit(1)

if avisos:
    print("\n✅ BASE FUNCIONAL")
    print("Hay advertencias, pero no se han detectado fallos críticos.")
    sys.exit(0)

print("\n✅ FINSCOPE SUPERA TODA LA AUDITORÍA")
