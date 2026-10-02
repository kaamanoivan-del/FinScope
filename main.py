from services.financial_data import obtener_datos_empresa

ticker = input("Introduce el ticker de una empresa: ").upper()

datos = obtener_datos_empresa(ticker)

print("\n--- FINSCOPE ---")

print("Empresa:", datos["nombre"])
print("Ticker:", datos["ticker"])
print("Sector:", datos["sector"])
print("País:", datos["pais"])

if datos["precio"] is not None:
    print(f'Precio actual: {datos["precio"]:,.2f} {datos["moneda"]}')
else:
    print("Precio actual: No disponible")

if datos["capitalizacion"] is not None:
    capitalizacion = datos["capitalizacion"] / 1_000_000_000
    print(f'Capitalización bursátil: {capitalizacion:,.2f} mil millones {datos["moneda"]}')
else:
    print("Capitalización bursátil: No disponible")

print("\n--- DATOS FINANCIEROS ---")

if datos["ingresos"] is not None:
    print(f'Ingresos: {datos["ingresos"] / 1_000_000_000:,.2f} mil millones {datos["moneda"]}')
else:
    print("Ingresos: No disponible")

if datos["beneficio_neto"] is not None:
    print(f'Beneficio neto: {datos["beneficio_neto"] / 1_000_000_000:,.2f} mil millones {datos["moneda"]}')
else:
    print("Beneficio neto: No disponible")

if datos["margen_neto"] is not None:
    print(f'Margen neto: {datos["margen_neto"] * 100:.2f}%')
else:
    print("Margen neto: No disponible")

if datos["per"] is not None:
    print(f'PER: {datos["per"]:.2f}x')
else:
    print("PER: No disponible")