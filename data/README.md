# Datos

Se usa **Individual Household Electric Power Consumption**, del UCI Machine Learning Repository. Son mediciones por minuto de un hogar en Sceaux, Francia, entre diciembre de 2006 y noviembre de 2010. El repositorio reporta 2.075.259 registros, valores faltantes y licencia **Creative Commons Atribución 4.0 (CC BY 4.0)**.

- Fuente y ficha: <https://archive.ics.uci.edu/dataset/235/individual>
- Cita: Hebrail, G. y Berard, A. (2006). *Individual Household Electric Power Consumption*. UCI Machine Learning Repository. DOI: <https://doi.org/10.24432/C58K54>.
- Atribución: conservar autores, nombre del conjunto, repositorio y licencia al redistribuirlo.

## Descarga y preparación

Ejecutar desde la raíz del proyecto:

    python3 scripts/descargar_datos.py

El script guarda el ZIP y extrae household_power_consumption.txt en data/raw/. No se debe editar el archivo original. El experimento usa Global_active_power, promedia las lecturas válidas por hora y descarta las horas que no tengan mediciones válidas. Los faltantes se cuentan y se informan; no se rellenan con valores inventados.

El archivo original es grande y no se incluye en este paquete de código. El script oficial permite volver a obtenerlo. Las anomalías de la evaluación se generan artificialmente después de separar los datos, con fines de prueba y sin alterar el archivo de origen.
