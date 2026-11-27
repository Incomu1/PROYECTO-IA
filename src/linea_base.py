#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

MULTIPLICADOR_ROBUSTO = 1.4826
UMBRAL_MAD = 3.0
SEMILLA_PREDETERMINADA = 2026

def cargar_consumo_horario(ruta: Path) -> tuple[list[float], dict[str, int]]:
    sumas: dict[tuple[object, int], float] = {}
    conteos: dict[tuple[object, int], int] = {}
    fechas: dict[str, object] = {}
    filas = 0
    omitidas = 0

    with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo, delimiter=";")
        campos = lector.fieldnames or []
        nombre_potencia = next(
            (campo for campo in campos if campo.strip() == "Global_active_power"),
            None,
        )
        nombre_fecha = next(
            (campo for campo in campos if campo.strip() == "Date"), None
        )
        nombre_hora = next(
            (campo for campo in campos if campo.strip() == "Time"), None
        )
        if not nombre_potencia or not nombre_fecha or not nombre_hora:
            raise ValueError(
                "El archivo no tiene las columnas Date, Time y Global_active_power."
            )

        for fila in lector:
            filas += 1
            fecha_texto = (fila.get(nombre_fecha) or "").strip()
            hora_texto = (fila.get(nombre_hora) or "").strip()
            potencia_texto = (fila.get(nombre_potencia) or "").strip()
            if not fecha_texto or not hora_texto or not potencia_texto or potencia_texto == "?":
                omitidas += 1
                continue
            try:
                fecha = fechas.get(fecha_texto)
                if fecha is None:
                    fecha = datetime.strptime(fecha_texto, "%d/%m/%Y").date()
                    fechas[fecha_texto] = fecha
                hora = int(hora_texto[:2])
                potencia = float(potencia_texto)
                if not 0 <= hora <= 23:
                    raise ValueError("hora inválida")
            except (ValueError, TypeError):
                omitidas += 1
                continue

            clave = (fecha, hora)
            sumas[clave] = sumas.get(clave, 0.0) + potencia
            conteos[clave] = conteos.get(clave, 0) + 1

    valores = [
        sumas[clave] / conteos[clave]
        for clave in sorted(sumas)
        if conteos[clave] > 0
    ]
    if len(valores) < 10:
        raise ValueError(
            f"Solo se obtuvieron {len(valores)} horas válidas; se necesitan al menos 10."
        )
    return valores, {
        "filas_leidas": filas,
        "filas_omitidas": omitidas,
        "horas_validas": len(valores),
    }

def dividir_cronologicamente(
    valores: Sequence[float],
) -> tuple[list[float], list[float], list[float]]:
    if len(valores) < 10:
        raise ValueError("Se necesitan al menos 10 valores horarios.")
    corte_entrenamiento = int(len(valores) * 0.60)
    corte_validacion = int(len(valores) * 0.80)
    return (
        list(valores[:corte_entrenamiento]),
        list(valores[corte_entrenamiento:corte_validacion]),
        list(valores[corte_validacion:]),
    )

def percentil(valores: Sequence[float], fraccion: float) -> float:
    ordenados = sorted(valores)
    if not ordenados:
        raise ValueError("No se puede calcular un percentil sin datos.")
    posicion = (len(ordenados) - 1) * fraccion
    inferior = int(posicion)
    superior = min(inferior + 1, len(ordenados) - 1)
    peso = posicion - inferior
    return ordenados[inferior] * (1 - peso) + ordenados[superior] * peso

def calcular_umbral_mad(
    valores_entrenamiento: Sequence[float],
    multiplicador: float = UMBRAL_MAD,
) -> dict[str, float | str]:
    if not valores_entrenamiento:
        raise ValueError("El entrenamiento no puede estar vacío.")
    centro = statistics.median(valores_entrenamiento)
    mad = statistics.median(
        [abs(valor - centro) for valor in valores_entrenamiento]
    )
    escala = MULTIPLICADOR_ROBUSTO * mad
    fuente_escala = "MAD"
    if escala == 0:
        escala = (
            percentil(valores_entrenamiento, 0.75)
            - percentil(valores_entrenamiento, 0.25)
        ) / 1.349
        fuente_escala = "IQR_por_MAD_cero"
        if escala == 0:
            escala = 1e-12
            fuente_escala = "epsilon_por_datos_constantes"
    ancho = multiplicador * escala
    return {
        "mediana": centro,
        "mad": mad,
        "escala": escala,
        "umbral_inferior": centro - ancho,
        "umbral_superior": centro + ancho,
        "fuente_escala": fuente_escala,
        "multiplicador": multiplicador,
    }

def predecir_anomalia(valor: float, umbral: dict[str, float | str]) -> bool:
    return (
        valor < float(umbral["umbral_inferior"])
        or valor > float(umbral["umbral_superior"])
    )

def inyectar_anomalias(
    valores: Sequence[float],
    semilla: int = SEMILLA_PREDETERMINADA,
    cantidad_picos: int | None = None,
    cantidad_bloques: int | None = None,
    duracion_bloque: int = 3,
) -> tuple[list[float], list[bool], dict[str, int | float]]:
    n = len(valores)
    if duracion_bloque < 1 or n < duracion_bloque + 1:
        raise ValueError("El tramo de prueba es demasiado corto para inyectar casos.")
    cantidad_picos = (
        max(1, round(n * 0.005)) if cantidad_picos is None else cantidad_picos
    )
    cantidad_bloques = (
        max(1, round(n * 0.002)) if cantidad_bloques is None else cantidad_bloques
    )
    if cantidad_picos < 0 or cantidad_bloques < 0:
        raise ValueError("Las cantidades de anomalías no pueden ser negativas.")

    rng = random.Random(semilla)
    etiquetas = [False] * n
    modificados = list(valores)
    candidatos_pico = list(range(n))
    rng.shuffle(candidatos_pico)
    seleccion_picos = candidatos_pico[:cantidad_picos]
    if len(seleccion_picos) < cantidad_picos:
        raise ValueError("Hay más picos solicitados que horas disponibles.")
    for indice in seleccion_picos:
        modificados[indice] = max(
            modificados[indice] * 4.0, modificados[indice] + 1.0
        )
        etiquetas[indice] = True

    inicios_seleccionados: list[int] = []
    if cantidad_bloques:
        inicios = list(range(n - duracion_bloque + 1))
        rng.shuffle(inicios)
        for inicio in inicios:
            indices = range(inicio, inicio + duracion_bloque)
            if any(etiquetas[indice] for indice in indices):
                continue
            inicios_seleccionados.append(inicio)
            for indice in indices:
                modificados[indice] = max(
                    modificados[indice] * 2.5, modificados[indice] + 0.5
                )
                etiquetas[indice] = True
            if len(inicios_seleccionados) == cantidad_bloques:
                break
    if len(inicios_seleccionados) < cantidad_bloques:
        raise ValueError("No hay espacio para tantos bloques sostenidos sin solaparlos.")

    detalle: dict[str, int | float] = {
        "picos_simulados": len(seleccion_picos),
        "bloques_sostenidos": len(inicios_seleccionados),
        "horas_por_bloque": duracion_bloque,
        "horas_etiquetadas": sum(etiquetas),
        "factor_pico": 4.0,
        "aumento_minimo_pico_kw": 1.0,
        "factor_aumento_sostenido": 2.5,
        "aumento_minimo_sostenido_kw": 0.5,
    }
    return modificados, etiquetas, detalle

def calcular_metricas(
    reales: Sequence[bool],
    predichas: Sequence[bool],
) -> dict[str, int | float]:
    if len(reales) != len(predichas):
        raise ValueError("Las etiquetas reales y predichas deben tener igual longitud.")
    verdaderos_positivos = sum(r and p for r, p in zip(reales, predichas))
    falsos_positivos = sum((not r) and p for r, p in zip(reales, predichas))
    falsos_negativos = sum(r and (not p) for r, p in zip(reales, predichas))
    precision = (
        verdaderos_positivos / (verdaderos_positivos + falsos_positivos)
        if verdaderos_positivos + falsos_positivos
        else 0.0
    )
    recall = (
        verdaderos_positivos / (verdaderos_positivos + falsos_negativos)
        if verdaderos_positivos + falsos_negativos
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "verdaderos_positivos": verdaderos_positivos,
        "falsos_positivos": falsos_positivos,
        "falsos_negativos": falsos_negativos,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def ejecutar(ruta_datos: Path, ruta_salida: Path, semilla: int) -> dict[str, object]:
    valores, resumen_datos = cargar_consumo_horario(ruta_datos)
    entrenamiento, validacion, prueba = dividir_cronologicamente(valores)
    umbral = calcular_umbral_mad(entrenamiento)
    prueba_inyectada, reales, detalle_inyeccion = inyectar_anomalias(
        prueba, semilla=semilla
    )
    predichas = [predecir_anomalia(valor, umbral) for valor in prueba_inyectada]
    metricas = calcular_metricas(reales, predichas)

    resultado: dict[str, object] = {
        "experimento": "linea_base_mad_consumo_horario",
        "fuente": "UCI Individual Household Electric Power Consumption",
        "semilla": semilla,
        "preparacion": {
            **resumen_datos,
            "unidad_temporal": "hora",
            "variable": "Global_active_power (kW)",
            "estadistico_hora": "promedio de lecturas válidas",
            "particion_cronologica": {
                "entrenamiento_horas": len(entrenamiento),
                "validacion_horas": len(validacion),
                "prueba_horas": len(prueba),
                "porcentajes_aproximados": "60/20/20",
            },
        },
        "linea_base": {
            "metodo": "mediana ± 3 × 1.4826 × MAD",
            **umbral,
            "ajuste_realizado_solo_con": "entrenamiento",
        },
        "inyeccion_sintetica_en_prueba": detalle_inyeccion,
        "matriz_confusion": {
            "verdaderos_positivos": metricas["verdaderos_positivos"],
            "falsos_positivos": metricas["falsos_positivos"],
            "falsos_negativos": metricas["falsos_negativos"],
        },
        "metricas": {
            clave: metricas[clave] for clave in ("precision", "recall", "f1")
        },
        "limite": (
            "Las etiquetas son de anomalías simuladas; las observaciones no modificadas "
            "se toman como normales solo para esta prueba, pues UCI no ofrece etiquetas. "
            "El resultado no valida anomalías reales ni generalización a otros hogares."
        ),
    }
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resultado

def main() -> int:
    raiz = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Ejecuta la regla MAD sobre el conjunto UCI."
    )
    parser.add_argument(
        "--datos",
        type=Path,
        default=raiz / "data" / "raw" / "household_power_consumption.txt",
        help="ruta al archivo TXT descargado desde UCI",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=raiz / "experiments" / "resultado_linea_base.json",
        help="ruta del JSON de resultados",
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=SEMILLA_PREDETERMINADA,
        help="semilla usada para inyectar los casos simulados",
    )
    args = parser.parse_args()

    if not args.datos.is_file():
        print(
            f"No se encuentra {args.datos}. Ejecuta primero "
            "python3 scripts/descargar_datos.py",
            file=sys.stderr,
        )
        return 2
    try:
        resultado = ejecutar(args.datos, args.salida, args.semilla)
    except (OSError, ValueError, csv.Error) as error:
        print(f"No se pudo ejecutar la línea base: {error}", file=sys.stderr)
        return 1

    print(json.dumps(resultado["metricas"], ensure_ascii=False))
    print(f"Resultado guardado en: {args.salida}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
