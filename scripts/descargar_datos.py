#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
ARCHIVE_NAME = "individual_household_power_consumption.zip"
DATA_NAME = "household_power_consumption.txt"

def descargar(destino: Path, forzar: bool = False) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    archivo_zip = destino / ARCHIVE_NAME
    archivo_datos = destino / DATA_NAME

    if not archivo_zip.exists() or forzar:
        temporal = archivo_zip.with_suffix(archivo_zip.suffix + ".part")
        solicitud = urllib.request.Request(
            URL,
            headers={"User-Agent": "Proyecto academico IA/1.0"},
        )
        try:
            with urllib.request.urlopen(solicitud, timeout=120) as respuesta:
                with temporal.open("wb") as salida:
                    shutil.copyfileobj(respuesta, salida)
            temporal.replace(archivo_zip)
        except (OSError, urllib.error.URLError) as error:
            temporal.unlink(missing_ok=True)
            raise RuntimeError(
                f"No se pudo descargar desde UCI ({URL}): {error}"
            ) from error

    try:
        with zipfile.ZipFile(archivo_zip) as paquete:
            miembro = next(
                (
                    nombre
                    for nombre in paquete.namelist()
                    if Path(nombre).name == DATA_NAME
                ),
                None,
            )
            if miembro is None:
                raise RuntimeError(
                    f"El ZIP no contiene el archivo esperado: {DATA_NAME}"
                )
            with paquete.open(miembro) as origen, archivo_datos.open("wb") as salida:
                shutil.copyfileobj(origen, salida)
    except zipfile.BadZipFile as error:
        raise RuntimeError(
            f"El archivo descargado no es un ZIP válido: {archivo_zip}"
        ) from error

    return archivo_datos

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Descarga y prepara los datos oficiales de UCI."
    )
    parser.add_argument(
        "--destino",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
        help="carpeta de salida (por defecto: data/raw)",
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="vuelve a descargar el archivo ZIP",
    )
    args = parser.parse_args()

    try:
        ruta = descargar(args.destino, args.forzar)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Datos listos en: {ruta}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
