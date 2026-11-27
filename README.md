# Detección exploratoria de consumos eléctricos inusuales

Proyecto académico para comparar una regla robusta MAD con Isolation Forest y Local Outlier Factor (LOF). Los datos proceden de un único hogar en Sceaux, Francia; no representan Cartagena ni Colombia. Como UCI no ofrece etiquetas de anomalía, los resultados supervisados de evaluación usan únicamente casos artificiales inyectados y no prueban la detección de fallas reales.

## Requisitos

Python 3.11 o superior. El proyecto fija NumPy y scikit-learn en `requirements.txt`; no requiere API ni servicio pago.

## Reproducir en cuatro comandos

```bash
python3 -m venv .venv
. .venv/bin/activate && pip install -r requirements.txt
python3 scripts/descargar_datos.py
python3 -m unittest discover -s tests -v && python3 -m src.experimento_modelos
```

La última ejecución guarda `experiments/comparacion_modelos.json` y `experiments/errores.csv`. La descarga respeta la licencia CC BY 4.0; conserve la atribución de UCI.

## Prototipo

Con el entorno activado y los datos descargados:

```bash
python3 -m src.prototipo --consumo 3.5 --modelo iforest
```

Modelos admitidos: `mad`, `iforest` y `lof`. La alerta es una demostración, no una recomendación de intervención ni un detector validado en otras viviendas. El CLI es el prototipo funcional exigido; la comparación experimental está en `src/experimento_modelos.py`.

## Estructura

- `src/`: línea base, comparación y CLI.
- `data/`: instrucciones y datos descargables; el archivo bruto no se versiona.
- `experiments/`: JSON de resultados y CSV de errores reproducibles.
- `notebooks/`: cuaderno descriptivo que muestra cómo leer resultados.
- `tests/`: pruebas automatizadas.
- `docs/bitacora_ia.md`: registro de asistencia humano-IA.

Semillas fijadas: 2026 (prueba) y 2027 (validación). Se parte el flujo cronológicamente en 60/20/20 para reducir fuga temporal. Los hiperparámetros se eligen solo con la validación. Como las etiquetas se generan artificialmente, precisión, recall y F1 describen ese escenario sintético, no el desempeño ante anomalías reales.
