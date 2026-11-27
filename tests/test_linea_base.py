import tempfile
import unittest
from pathlib import Path

from src.linea_base import (
    calcular_metricas,
    calcular_umbral_mad,
    cargar_consumo_horario,
    inyectar_anomalias,
    predecir_anomalia,
)

class PruebasLineaBase(unittest.TestCase):
    def test_mediana_mad_y_umbral_marcan_extremo(self):
        ajuste = calcular_umbral_mad([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(ajuste["mediana"], 3.0)
        self.assertEqual(ajuste["mad"], 1.0)
        self.assertTrue(predecir_anomalia(100.0, ajuste))
        self.assertFalse(predecir_anomalia(3.0, ajuste))

    def test_mad_cero_usa_respaldo_sin_division_por_cero(self):
        ajuste = calcular_umbral_mad([2.0, 2.0, 2.0, 3.0, 5.0])
        self.assertEqual(ajuste["fuente_escala"], "IQR_por_MAD_cero")
        self.assertGreater(float(ajuste["escala"]), 0.0)

    def test_inyeccion_es_reproducible_y_etiqueta_cada_caso(self):
        valores = [1.0] * 100
        primero = inyectar_anomalias(
            valores, semilla=7, cantidad_picos=4, cantidad_bloques=2
        )
        segundo = inyectar_anomalias(
            valores, semilla=7, cantidad_picos=4, cantidad_bloques=2
        )
        self.assertEqual(primero, segundo)
        modificados, etiquetas, detalle = primero
        self.assertEqual(sum(etiquetas), 4 + 2 * 3)
        self.assertEqual(detalle["horas_etiquetadas"], sum(etiquetas))
        self.assertTrue(any(valor > 1.0 for valor in modificados))

    def test_lector_agrupa_por_hora_y_omite_faltantes(self):
        contenido = (
            "Date;Time;Global_active_power\n"
            "01/01/2007;00:00:00;1.0\n"
            "01/01/2007;00:30:00;3.0\n"
            "01/01/2007;01:00:00;?\n"
            + "".join(
                f"01/01/2007;{hora:02d}:00:00;2.0\n"
                for hora in range(2, 11)
            )
        )
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "muestra.txt"
            ruta.write_text(contenido, encoding="utf-8")
            valores, resumen = cargar_consumo_horario(ruta)
        self.assertEqual(valores[0], 2.0)
        self.assertEqual(resumen["filas_omitidas"], 1)
        self.assertEqual(resumen["horas_validas"], 10)

    def test_metricas_de_caso_sin_alertas(self):
        metricas = calcular_metricas([True, False], [False, False])
        self.assertEqual(metricas["verdaderos_positivos"], 0)
        self.assertEqual(metricas["falsos_negativos"], 1)
        self.assertEqual(metricas["precision"], 0.0)
        self.assertEqual(metricas["recall"], 0.0)
        self.assertEqual(metricas["f1"], 0.0)

    def test_metricas_perfectas(self):
        metricas = calcular_metricas([True, False, True], [True, False, True])
        self.assertEqual(metricas["precision"], 1.0)
        self.assertEqual(metricas["recall"], 1.0)
        self.assertEqual(metricas["f1"], 1.0)

if __name__ == "__main__":
    unittest.main()
