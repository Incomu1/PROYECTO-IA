import unittest
import numpy as np
from pathlib import Path
from src.experimento_modelos import caracteristicas, pred_score
from src.prototipo import predecir
class EstimadorFalso:
 def predict(self, x): return np.array([-1, 1])
 def score_samples(self, x): return [0.0, 1.0]

class PruebasPrototipo(unittest.TestCase):
 def test_corte_del_modelo_no_se_recalcula_sobre_prueba(self):
  self.assertEqual(pred_score(EstimadorFalso(), [[1],[2]]).tolist(), [True, False])
 def test_caracteristicas_tres_columnas(self):
  x=caracteristicas([1.0,2.0,3.0]);self.assertEqual(x.shape,(3,3))
 def test_mad_devuelve_resultado_y_advertencia(self):
  datos=Path('data/raw/household_power_consumption.txt')
  if not datos.exists(): self.skipTest('requiere descargar el conjunto UCI')
  r=predecir(2.0,'mad',datos);self.assertIn(r['resultado'],['posible valor inusual','sin alerta']);self.assertIn('académico',r['advertencia'])
if __name__=='__main__':unittest.main()
