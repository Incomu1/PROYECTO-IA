import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from src.experimento_modelos import caracteristicas
from src.linea_base import cargar_consumo_horario, dividir_cronologicamente, calcular_umbral_mad, predecir_anomalia

def predecir(consumo:float, modelo:str, datos:Path)->dict:
    valores,_=cargar_consumo_horario(datos); tr,_,_=dividir_cronologicamente(valores)
    if modelo=='mad':
        alerta=predecir_anomalia(consumo,calcular_umbral_mad(tr))
    else:
        x=caracteristicas(tr)
        if modelo=='iforest':
            clf=IsolationForest(n_estimators=200,contamination=.005,random_state=2026,n_jobs=1).fit(x)
        else:
            clf=LocalOutlierFactor(n_neighbors=35,novelty=True,contamination=.005,n_jobs=1).fit(x)
        muestra=np.array([[consumo,0.0,0.0]])
        alerta=clf.predict(muestra)[0] == -1
    return {'consumo_kW':consumo,'modelo':modelo,'resultado':'posible valor inusual' if alerta else 'sin alerta','advertencia':'Prototipo académico: no es diagnóstico de falla ni está validado en otros hogares.'}

def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description='Analiza una lectura horaria de consumo.');p.add_argument('--consumo',type=float,required=True);p.add_argument('--modelo',choices=['mad','iforest','lof'],default='iforest');p.add_argument('--datos',type=Path,default=root/'data/raw/household_power_consumption.txt');a=p.parse_args()
    if a.consumo<0: p.error('--consumo debe ser no negativo')
    print(json.dumps(predecir(a.consumo,a.modelo,a.datos),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
