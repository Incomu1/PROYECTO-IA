from __future__ import annotations
import csv, json, random
from pathlib import Path
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from src.linea_base import (cargar_consumo_horario, dividir_cronologicamente,
    calcular_umbral_mad, predecir_anomalia, inyectar_anomalias, calcular_metricas)

SEMILLA = 2026

def caracteristicas(x):
    a=np.asarray(x,dtype=float)
    prev=np.r_[a[0],a[:-1]]
    med=np.array([np.median(a[max(0,i-3):i+1]) for i in range(len(a))])
    return np.column_stack([a, a-prev, a-med])

def pred_score(modelo, x, tasa=None):
    return modelo.predict(x) == -1

def ejecutar(datos:Path, salida:Path, errores:Path, semilla=SEMILLA):
    valores,resumen=cargar_consumo_horario(datos)
    tr,va,te=dividir_cronologicamente(valores)
    va_x,va_y,va_iny=inyectar_anomalias(va,semilla=semilla+1)
    te_x,te_y,te_iny=inyectar_anomalias(te,semilla=semilla)
    tr_f=caracteristicas(tr); va_f=caracteristicas(va_x); te_f=caracteristicas(te_x)
    candidatos=[]
    for cont in [0.005,0.01,0.02,0.05,0.1]:
        mod=IsolationForest(n_estimators=200,contamination=cont,random_state=semilla,n_jobs=1).fit(tr_f)
        p=pred_score(mod,va_f,cont); m=calcular_metricas(va_y,p.tolist())
        candidatos.append((m['f1'],cont))
    cont_if=max(candidatos)[1]
    candidatos=[]
    for k in [10,20,35,50]:
        for cont in [0.005,0.01,0.02,0.05,0.1]:
            mod=LocalOutlierFactor(n_neighbors=k,novelty=True,contamination=cont,n_jobs=1).fit(tr_f)
            p=mod.predict(va_f)==-1; m=calcular_metricas(va_y,p.tolist())
            candidatos.append((m['f1'],k,cont))
    _,k_lof,cont_lof=max(candidatos)
    um=calcular_umbral_mad(tr)
    mad_pred=[predecir_anomalia(x,um) for x in te_x]
    ifm=IsolationForest(n_estimators=200,contamination=cont_if,random_state=semilla,n_jobs=1).fit(tr_f)
    if_pred=pred_score(ifm,te_f,cont_if)
    lof=LocalOutlierFactor(n_neighbors=k_lof,novelty=True,contamination=cont_lof,n_jobs=1).fit(tr_f)
    lof_pred=lof.predict(te_f)==-1
    preds={'MAD':mad_pred,'Isolation Forest':if_pred.tolist(),'LOF':lof_pred.tolist()}
    metr={n:calcular_metricas(te_y,p) for n,p in preds.items()}
    casos=[]
    for i,y in enumerate(te_y):
        for name,ps in preds.items():
            if bool(ps[i]) != bool(y):
                casos.append({'modelo':name,'indice_prueba':i,'hora_serie':len(tr)+len(va)+i,
                  'tipo':'falso negativo' if y else 'falso positivo','etiqueta_sintetica':bool(y),
                  'prediccion':bool(ps[i]),'consumo_kW':round(te_x[i],5),
                  'causa_probable':'El aumento simulado puede solaparse con picos de consumo habituales.' if y else 'La variación natural del hogar puede parecerse a una inyección.'})
    errores.parent.mkdir(parents=True,exist_ok=True)
    elegidos=[]
    for name in preds:
      for tipo in ('falso negativo','falso positivo'):
       c=next((z for z in casos if z['modelo']==name and z['tipo']==tipo and z not in elegidos),None)
       if c: elegidos.append(c)
    for c in casos:
      if len(elegidos)>=10: break
      if c not in elegidos:elegidos.append(c)
    with errores.open('w',encoding='utf-8',newline='') as f:
      w=csv.DictWriter(f,fieldnames=elegidos[0].keys());w.writeheader();w.writerows(elegidos)
    out={'semilla':semilla,'semilla_validacion':semilla+1,'datos':resumen,'particion':'cronológica 60/20/20',
      'inyeccion_validacion':va_iny,'inyeccion_prueba':te_iny,
      'ajuste':{'Isolation Forest':{'contamination':cont_if,'n_estimators':200},'LOF':{'n_neighbors':k_lof,'contamination':cont_lof,'novelty':True}},
      'metricas_prueba':metr,'meta_hito1':{'f1_objetivo':0.8,'mejora_minima_sobre_mad':0.05,
       'cumple_f1':any(m['f1']>=0.8 for m in metr.values()),
       'cumple_mejora':max(m['f1'] for n,m in metr.items() if n!='MAD')>=metr['MAD']['f1']+0.05},
      'limite':'Etiquetas sintéticas, no verdad-terreno de anomalías reales; un único hogar de Francia.'}
    salida.parent.mkdir(parents=True,exist_ok=True);salida.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return out
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--datos',type=Path,default=Path('data/raw/household_power_consumption.txt'));p.add_argument('--salida',type=Path,default=Path('experiments/comparacion_modelos.json'));p.add_argument('--errores',type=Path,default=Path('experiments/errores.csv'));a=p.parse_args();print(json.dumps(ejecutar(a.datos,a.salida,a.errores),ensure_ascii=False,indent=2))
