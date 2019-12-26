#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from statistics import mode
from datetime import datetime
import time
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.ensemble import BalancedBaggingClassifier # doctest: +NORMALIZE_WHITESPACE
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
data=pd.read_excel('C:/Users/pneira/Desktop/Garantías Financieras/bfer&cuadratura_v2.xlsx')

#donde c es el random state del split data, y op es el random state del algoritmo
'''Esta función retorna:
    (accuracy,recall, precision)
    nota: modifiqué ahora solo entrega recall
    '''
def func(semilla,maxsamples,maxfeatures,nestimadores=10,it=10,data=data):
    recall=[]
    for i in range(0,it):
        train, test = train_test_split(data, test_size=0.30,random_state=i)
        X_train= train[['MONTO_EN_PESO (MLOCAL)', 'DURATION', 'TIENE_AVALES', 'NMPERIODOTASA',
           'valor_cuota_1', 'gracia_dias', 'Cuoton', 'cuota_unica',
           'iguales', 'variable', 'nm_cod_garantia', 'valor_PESO',
           'Acciones de oferta pública', 'Aeronaves',
           'Automóviles (no inventario)', 'Bienes raíces rurales', 'Bodegas',
           'Buses (no inventario)', 'Camiones o camionetas (inventario)',
           'Camiones o camionetas (no inventario)', 'Casas',
           'Construcciones industriales', 'Departamentos',
           'Depósitos a plazo en el país', 'Derechos de aguas',
           'Edificios de destino específicos (clínicas, colegios, etc)',
           'Embarcaciones menores', 'Embarcaciones o naves marinas',
           'Estacionamientos', 'Instalaciones', 'Locales comerciales',
           'Maquinarias y/o equipos (inventario)',
           'Maquinarias y/o equipos (no inventario)', 'Oficinas',
           'Otras garantías', 'Otros bienes corporales en prenda o warrant',
           'Otros bienes hipotecados', 'Otros vehículos (no inventario)',
           'Plantaciones', 'Sitios y terrenos urbanos', 'Dólar Estados Unidos',
           'Peso Chile', 'U.F. Pesos chilenos reajustables',
           'Addwise Inversiones S.A.', 'Addwise Servicios Financieros S.A.',
           'BANCO BBVA', 'BP Cred con Gtia Fondo de Inversion Privado',
           'BP Credito con Garantia Fondo de Inversion Privado', 'Banco Consorcio',
           'Banco CorpBanca', 'Banco Internacional', 'Banco Itau Chile',
           'Banco Itau Corpbanca', 'Banco Santander- Chile', 'Banco Security',
           'Banco del Estado de Chile', 'Becual SA',
           'CAPITAL EXPRESS FONDO DE INVERSIÓN PRIVADO', 'CUMPLO CHILE S.A.',
           'Capital Express Servicios Financieros S.A.', 'Endurance Leasing SPA',
           'FONDO DE INVERSION PRIVADO SARTOR PROYECCION',
           'FONDO DE INVERSION PRIVADO SGR',
           'FONDO DE INVERSION PRIVADO TOESCA-INGE CREDITOS SGR',
           'FONDO DE INVERSION SARTOR PROYECCION',
           'Fondo de Inversion FYNSA Renta Fija Privada I',
           'Fondo de Inversion Privado Addwise Financista Uno',
           'Fondo de Inversion Privado Creditos SGR INGE',
           'Fondo de Inversion Privado MBI - BP Deuda Con Garantia',
           'Fondo de inversion activa Financiamiento Estructurado',
           'Fondo de inversion activa deuda SGR',
           'MBI-BP Deuda Fondo de Inversion',
           'Neorentas Deuda Privada Fondo de Inversion',
           'PENTA VIDA COMPANIA DE SEGUROS VIDA S A', 'PROYECTA CAPITAL SpA',
           'RedPyme S.A', 'RedPyme S.A (según listado anexo de acreedores)',
           'Sartor Administradora de Fondos de Inversión Privado S.A',
           'TANNER SERVICIOS FINANCIEROS S.A',
           'VOLCOMCAPITAL Deuda Privada Fondo de Inversion']]
        y_train= train['estado_1']
        X_test= test[['MONTO_EN_PESO (MLOCAL)', 'DURATION', 'TIENE_AVALES', 'NMPERIODOTASA',
           'valor_cuota_1', 'gracia_dias', 'Cuoton', 'cuota_unica',
           'iguales', 'variable', 'nm_cod_garantia', 'valor_PESO',
           'Acciones de oferta pública', 'Aeronaves',
           'Automóviles (no inventario)', 'Bienes raíces rurales', 'Bodegas',
           'Buses (no inventario)', 'Camiones o camionetas (inventario)',
           'Camiones o camionetas (no inventario)', 'Casas',
           'Construcciones industriales', 'Departamentos',
           'Depósitos a plazo en el país', 'Derechos de aguas',
           'Edificios de destino específicos (clínicas, colegios, etc)',
           'Embarcaciones menores', 'Embarcaciones o naves marinas',
           'Estacionamientos', 'Instalaciones', 'Locales comerciales',
           'Maquinarias y/o equipos (inventario)',
           'Maquinarias y/o equipos (no inventario)', 'Oficinas',
           'Otras garantías', 'Otros bienes corporales en prenda o warrant',
           'Otros bienes hipotecados', 'Otros vehículos (no inventario)',
           'Plantaciones', 'Sitios y terrenos urbanos', 'Dólar Estados Unidos',
           'Peso Chile', 'U.F. Pesos chilenos reajustables',
           'Addwise Inversiones S.A.', 'Addwise Servicios Financieros S.A.',
           'BANCO BBVA', 'BP Cred con Gtia Fondo de Inversion Privado',
           'BP Credito con Garantia Fondo de Inversion Privado', 'Banco Consorcio',
           'Banco CorpBanca', 'Banco Internacional', 'Banco Itau Chile',
           'Banco Itau Corpbanca', 'Banco Santander- Chile', 'Banco Security',
           'Banco del Estado de Chile', 'Becual SA',
           'CAPITAL EXPRESS FONDO DE INVERSIÓN PRIVADO', 'CUMPLO CHILE S.A.',
           'Capital Express Servicios Financieros S.A.', 'Endurance Leasing SPA',
           'FONDO DE INVERSION PRIVADO SARTOR PROYECCION',
           'FONDO DE INVERSION PRIVADO SGR',
           'FONDO DE INVERSION PRIVADO TOESCA-INGE CREDITOS SGR',
           'FONDO DE INVERSION SARTOR PROYECCION',
           'Fondo de Inversion FYNSA Renta Fija Privada I',
           'Fondo de Inversion Privado Addwise Financista Uno',
           'Fondo de Inversion Privado Creditos SGR INGE',
           'Fondo de Inversion Privado MBI - BP Deuda Con Garantia',
           'Fondo de inversion activa Financiamiento Estructurado',
           'Fondo de inversion activa deuda SGR',
           'MBI-BP Deuda Fondo de Inversion',
           'Neorentas Deuda Privada Fondo de Inversion',
           'PENTA VIDA COMPANIA DE SEGUROS VIDA S A', 'PROYECTA CAPITAL SpA',
           'RedPyme S.A', 'RedPyme S.A (según listado anexo de acreedores)',
           'Sartor Administradora de Fondos de Inversión Privado S.A',
           'TANNER SERVICIOS FINANCIEROS S.A',
           'VOLCOMCAPITAL Deuda Privada Fondo de Inversion']]
        y_test= test['estado_1']

        bbc = BalancedBaggingClassifier(n_estimators=nestimadores,max_samples=maxsamples,max_features=maxfeatures,random_state=semilla)
        bbc.fit(X_train, y_train) 
        y_pred = bbc.predict(X_test)
        recall.append(recall_score(y_test,y_pred, pos_label=1,average="binary"))
    return np.mean(recall)