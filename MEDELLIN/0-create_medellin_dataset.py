"""
FelipedelosH

Read INCAUTACIONES_DE_MARIHUANA_20241108.csv and extact only information of medellin
"""
import pandas as pd

_csv_file = "INCAUTACIONES_DE_MARIHUANA_20241108.csv"
data = pd.read_csv(_csv_file)
_filter_medellin = data[data['COD_MUNI'] == '05001']
_filter_medellin = _filter_medellin.drop(columns=['COD_DEPTO','DEPARTAMENTO','COD_MUNI','MUNICIPIO','UNIDAD'])

_filter_medellin.to_csv('dataset.csv', index=False)
