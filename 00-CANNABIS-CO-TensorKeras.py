"""
FelipedelosH

TensorFlow - Keras
colombian marijuana seizure
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


dtype_spec = {
    'COD_DEPTO': str,
    'DEPARTAMENTO': str,
    'COD_MUNI': str,
    'MUNICIPIO': str,
    'CANTIDAD': float,
    'UNIDAD': str
}

# STEP 01: LOAD DATA
df = pd.read_csv("INCAUTACIONES_DE_MARIHUANA_20241108.csv", delimiter=",", dtype=dtype_spec, parse_dates=['FECHA HECHO'], dayfirst=True)

# See M Units labels only KG need delete it
# m_units = df['UNIDAD'].unique()
# print(m_units)

# STEP 02 CLEAN DATA
df = df.drop(columns=['COD_DEPTO', 'DEPARTAMENTO', 'MUNICIPIO', 'UNIDAD'])


# Step 03 Preprocesing data
# Dates preprocesing
df['YEAR'] = df['FECHA HECHO'].dt.year
df['MONTH'] = df['FECHA HECHO'].dt.month
df['DAY'] = df['FECHA HECHO'].dt.day
df = df.drop(columns=['FECHA HECHO'])

# Input places only int
del_codes_by_int_error = []

for index, row in df.iterrows():
    try:
        int(row['COD_MUNI'])
    except ValueError:
        del_codes_by_int_error.append(index)
df = df.drop(del_codes_by_int_error)

# Convert place code to INT
df['COD_MUNI'] = df['COD_MUNI'].astype(int)


# Select target X and result Y
X = df[['YEAR', 'MONTH', 'DAY', 'COD_MUNI']]
y = df['CANTIDAD']

# dataset >> fit & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Normalized
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Mostrar las primeras filas de X_train para verificar
# print(X_train[:5])

# Step 04 create model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


# Step 05 fit the model

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# fit
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Evaluate
loss, mse = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on test set: {mse}')

# Predit
# INPUT DATA TO PREDICT:
_YYYY = 2024
_MM = 11
_DD = 8
_COD_MUN = 5001
input_data = {
    'YEAR': [_YYYY],  # Año de la predicción
    'MONTH': [_MM],  # Mes de la predicción
    'DAY': [_DD],  # Día de la predicción
    'COD_MUNI': [_COD_MUN]  # Código del municipio
}

input_df = pd.DataFrame(input_data)
input_scaled = scaler.transform(input_df)

# Predit
prediction = model.predict(input_scaled)


print(f'Predicción de incautación para el municipio {_COD_MUN}: {prediction[0][0]:.4f} kilogramos')
