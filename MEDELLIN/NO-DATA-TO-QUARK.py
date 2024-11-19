import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

dtype_spec = {
    'CANTIDAD': float,
}

# Cargar los datos
df = pd.read_csv("dataset.csv", delimiter=",", dtype=dtype_spec, parse_dates=['FECHA HECHO'], dayfirst=True)

# Preprocesamiento de fechas
df['YEAR'] = df['FECHA HECHO'].dt.year
df['QUARTER'] = df['FECHA HECHO'].dt.quarter
df = df.drop(columns=['FECHA HECHO'])

# Agrupar por año y trimestre y sumar las cantidades
df_grouped = df.groupby(['YEAR', 'QUARTER']).sum().reset_index()


# Seleccionar variables independientes (X) y dependiente (y)
X = df_grouped[['YEAR', 'QUARTER']]
y = df_grouped['CANTIDAD']

# Dividir en conjuntos de entrenamiento y prueba
_test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_test_size, random_state=42)

# Normalizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs. Valores Reales')
plt.show()

