"""
FelipedelosH

TensorFlow - Keras
medellin Colombian marijuana seizure
"""
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

dtype_spec = {
    'CANTIDAD': float,
}

# STEP 01: LOAD DATA
df = pd.read_csv("dataset.csv", delimiter=",", dtype=dtype_spec, parse_dates=['FECHA HECHO'], dayfirst=True)
_countInputData = df.shape[0]

# Step 03 Preprocesing data
# Dates preprocesing
df['YEAR'] = df['FECHA HECHO'].dt.year
df['MONTH'] = df['FECHA HECHO'].dt.month
df['DAY'] = df['FECHA HECHO'].dt.day
df = df.drop(columns=['FECHA HECHO'])

# Select target X and result Y
X = df[['YEAR', 'MONTH', 'DAY']]
y = df['CANTIDAD']


# dataset >> fit & test
_test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_test_size, random_state=42)

# Normalized
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 04 create model
model = Sequential()
_input_shape = (X_train.shape[1],)
model.add(Dense(8, activation='relu', input_shape=_input_shape))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))


# Step 05 fit the model
# Compile model
_learning_rate=0.01
optimizer = Adam(learning_rate=_learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])


# FIT with early stopping
_epochs = 100
_patience = 10
early_stopping = EarlyStopping(monitor='val_loss', patience=_patience, restore_best_weights=True)
# fit
history = model.fit(X_train, y_train, epochs=_epochs, validation_data=(X_test, y_test), callbacks=[early_stopping])
# Save Staticsts
_epochs_trained = len(history.epoch)

# SHOW LOSS
#import matplotlib.pyplot as plt

# Mostrar la pérdida durante las épocas
# plt.plot(history.history["loss"])
# plt.xlabel("#Época")
# plt.ylabel("Magnitud de pérdida")
# plt.title("Pérdida durante el entrenamiento")
# plt.show()

# Step 06 Evaluate
loss, mse = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Save train MODEL
now = datetime.now()
formatted_date = now.strftime("%Y-%m-%d-%H.%M")
_output_model_filename = f"model-{formatted_date}.keras"
model.save(_output_model_filename)


# INFO TO CREATE LOG
_output_evaluate = f"\nModel Evaluation Metrics {_output_model_filename}:\n"
_output_evaluate = _output_evaluate + f"Total INPUT X data: {_countInputData}\n"
_layer_sizes = [layer.units for layer in model.layers if isinstance(layer, Dense)]
_output_evaluate = _output_evaluate + "HYPERPARAMERS:\n"
_output_evaluate = _output_evaluate + f'Input Shape: {_input_shape}\n'
_output_evaluate = _output_evaluate + f'Learning rate: {_learning_rate}\n'
_output_evaluate = _output_evaluate + f'EPOCHS: {_epochs_trained}/{_epochs}\n'
_output_evaluate = _output_evaluate + f'LAYERS: {_layer_sizes}\n'
_output_evaluate = _output_evaluate + f'Patience: {_patience}\n'
_output_evaluate = _output_evaluate + f'test size: {_test_size}\n'
_output_evaluate = _output_evaluate + "TEST RESULTS:\n"
_output_evaluate = _output_evaluate + f'LOSS & Mean Squared Error on test set: {mse}\n'
_output_evaluate = _output_evaluate + f'Mean Absolute Error on test set: {mae}\n'
_output_evaluate = _output_evaluate + f'R-squared on test set: {r2}\n'

with open("log.metrics.log", "a", encoding="UTF-8") as f:
    f.write(_output_evaluate)



# Step 07 Predit
# INPUT DATA TO PREDICT:
_YYYY = 2024
_MM = 11
input_data = {
    'YEAR': [_YYYY],  # Año de la predicción
    'MONTH': [_MM],  # Mes de la predicción
}

input_df = pd.DataFrame(input_data)
input_scaled = scaler.transform(input_df)

# Predit
prediction = model.predict(input_scaled)


print(f'Predicción de incautación para el municipio de medellin: {prediction[0][0]:.4f} kilogramos')