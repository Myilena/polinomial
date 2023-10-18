import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Datos de ejemplo: tiempo (en minutos), velocidad (en mph), y distancia (en millas)
tiempo = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
velocidad = np.array([0, 5, 15, 20, 30, 40, 45, 55, 60, 70])
distancia = np.array([0, 5, 20, 45, 80, 125, 180, 245, 320, 405])

# Crear un DataFrame con las variables
df = pd.DataFrame({'Tiempo (minutos)': tiempo, 'Velocidad (mph)': velocidad, 'Distancia (millas)': distancia})

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df[['Tiempo (minutos)', 'Distancia (millas)']]
y = df['Velocidad (mph)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Crear un objeto PolynomialFeatures para transformar los datos
grado_del_polinomio = 2
poly = PolynomialFeatures(degree=grado_del_polinomio)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Crear y entrenar un modelo de regresión lineal con los datos transformados
regression = LinearRegression()
regression.fit(X_train_poly, y_train)

# Predecir la velocidad en función del tiempo y la distancia
X_pred = poly.transform(X)
velocidad_pred = regression.predict(X_pred)

# Agregar las predicciones al DataFrame
df['Predicción (mph)'] = velocidad_pred

# Guardar el DataFrame en un archivo Excel
df.to_excel('regresion_polinomial.xlsx', index=False)

# Visualizar los datos y la regresión polinomial
plt.scatter(tiempo, velocidad, label="Datos reales", color='blue')
plt.plot(tiempo, velocidad_pred, label="Regresión Polinomial", color='red')
plt.xlabel("Tiempo (minutos)")
plt.ylabel("Velocidad (mph)")
plt.legend()
plt.show()
