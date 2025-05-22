# ---------------- DETECCIÓN DE FRAUDES CON TRAJETA DE CREDITO ------------------------------

# --- Importar las librerias ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix, classification_report


# -----------------------PREPARACIÓN Y EXPLORACIÒN DE DATOS --------------------------------------

# Cargar el conjunto de datos
df = pd.read_csv('creditcard.csv')

# Mostrar las primeras filas
print(df.head())

# Información general del conjunto de datos 
print(df.info())
#En este paso comprobamos como el archivo no tiene datos sin completar, por lo cual podemos trabajar con la hoja sin modificar ni limpiar datos

# Distribución de clases
print(df['Class'].value_counts())
sns.countplot(x='Class', data=df)
plt.title('Distribución de Clases (0: No Fraude, 1: Fraude)')
plt.show() 

# ----------------------- ANALISIS  --------------------------------------
# Medias estadisticas basicas 
print(df.describe().T) 

# --------------------------MODELADO ----------------------------------
scaler = StandardScaler()
X = df.drop(columns=['Class'])
X_scaled = scaler.fit_transform(X)
y_true = df['Class']

# Modelo Isolation Forest
iso_forest = IsolationForest(contamination=0.0017, random_state=42)
df['anomaly_iforest'] = iso_forest.fit_predict(X_scaled)

# Convertimos a formato 0 = normal, 1 = fraude
df['anomaly_iforest'] = df['anomaly_iforest'].map({1: 0, -1: 1})

# Solo usamos 2 componentes (V1, V2) para visualización y DBSCAN
subset = df[['V1', 'V2']].sample(n=5000, random_state=1)
scaled_subset = scaler.fit_transform(subset)

db = DBSCAN(eps=1.5, min_samples=5)
labels = db.fit_predict(scaled_subset)

# Visualización
plt.figure(figsize=(8, 6))
plt.scatter(scaled_subset[:, 0], scaled_subset[:, 1], c=labels, cmap='viridis', s=10)
plt.title("DBSCAN - Detección de Clusters y Outliers")
plt.xlabel("V1")
plt.ylabel("V2")
plt.show()

# ---------------------------- Evaluacion del modelo ------------------

# Métricas para Isolation Forest
y_pred = df['anomaly_iforest']
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

# Visualización de la matriz de confusión
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - Isolation Forest")
plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.show()

# Visualización simple de anomalías vs fraude
sns.countplot(x='anomaly_iforest', hue='Class', data=df)
plt.title("Outliers Detectados vs Clases Reales")
plt.xlabel("Outlier detectado (1 = fraude detectado)")
plt.ylabel("Cantidad")
plt.legend(title="Clase Real", labels=["No Fraude", "Fraude"])
plt.show()


