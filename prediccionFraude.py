# ---------------- PREDICCIÒN DE FRAUDES CON TRAJETA DE CREDITO ------------------------------

# --- Importar las librerias ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,classification_report, roc_auc_score)
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import RocCurveDisplay

# -----------------------PREPARACIÓN Y EXPLORACIÒN DE DATOS --------------------------------------

# Cargar datos
df = pd.read_csv('creditcard.csv')

# Mostrar las primeras filas
print(df.head())

# Información general del conjunto de datos
print(df.info())

# Estadísticas básicas
print(df.describe(include=[np.number]))

# Distribución de clases
print(df['Class'].value_counts())
sns.countplot(x='Class', data=df)

# ----------------------PREPROCESAMIENTO Y SMOTE ---------------------------------

# Separar características y etiquetas
X = df.drop('Class', axis=1)
y = df['Class']

# Escalado (Normalizar los datos-> Transforma los datos con media = 0 y desviacion estandar =1 )
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---------------------- MODELADO PREDICTIVO---------------------------------

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.8, #Usamos el 20% para optimizar el codigo
    random_state=42, 
    stratify=y
)

# Usamos la tecnica SMOTE para balancear los datos 
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Observamos como se aplica la tecnica 
print("Distribución después de SMOTE:")
print(pd.Series(y_train_smote).value_counts())

#------------------------------------- MODELO RANDOM FOREST --------------------------------

# -------------------> Creacion del modelo <---------------------
rf_fraud = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',  
    random_state=42,
    n_jobs=-1
)

# -------------------> Entrenamiento del modelo <------------------
rf_fraud.fit(X_train_smote, y_train_smote)
print("Entrenando Random Forest")

# -------------------> Evaluación del modelo <---------------------
train_pred = rf_fraud.predict(X_train_smote)
test_pred = rf_fraud.predict(X_test)

print("\nEvaluación en conjunto de entrenamiento:")
print("Accuracy:", accuracy_score(y_train_smote, train_pred))
print(classification_report(y_train_smote, train_pred))

print("\nEvaluación en conjunto de prueba:")
print("Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

# ----------------------VISUALIZACIÓN---------------------------------

# Matriz de confusión
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, test_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Fraude', 'Fraude'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Modelo de Predicción de Fraude')
plt.show()

# Curva ROC
plt.figure(figsize=(8,6))
RocCurveDisplay.from_estimator(rf_fraud, X_test, y_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Curva ROC - Modelo de Predicción de Fraude')
plt.show()

# ----------------------REPORTE---------------------------------
plt.figure(figsize=(12, 5))

def generate_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    return df_report

full_report = generate_classification_report(y_test, test_pred)

fraud_metrics = {
    'Precisión': full_report.loc['1', 'precision'],
    'Recall (Sensibilidad)': full_report.loc['1', 'recall'],
    'F1-Score': full_report.loc['1', 'f1-score'],
    'AUC-ROC': roc_auc_score(y_test, rf_fraud.predict_proba(X_test)[:, 1])
}

print("\nMétricas Clave para Detección de Fraude:")
for metric, value in fraud_metrics.items():
    print(f"{metric}: {value:.4f}")


print("\nModelo entrenado")