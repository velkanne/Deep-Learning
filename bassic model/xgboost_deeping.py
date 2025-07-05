import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report

# --- 1. GENERAR DATOS DE MUESTRA ---
# Creamos un conjunto de datos sintético para clasificación.
# Esto simula un problema real sin necesidad de un archivo CSV.
X, y = make_classification(
    n_samples=1000,       # 1000 filas de datos
    n_features=20,        # 20 columnas (características)
    n_informative=10,     # 10 de estas características son útiles
    n_redundant=5,        # 5 son redundantes (combinaciones de las útiles)
    n_classes=2,          # 2 posibles resultados (0 o 1)
    random_state=42       # Semilla para que los resultados sean reproducibles
)

print("[OK] Datos de muestra generados.")

# --- 2. DIVIDIR LOS DATOS EN ENTRENAMIENTO Y PRUEBA ---
# 80% para entrenar el modelo, 20% para probarlo.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("[OK] Datos divididos para entrenamiento y prueba.")

# --- 3. CREAR Y ENTRENAR EL MODELO XGBOOST ---
# Creamos una instancia del clasificador de XGBoost.
# Parámetros comunes:
#   - use_label_encoder=False: Evita una advertencia sobre codificación de etiquetas.
#   - eval_metric='logloss': Métrica de evaluación para problemas de clasificación binaria.
#   - n_estimators: Número de árboles a construir.
#   - max_depth: Profundidad máxima de cada árbol.
#   - learning_rate: Tasa de aprendizaje, controla el peso de cada nuevo árbol.
modelo_xgb = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Entrenamos el modelo con los datos de entrenamiento.
print("Entrenando el modelo XGBoost...")
modelo_xgb.fit(X_train, y_train)
print("[OK] Modelo entrenado con éxito.")

# --- 4. HACER PREDICCIONES ---
# Usamos el modelo ya entrenado para predecir sobre los datos de prueba.
predicciones = modelo_xgb.predict(X_test)

# --- 5. EVALUAR EL MODELO ---
# Comparamos las predicciones con los valores reales para ver qué tan bien lo hizo.
precision = accuracy_score(y_test, predicciones)

print("\n--- RESULTADOS DE LA EVALUACIÓN ---")
print(f"Precisión del modelo (Accuracy): {precision:.2%}")
print("\nInforme de Clasificación detallado:")
print(classification_report(y_test, predicciones))
print("------------------------------------")

# --- 6. IMPORTANCIA DE LAS CARACTERÍSTICAS ---
# XGBoost puede decirnos qué características fueron más importantes para tomar sus decisiones.
print("\n--- IMPORTANCIA DE LAS CARACTERÍSTICAS ---")
# Obtenemos la importancia y la asociamos con un nombre de característica
importancias = modelo_xgb.feature_importances_
for i, importancia in enumerate(importancias):
    # Imprimimos solo las 10 más importantes para no saturar la salida
    if i < 10:
        print(f"  - Característica {i+1}: {importancia:.4f}")
print("------------------------------------------")
