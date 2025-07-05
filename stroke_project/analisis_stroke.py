import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Cargar el conjunto de datos desde la misma carpeta
file_path = 'C:\Users\ttvga\AppData\Roaming\npm\stroke_project'
df = pd.read_csv(file_path)

# --- Preprocesamiento de Datos ---

# 1. Manejar valores faltantes en 'bmi' con la media
imputer = SimpleImputer(strategy='mean')
df['bmi'] = imputer.fit_transform(df[['bmi']])

# 2. Eliminar filas con valores faltantes en 'smoking_status' (si las hubiera)
df.dropna(subset=['smoking_status'], inplace=True)

# 3. Codificar variables categóricas a numéricas
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4. Eliminar la columna 'id' ya que no es útil para el modelo
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# --- Preparación para el Modelo ---

# Separar las características (X) y la variable objetivo (y)
X = df.drop('stroke', axis=1)
y = df['stroke']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Entrenamiento del Modelo ---

# Inicializar y entrenar el clasificador RandomForest
# n_estimators: número de árboles en el bosque
# random_state: para reproducibilidad
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# --- Evaluación del Modelo ---

# Realizar predicciones en el conjunto de prueba
y_pred = rf_classifier.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.4f}")

# Mostrar un reporte de clasificación más detallado
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Mostrar la importancia de cada característica
print("\nImportancia de las Características:")
feature_importances = pd.DataFrame(rf_classifier.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)
