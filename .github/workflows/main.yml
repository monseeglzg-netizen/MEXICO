import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------------------------
# CARGA Y PREPARACIÓN DE DATOS
# ---------------------------

# Cargar el archivo CSV (ajusta el nombre si es necesario)
df = pd.read_csv("AmericaTemperaturesByCity.csv", encoding="latin-1")

# Filtrar solo las filas de México
df_mex = df[df["Country"] == "Mexico"].copy()

# Convertir la columna de fecha a tipo datetime
df_mex["dt"] = pd.to_datetime(df_mex["dt"])

# Crear columnas de Año y Mes
df_mex["Year"] = df_mex["dt"].dt.year
df_mex["Month"] = df_mex["dt"].dt.month

# Eliminar filas que no tengan temperatura
df_mex = df_mex.dropna(subset=["AverageTemperature"])

# Definir variables independientes (X) y dependiente (y)
X = df_mex[["Year", "Month", "City"]]
y = df_mex["AverageTemperature"]

# Convertir la ciudad a variables dummy
X_dum = pd.get_dummies(X, columns=["City"], drop_first=False)

# Entrenar el modelo de regresión lineal múltiple
modelo = LinearRegression()
modelo.fit(X_dum, y)

# Guardar las columnas que usó el modelo (para alinear después)
columnas_modelo = X_dum.columns

# ---------------------------
# INTERFAZ DE STREAMLIT
# ---------------------------

st.title("Predicción de temperatura en ciudades de México")
st.write("Aplicación de regresión lineal múltiple usando datos históricos de temperatura.")

# Imagen ad hoc (debes tener un archivo 'mexico.png' en la misma carpeta)
st.image("mexico.png", caption="Temperaturas en México", use_column_width=True)

# Lista de ciudades disponibles (solo México)
ciudades = sorted(df_mex["City"].unique())

# Entradas del usuario
ciudad = st.selectbox("Selecciona la ciudad:", ciudades)
mes = st.number_input("Mes (1-12):", min_value=1, max_value=12, step=1)
anio = st.number_input(
    "Año:",
    min_value=int(df_mex["Year"].min()),
    max_value=int(df_mex["Year"].max()),
    step=1
)

# Botón para predecir
if st.button("Predecir temperatura"):

    # Crear un DataFrame con los datos ingresados
    nueva_fila = pd.DataFrame([[anio, mes, ciudad]], columns=["Year", "Month", "City"])

    # Convertir ciudad a dummies igual que en el entrenamiento
    nueva_dum = pd.get_dummies(nueva_fila, columns=["City"], drop_first=False)

    # Alinear columnas con las del modelo (las que faltan se ponen en 0)
    nueva_dum = nueva_dum.reindex(columns=columnas_modelo, fill_value=0)

    # Hacer la predicción
    prediccion = modelo.predict(nueva_dum)[0]

    # Mostrar resultado
    st.subheader(f"🌡️ Temperatura esperada: {prediccion:.2f} °C")
