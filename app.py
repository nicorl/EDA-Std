# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

st.set_page_config(page_title="EDA Interactivo", layout="wide")
st.title("Exploración de datos interactiva")

# ---------------------
# 1. Carga de datos
# ---------------------
st.sidebar.header("Carga de datos")
uploaded_file = st.sidebar.file_uploader("Sube tu CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Vista Previa de los Datos")
    st.dataframe(df.head())

    # ---------------------
    # 2. Información general
    # ---------------------
    st.subheader("Información general")
    st.write("Dimensiones del dataset:", df.shape)
    st.write("Tipos de datos:")
    st.write(df.dtypes)
    st.write("Valores nulos por columna:")
    st.write(df.isnull().sum())

    # ---------------------
    # 3. Estadísticas descriptivas
    # ---------------------
    st.subheader("Estadísticas descriptivas")
    st.write(df.describe(include='all'))

    # ---------------------
    # 4. Visualización interactiva
    # ---------------------
    st.subheader("Visualización de datos")

    numeric_cols = df.select_dtypes('number').columns.tolist()
    categorical_cols = df.select_dtypes('object').columns.tolist()

    st.markdown("### Gráfico Numérico")
    col1, col2 = st.columns(2)
    with col1:
        x = st.selectbox("Eje X", numeric_cols)
    with col2:
        y = st.selectbox("Eje Y", numeric_cols)

    fig, ax = plt.subplots()
    sns.scatterplot(df, x=x, y=y, ax=ax)
    st.pyplot(fig)

    st.markdown("### Distribución de una columna")
    col = st.selectbox("Selecciona columna numérica", numeric_cols, key="dist")
    fig2, ax2 = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax2)
    st.pyplot(fig2)

    if categorical_cols:
        st.markdown("### Conteo por categoría")
        cat_col = st.selectbox("Selecciona columna categórica", categorical_cols, key="cat")
        fig3, ax3 = plt.subplots()
        sns.countplot(data=df, x=cat_col, ax=ax3)
        plt.xticks(rotation=45)
        st.pyplot(fig3)

    # ---------------------
    # 5. Correlación
    # ---------------------
    st.subheader("Mapa de correlación")
    corr = df[numeric_cols].corr()
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)

    # ---------------------
    # 6. Mini modelo predictivo
    # ---------------------
    st.subheader("Mini modelo predictivo (beta)")
    target_col = st.selectbox("Selecciona columna target", numeric_cols, key="target")
    feature_cols = st.multiselect("Selecciona columnas predictoras", numeric_cols, default=[c for c in numeric_cols if c != target_col])

    if st.button("Entrenar modelo"):
        X = df[feature_cols]
        y = df[target_col]

        # Clasificación vs Regresión
        if y.nunique() <= 10 and y.dtype in ['int64','object']:
            model = RandomForestClassifier()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.write(f"Accuracy: {acc:.2f}")
        else:
            model = RandomForestRegressor()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            st.write(f"MSE: {mse:.2f}")

        st.write("Importancia de características")
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        st.bar_chart(importances)

else:
    st.info("Por favor sube un archivo CSV para comenzar la exploración de datos.")
