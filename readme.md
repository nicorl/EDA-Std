EDA Interactivo con Streamlit
=============================

Este proyecto es una **aplicación interactiva de Exploración y Análisis de Datos (EDA)** desarrollada con **Streamlit**. Permite cargar cualquier dataset en formato CSV y explorar estadísticas, gráficos y relaciones entre variables de manera rápida y visual. Además, incluye un mini modelo predictivo opcional para probar regresión o clasificación.

🔹 Funcionalidades
------------------

1.  **Carga de datos**
    
    *   Subida de archivos CSV mediante st.file\_uploader.
        
    *   Vista previa de los datos (head()).
        
2.  **Información general**
    
    *   Dimensiones del dataset.
        
    *   Tipos de datos y valores nulos por columna.
        
3.  **Estadísticas descriptivas**
    
    *   Estadísticas numéricas y categóricas (describe()).
        
    *   Conteo de valores y distribución básica.
        
4.  **Visualización interactiva**
    
    *   Scatterplots entre variables numéricas.
        
    *   Histogramas y densidades.
        
    *   Countplots para columnas categóricas.
        
    *   Heatmap de correlación para variables numéricas.
        
5.  **Mini modelo predictivo**
    
    *   Selección de target y features.
        
    *   Regresión automática con RandomForest si target es numérico.
        
    *   Clasificación automática con RandomForest si target es categórico.
        
    *   Visualización de la importancia de las características.
        


🔹 Requisitos
-------------

Python 3.10+ y las siguientes librerías (especificadas en requirements.txt):

`   streamlit  pandas  numpy  matplotlib  seaborn  scikit-learn   `

Instalación rápida:
`   pip install -r requirements.txt   `

🔹 Cómo usar la app
-------------------

1.  Clona el repositorio:
    

```
git clone https://github.com/nicorl/EDA-Std.git  
cd EDA-Std
```

1.  Instala dependencias:
    

`   pip install -r requirements.txt   `

1.  Ejecuta la app:
    

`   streamlit run app.py   `

1.  Sube un **archivo CSV** en la barra lateral y comienza a explorar los datos.
    

🔹 Requisitos del CSV
---------------------

*   Archivo .csv con **header** en la primera fila.
    
*   **Columnas numéricas** para scatterplots, histogramas y correlación.
    
*   **Columnas categóricas** (strings) para countplots.
    
*   Target numérico o categórico para el mini modelo predictivo.
    
*   Valores nulos permitidos, pero se recomienda limpieza previa.
    
*   Recomendado: menos de 50k filas para mantener la app ágil.
    

**Ejemplo CSV:**

`   Age,Gender,Height,Weight,City,Score  25,M,175,70,Madrid,88  30,F,160,55,Barcelona,92  22,M,180,75,Valencia,85  28,F,165,60,Sevilla,90   `

🔹 Despliegue en Streamlit Community Cloud
------------------------------------------

1.  Sube tu proyecto a GitHub.
    
2.  Conecta tu repositorio en [Streamlit Cloud](https://share.streamlit.io/).
    
3.  Selecciona el archivo app.py y rama main.
    
4.  Clic en **Deploy**.
    
5.  Comparte la URL generada para que otros puedan interactuar con tu app.
    

🔹 Licencia
-----------

Este proyecto está bajo la licencia **MIT**.Puedes usarlo, modificarlo y compartirlo libremente.
