import streamlit as st
import pandas as pd

# Cargar archivos CSV
train_df = pd.read_csv('./Project/train.csv')
submission_df = pd.read_csv('./Project/submissionDR.csv')

# Título de la aplicación
st.title('Visualización de la Base de Datos de Retinopatía')

# Mostrar el DataFrame de entrenamiento
st.header('Datos de Entrenamiento')
st.write(train_df)

# Mostrar el DataFrame de sumisión
st.header('Datos de Sumisión')
st.write(submission_df)

# Opciones adicionales de visualización
st.header('Opciones Adicionales')

# Mostrar estadísticas descriptivas
if st.checkbox('Mostrar estadísticas descriptivas de entrenamiento'):
    st.write(train_df.describe())

if st.checkbox('Mostrar estadísticas descriptivas de sumisión'):
    st.write(submission_df.describe())

# Filtros por columnas
st.header('Filtrar Datos')

# Filtrar por columna en el dataset de entrenamiento
column_train = st.selectbox('Selecciona una columna para filtrar en el dataset de entrenamiento', train_df.columns)
value_train = st.text_input(f'Introduce un valor para filtrar en la columna {column_train}', key='value_train')
if value_train:
    filtered_train_df = train_df[train_df[column_train].astype(str).str.contains(value_train, na=False)]
    st.write(filtered_train_df)

# Filtrar por columna en el dataset de sumisión
column_submission = st.selectbox('Selecciona una columna para filtrar en el dataset de sumisión', submission_df.columns)
value_submission = st.text_input(f'Introduce un valor para filtrar en la columna {column_submission}', key='value_submission')
if value_submission:
    filtered_submission_df = submission_df[submission_df[column_submission].astype(str).str.contains(value_submission, na=False)]
    st.write(filtered_submission_df)

# Ejecutar la aplicación en Streamlit
if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title('Visualización de la Base de Datos de Retinopatía')
    st.write('Carga tus archivos CSV para visualizarlos.')
