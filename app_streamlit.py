import streamlit as st
import pandas as pd

titulos_pestanas = ['Página principal', 'Datos de entrenamiento', 'Datos de Submission','Sobre nosotras']
pestaña1, pestaña2, pestaña3, pestaña4 = st.tabs(titulos_pestanas)

with pestaña1:
    st.title('Análisis de Población identificada con DNI de mayor de edad por condición de donante de órganos')
    st.write("Texto sobre donación de órganos")
    st.write("")
    with st.container():
        left_column, right_column = st.columns(2)
        with left_column:
            st.button("Nacional", type="secondary")
            chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
            st.bar_chart(chart_data)
        with right_column:
            st.button("Internacional", type="secondary") 
            chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
            st.bar_chart(chart_data)
            st.caption('Los datos de este gráfico no están actualizados a la fecha actual.')
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
with pestaña6:
    st.title("Sobre nosotras")
    st.image("./nosotras/intro.pdf")
    
# Ejecutar la aplicación en Streamlit
if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title('Visualización de la Base de Datos de Retinopatía')
    st.write('Carga tus archivos CSV para visualizarlos.')
