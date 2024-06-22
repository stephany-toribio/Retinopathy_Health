import streamlit as st
import pandas as pd

titulos_pestanas = ['Página principal', 'Datos de entrenamiento', 'Datos de Submission','Sobre nosotras']
pestaña1, pestaña2, pestaña3, pestaña4 = st.tabs(titulos_pestanas)

with pestaña1:
    st.title('Detección de retinopatías diabéticas')
    st.write("La retinopatía diabética es una complicación ocular grave asociada con la diabetes.")
    st.write("Se produce cuando la diabetes afecta los vasos sanguíneos de la retina, la capa de tejido sensible a la luz en la parte posterior del ojo.")
    st.write("Algunos puntos clave:")
    st.write("- Causas: La retinopatía diabética se debe al daño que la diabetes provoca en los vasos sanguíneos de la retina.")
    st.write("  Este daño puede llevar a hemorragias retinianas y pérdida de visión. Es crucial mantener un control efectivo de la diabetes para prevenir o retrasar esta complicación.")
    st.write("- Etapas:")
    st.write("  - No proliferativa (NPDR): En esta etapa inicial, los vasos sanguíneos de la retina muestran signos de daño.")
    st.write("    Puede progresar a la etapa más grave.")
    st.write("  - Proliferativa (PDR): Se forman nuevos vasos sanguíneos anormales en la retina, lo que puede llevar a hemorragias y cicatrices.")
    st.write("    El diagnóstico temprano y el tratamiento son cruciales para prevenir la pérdida de visión.")
    st.write("- Síntomas: La retinopatía diabética a menudo comienza sin síntomas notorios, por lo que es importante realizar exámenes oculares regulares.")
    st.write("  A medida que avanza, pueden aparecer síntomas en el ojo diabético que no deben ignorarse.")
    st.write("En resumen, detectar y tratar la retinopatía diabética a tiempo es fundamental para preservar la visión.")
    st.write("Si tienes diabetes, consulta con un especialista en oftalmología para evaluar tu salud ocular.")
    st.write("")
    with st.container():
        left_column, right_column = st.columns(2)
        with left_column:
            st.button("train", type="secondary")
            train_df = pd.read_csv('./Project/train.csv')
            st.header('Datos de Entrenamiento')
            st.write(train_df)
            st.title('Visualización de la Base de Datos de Train de Retinopatía')
            st.bar_chart(train_df)
            st.header('Opciones Adicionales')
            st.header('Opciones Adicionales')
            st.title('Visualización de la Base de Datos de Retinopatía')

            if st.checkbox('Mostrar estadísticas descriptivas de entrenamiento'):
                st.write(train_df.describe())

        with right_column:
            st.button("Predicciones", type="secondary") 
            submission_df = pd.read_csv('./Project/submissionDR.csv')
            st.header('Datos predictivos')
            st.write(submission_df)
            st.bar_chart(chart_data)
            st.header('Opciones adicionales')
     # Título de la aplicación
            st.title('Visualización de la Base de Datos de Retinopatía')


            if st.checkbox('Mostrar estadísticas descriptivas de las predicciones'):
                st.write(submission_df.describe())
with pestaña2:
        with st.container():

# Cargar archivos CSV
            train_df = pd.read_csv('./Project/train.csv')



# Filtros por columnas
st.header('Filtrar Datos')
#datos de entrenamiento 
with pestaña2:
        with st.container():
# Filtrar por columna en el dataset de entrenamiento
            column_train = st.selectbox('Selecciona una columna para filtrar en el dataset de entrenamiento', train_df.columns)
            value_train = st.text_input(f'Introduce un valor para filtrar en la columna {column_train}', key='value_train')
            if value_train:
                filtered_train_df = train_df[train_df[column_train].astype(str).str.contains(value_train, na=False)]
                st.write(filtered_train_df)
# temática predicciones 
with pestaña3:
        with st.container():

# Cargar archivos CSV
            submission_df = pd.read_csv('./Project/submissionDR.csv')
# Filtrar por columna en el dataset de submisión
            column_submission = st.selectbox('Selecciona una columna para filtrar en el dataset de submisión', submission_df.columns)
            value_submission = st.text_input(f'Introduce un valor para filtrar en la columna {column_submission}', key='value_submission')
            if value_submission:
                filtered_submission_df = submission_df[submission_df[column_submission].astype(str).str.contains(value_submission, na=False)]
                st.write(filtered_submission_df)
with pestaña4:
    st.title("Sobre nosotras")
    st.image("./nosotras/intro.pdf")
st.link_button("Para más información de click aquí", "https://www.kaggle.com/competitions/upch-intro-ml")
# Ejecutar la aplicación en Streamlit
if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title('Visualización de la Base de Datos de Retinopatía')
    st.write('Carga tus archivos CSV para visualizarlos.')

streamlit run app_streamlit.py


